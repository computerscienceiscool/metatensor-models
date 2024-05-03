import sphericart.torch
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from .labels_to_device import move_labels_to_device


class Precomputer(torch.nn.Module):
    def __init__(self, l_max, normalize=True):
        super().__init__()
        self.spherical_harmonics_split_list = [(2 * l + 1) for l in range(l_max + 1)]
        self.normalize = normalize
        self.spherical_harmonics_calculator = sphericart.torch.SphericalHarmonics(
            l_max, normalized=True
        )

        # labels
        self.spherical_harmonics_components = [Labels(
            names=["m"],
            values=torch.arange(
                start=-l, end=l + 1, dtype=torch.int32
            ).reshape(2 * l + 1, 1),
        ) for l in range(l_max + 1)]
        self.spherical_harmonics_properties = Labels(
            names=["properties"],
            values=torch.tensor([[0]], dtype=torch.int32),
        )
        self.spherical_harmonics_keys = Labels(
            names=["o3_lambda", "o3_sigma"],
            values=torch.tensor([[l, 1] for l in range(l_max + 1)], dtype=torch.int32),
        )
        self.r_properties = Labels(
            names=["properties"],
            values=torch.tensor([[0]], dtype=torch.int32),
        )

    def forward(
        self,
        positions,
        cells,
        species,
        cell_shifts,
        centers,
        pairs,
        structure_centers,
        structure_pairs,
        structure_offsets,
    ):
        device = positions.device
        self.spherical_harmonics_components = [move_labels_to_device(s, device) for s in self.spherical_harmonics_components]
        self.spherical_harmonics_properties = move_labels_to_device(self.spherical_harmonics_properties, device)
        self.spherical_harmonics_keys = move_labels_to_device(self.spherical_harmonics_keys, device)
        self.r_properties = move_labels_to_device(self.r_properties, device)

        cartesian_vectors = get_cartesian_vectors(
            positions,
            cells,
            species,
            cell_shifts,
            centers,
            pairs,
            structure_centers,
            structure_pairs,
            structure_offsets,
        )

        bare_cartesian_vectors = cartesian_vectors.values.squeeze(dim=-1)
        r = torch.sqrt((bare_cartesian_vectors**2).sum(dim=-1))

        spherical_harmonics = self.spherical_harmonics_calculator.compute(
            bare_cartesian_vectors
        )  # Get the spherical harmonics
        if self.normalize:
            spherical_harmonics = spherical_harmonics * (4.0 * torch.pi) ** (
                0.5
            )  # normalize them
        spherical_harmonics = torch.split(
            spherical_harmonics, self.spherical_harmonics_split_list, dim=1
        )  # Split them into l chunks

        spherical_harmonics_blocks = [
            TensorBlock(
                values=spherical_harmonics_l.unsqueeze(-1),
                samples=cartesian_vectors.samples,
                components=[spherical_harmonics_components_l],
                properties=self.spherical_harmonics_properties,
            )
            for spherical_harmonics_components_l, spherical_harmonics_l in zip(self.spherical_harmonics_components, spherical_harmonics)
        ]
        spherical_harmonics_map = TensorMap(
            keys=self.spherical_harmonics_keys,
            blocks=spherical_harmonics_blocks,
        )

        r_block = TensorBlock(
            values=r.unsqueeze(-1),
            samples=cartesian_vectors.samples,
            components=[],
            properties=self.r_properties,
        )

        return r_block, spherical_harmonics_map


def get_cartesian_vectors(
    positions,
    cells,
    species,
    cell_shifts,
    centers,
    pairs,
    structure_centers,
    structure_pairs,
    structure_offsets,
):
    """
    Wraps direction vectors into TensorBlock object with metadata information
    """

    # calculate interatomic vectors
    pairs_offsets = structure_offsets[structure_pairs]
    shifted_pairs = pairs_offsets[:, None] + pairs
    shifted_pairs_i = shifted_pairs[:, 0]
    shifted_pairs_j = shifted_pairs[:, 1]
    direction_vectors = (
        positions[shifted_pairs_j]
        - positions[shifted_pairs_i]
        + torch.einsum(
            "ab, abc -> ac", cell_shifts.to(cells.dtype), cells[structure_pairs]
        )
    )

    # find associated metadata
    pairs_i = pairs[:, 0]
    pairs_j = pairs[:, 1]
    labels = torch.stack(
        [
            structure_pairs,
            pairs_i,
            pairs_j,
            species[shifted_pairs_i],
            species[shifted_pairs_j],
            cell_shifts[:, 0],
            cell_shifts[:, 1],
            cell_shifts[:, 2],
        ],
        dim=-1,
    )

    # build TensorBlock
    block = TensorBlock(
        values=direction_vectors.unsqueeze(dim=-1),
        samples=Labels(
            names=[
                "system",
                "atom",
                "neighbor",
                "center_type",
                "species_neighbor",
                "cell_x",
                "cell_y",
                "cell_z",
            ],
            values=labels,
        ),
        components=[
            Labels(
                names=["cartesian_dimension"],
                values=torch.tensor([-1, 0, 1], dtype=torch.int32, device=direction_vectors.device).reshape((-1, 1))
            )
        ],
        properties=Labels(
            names=["_"],
            values=torch.tensor(
                [[0]], dtype=torch.int32, device=direction_vectors.device
            ),
        ),
    )

    return block
