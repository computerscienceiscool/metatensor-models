from typing import Dict, List

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from .labels_to_device import move_labels_to_device

from .radial_basis import RadialBasis


class InvariantMessagePasser(torch.nn.Module):

    def __init__(
        self, hypers: Dict, all_species: List[int]
    ) -> None:
        super().__init__()

        self.all_species = all_species
        hypers["radial_basis"]["r_cut"] = hypers["cutoff_radius"]
        hypers["radial_basis"]["normalize"] = hypers["normalize"]
        hypers["radial_basis"]["n_element_channels"] = hypers["n_element_channels"]
        self.radial_basis_calculator = RadialBasis(
            hypers["radial_basis"], all_species
        )
        self.n_max_l = self.radial_basis_calculator.n_max_l
        self.k_max_l = [hypers["n_element_channels"] * n_max for n_max in self.n_max_l]
        self.l_max = len(self.n_max_l) - 1

        self.properties = [Labels(
            names=["properties"],
            values=torch.arange(
                k_max,
                dtype=torch.int,
            ).reshape(k_max, 1),
        ) for k_max in self.k_max_l]

    def forward(
        self,
        r: TensorBlock,
        sh: TensorMap,
        centers,
        neighbors,
        n_atoms: int,
        initial_center_embedding: TensorMap,
        samples: Labels,
    ) -> TensorMap:
        device = r.values.device
        self.properties = [move_labels_to_device(p, device) for p in self.properties]

        radial_basis = self.radial_basis_calculator(r.values.squeeze(-1), r.samples)

        blocks: List[TensorBlock] = []
        for l in range(self.l_max + 1):
            spherical_harmonics_l = sh.block({"o3_lambda": l}).values
            radial_basis_l = radial_basis[l]
            densities_l = torch.zeros(
                (n_atoms, spherical_harmonics_l.shape[1], radial_basis_l.shape[1]),
                device=device,
                dtype=radial_basis_l.dtype,
            )
            densities_l.index_add_(
                dim=0,
                index=centers,
                source=spherical_harmonics_l
                * radial_basis_l.unsqueeze(1)
                * initial_center_embedding.block().values[neighbors][
                    :, :, : radial_basis_l.shape[1]
                ],
            )
            blocks.append(
                TensorBlock(
                    values=densities_l,
                    samples=samples,
                    components=sh.block({"o3_lambda": l}).components,
                    properties=self.properties[l],
                )
            )

        return TensorMap(
            keys=sh.keys,
            blocks=blocks,
        )
