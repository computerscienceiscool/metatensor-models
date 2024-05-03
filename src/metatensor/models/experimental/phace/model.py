from typing import Dict, List, Optional

import metatensor.torch
import torch
from metatensor.torch import Labels, TensorMap
from metatensor.torch.atomistic import (
    ModelCapabilities,
    ModelOutput,
    NeighborsListOptions,
    System,
)
from omegaconf import OmegaConf

from ... import ARCHITECTURE_CONFIG_PATH
from ...utils.composition import apply_composition_contribution_samples
from ...utils.scaling import apply_scaling
from .modules.cg_iterator import CGIterator, get_cg_coefficients
from .modules.initial_features import InitialFeatures
from .utils import systems_to_batch
from .modules.center_embedding import CenterEmbedding
from .modules.invariant_message_passing import InvariantMessagePasser
from .modules.linear_map import LinearMap
from .modules.precomputations import Precomputer


ARCHITECTURE_NAME = "experimental.phace"
DEFAULT_HYPERS = OmegaConf.to_container(
    OmegaConf.load(ARCHITECTURE_CONFIG_PATH / f"{ARCHITECTURE_NAME}.yaml")
)
DEFAULT_MODEL_HYPERS = DEFAULT_HYPERS["model"]


class Model(torch.nn.Module):

    def __init__(
        self, capabilities: ModelCapabilities, hypers: Dict = DEFAULT_MODEL_HYPERS
    ) -> None:
        super().__init__()
        self.capabilities = capabilities
        hypers["normalize"] = False
        all_species = capabilities.atomic_types
        n_channels = hypers["n_element_channels"]
        self.all_species = all_species
        self.invariant_message_passer = InvariantMessagePasser(
            hypers, self.all_species
        )
        n_max = self.invariant_message_passer.n_max_l
        self.l_max = len(n_max) - 1
        self.k_max_l = [n_channels * n_max[l] for l in range(self.l_max + 1)]

        cgs = get_cg_coefficients(self.l_max, sparse=True)
        cgs = {
            str(l1) + "_" + str(l2): tensors
            for (l1, l2), tensors in cgs._cgs.items()
        }
        irreps_spex = [(l, 1) for l in range(self.l_max + 1)]
        self.precomputer = Precomputer(self.l_max, normalize=True)

        self.cg_iterator = CGIterator(
            self.k_max_l,
            5,
            cgs,
            irreps_in=irreps_spex,
            exponential_algorithm=False,
            requested_LS_string="0_1",
        )

        self.initial_features = InitialFeatures(self.k_max_l[0])

        self.element_embedding = CenterEmbedding(
            all_species, n_channels
        )  # This will break for non-invariants! self.k_max_l and somehow extract inside

        self.last_layers = torch.nn.ModuleDict(
            {
                output_name: LinearMap(self.k_max_l[0])
                for output_name in capabilities.outputs.keys()
            }
        )

        # creates a composition weight tensor that can be directly indexed by species,
        # this can be left as a tensor of zero or set from the outside using
        # set_composition_weights (recommended for better accuracy)
        n_outputs = len(capabilities.outputs)
        self.register_buffer(
            "composition_weights", torch.zeros((n_outputs, max(self.all_species) + 1))
        )
        # buffers cannot be indexed by strings (torchscript), so we create a single
        # tensor for all output. Due to this, we need to slice the tensor when we use
        # it and use the output name to select the correct slice via a dictionary
        self.output_to_index = {
            output_name: i for i, output_name in enumerate(capabilities.outputs.keys())
        }

        # we also register a buffer for the shifts:
        # these are meant to be modified from outside
        self.register_buffer("scalings", torch.ones((n_outputs,)))

        self.cutoff_radius = hypers["cutoff_radius"]
        self.name = "phace"

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if selected_atoms is not None:
            raise NotImplementedError("PhACE does not support selected atoms.")

        options = self.requested_neighbors_lists()[0]
        structures = systems_to_batch(systems, options)

        n_atoms = len(structures["positions"])

        r, sh = self.precomputer(
            positions=structures["positions"],
            cells=structures["cells"],
            species=structures["species"],
            cell_shifts=structures["cell_shifts"],
            centers=structures["centers"],
            pairs=structures["pairs"],
            structure_centers=structures["structure_centers"],
            structure_pairs=structures["structure_pairs"],
            structure_offsets=structures["structure_offsets"],
        )

        samples_values = torch.stack(
            (
                structures["structure_centers"],
                structures["centers"],
                structures["species"],
            ),
            dim=1,
        )
        samples = metatensor.torch.Labels(
            names=["system", "atom", "center_type"],
            values=samples_values,
        )

        initial_features = self.initial_features(
            samples,
            structures["positions"].dtype,
        )
        initial_element_embedding = self.element_embedding(initial_features)

        spherical_expansion = self.invariant_message_passer(
            r,
            sh,
            structures["structure_offsets"][structures["structure_pairs"]]
            + structures["pairs"][:, 0],
            structures["structure_offsets"][structures["structure_pairs"]]
            + structures["pairs"][:, 1],
            n_atoms,
            initial_element_embedding,
            samples,
        )

        nu4_features = self.cg_iterator(spherical_expansion)
        hidden_features = self.element_embedding(nu4_features)

        atomic_energies: Dict[str, TensorMap] = {}
        for output_name, output_layer in self.last_layers.items():
            if output_name in outputs:
                atomic_energies[output_name] = apply_composition_contribution_samples(
                    apply_scaling(
                        output_layer(hidden_features),
                        self.scalings[self.output_to_index[output_name]].item(),
                    ),
                    self.composition_weights[  # type: ignore
                        self.output_to_index[output_name]
                    ],
                )

        # Sum the atomic energies coming from the BPNN to get the total energy
        total_energies: Dict[str, TensorMap] = {}
        for output_name, atomic_energy in atomic_energies.items():
            total_energies[output_name] = metatensor.torch.sum_over_samples(
                atomic_energy, ["atom", "center_type"]
            )

        return total_energies

    @torch.jit.export
    def set_composition_weights(
        self,
        output_name: str,
        input_composition_weights: torch.Tensor,
        species: List[int],
    ) -> None:
        """Set the composition weights for a given output."""
        # all species that are not present retain their weight of zero
        self.composition_weights[self.output_to_index[output_name]][  # type: ignore
            species
        ] = input_composition_weights.to(
            dtype=self.composition_weights.dtype,  # type: ignore
            device=self.composition_weights.device,  # type: ignore
        )

    @torch.jit.export
    def add_output(self, output_name: str) -> None:
        """Add a new output to the model."""
        # add a new row to the composition weights tensor
        self.composition_weights = torch.cat(
            [
                self.composition_weights,  # type: ignore
                torch.zeros(
                    1,
                    self.composition_weights.shape[1],  # type: ignore
                    dtype=self.composition_weights.dtype,  # type: ignore
                    device=self.composition_weights.device,  # type: ignore
                ),
            ]
        )  # type: ignore
        self.output_to_index[output_name] = len(self.output_to_index)
        # add a new linear layer to the last layers
        #### TODO: THIS IS SO WRONG!!!
        # hypers_bpnn = self.hypers["bpnn"]
        # if hypers_bpnn["num_hidden_layers"] == 0:
        #     n_inputs_last_layer = hypers_bpnn["input_size"]
        # else:
        #     n_inputs_last_layer = hypers_bpnn["num_neurons_per_layer"]
        # self.last_layers[output_name] = LinearMap(self.all_species, n_inputs_last_layer)

    @torch.jit.export
    def requested_neighbors_lists(
        self,
    ) -> List[NeighborsListOptions]:
        return [
            NeighborsListOptions(
                cutoff=self.cutoff_radius,
                full_list=True,
            )
        ]
