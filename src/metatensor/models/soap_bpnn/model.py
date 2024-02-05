from typing import Dict, List, Optional

import metatensor.torch
import rascaline.torch
import torch
from metatensor.torch import Labels, TensorMap
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput, System
from metatensor.torch.learn.nn import Linear as LinearMap
from metatensor.torch.learn.nn import ModuleMap
from omegaconf import OmegaConf

from .. import ARCHITECTURE_CONFIG_PATH
from ..utils.composition import apply_composition_contribution


DEFAULT_HYPERS = OmegaConf.to_container(
    OmegaConf.load(ARCHITECTURE_CONFIG_PATH / "soap_bpnn.yaml")
)

DEFAULT_MODEL_HYPERS = DEFAULT_HYPERS["model"]

ARCHITECTURE_NAME = "soap_bpnn"


class MLPMap(ModuleMap):
    def __init__(self, all_species: List[int], hypers: dict) -> None:
        activation_function_name = hypers["activation_function"]
        if activation_function_name == "SiLU":
            activation_function = torch.nn.SiLU()
        else:
            raise ValueError(
                f"Unsupported activation function: {activation_function_name}"
            )

        # Build a neural network for each species
        nns_per_species = []
        for _ in all_species:
            module_list: List[torch.nn.Module] = []
            for _ in range(hypers["num_hidden_layers"]):
                if len(module_list) == 0:
                    module_list.append(
                        torch.nn.Linear(
                            hypers["input_size"], hypers["num_neurons_per_layer"]
                        )
                    )
                else:
                    module_list.append(
                        torch.nn.Linear(
                            hypers["num_neurons_per_layer"],
                            hypers["num_neurons_per_layer"],
                        )
                    )
                module_list.append(activation_function)

            nns_per_species.append(torch.nn.Sequential(*module_list))
        in_keys = Labels(
            "central_species",
            values=torch.tensor(all_species).reshape(-1, 1),
        )

        # PR TODO check how to solve device issue
        #         before the device was infered in the forward path
        #         but now this does not work
        #         need to now how device is generally determined
        out_properties = [
            Labels(
                names=["properties"],
                values=torch.arange(
                    hypers["num_neurons_per_layer"],
                ).reshape(-1, 1),
            )
            for _ in range(len(in_keys))
        ]
        super().__init__(in_keys, nns_per_species, out_properties)
        self.activation_function = activation_function


class LayerNormMap(ModuleMap):
    def __init__(self, all_species: List[int], n_layer: int) -> None:
        # Initialize a layernorm for each species
        layernorm_per_species = []
        for _ in all_species:
            layernorm_per_species.append(torch.nn.LayerNorm((n_layer,)))

        in_keys = Labels(
            "central_species",
            values=torch.tensor(all_species).reshape(-1, 1),
        )

        # PR COMMENT this removes properties labels information
        #            do you think a flag in ModuleMap that uses
        #            the property labels from the input for the output
        #            is a useful flag for maps that don't change property size
        super().__init__(in_keys, layernorm_per_species)


class Model(torch.nn.Module):
    def __init__(
        self, capabilities: ModelCapabilities, hypers: Dict = DEFAULT_MODEL_HYPERS
    ) -> None:
        super().__init__()
        self.name = ARCHITECTURE_NAME

        # Check capabilities
        for output in capabilities.outputs.values():
            if output.quantity != "energy":
                raise ValueError(
                    "SOAP-BPNN only supports energy-like outputs, "
                    f"but a {output.quantity} was provided"
                )
            if output.per_atom:
                raise ValueError(
                    "SOAP-BPNN only supports per-structure outputs, "
                    "but a per-atom output was provided"
                )

        self.capabilities = capabilities
        self.all_species = capabilities.species
        self.hypers = hypers

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

        self.soap_calculator = rascaline.torch.SoapPowerSpectrum(**hypers["soap"])
        soap_size = (
            len(self.all_species) ** 2
            * hypers["soap"]["max_radial"] ** 2
            * (hypers["soap"]["max_angular"] + 1)
        )

        self.layernorm = LayerNormMap(self.all_species, soap_size)

        hypers_bpnn = hypers["bpnn"]
        hypers_bpnn["input_size"] = soap_size

        self.bpnn = MLPMap(self.all_species, hypers_bpnn)
        self.neighbor_species_1_labels = Labels(
            names=["species_neighbor_1"],
            values=torch.tensor(self.all_species).reshape(-1, 1),
        )
        self.neighbor_species_2_labels = Labels(
            names=["species_neighbor_2"],
            values=torch.tensor(self.all_species).reshape(-1, 1),
        )

        if hypers_bpnn["num_hidden_layers"] == 0:
            n_inputs_last_layer = hypers_bpnn["input_size"]
        else:
            n_inputs_last_layer = hypers_bpnn["num_neurons_per_layer"]

        self.last_layers = torch.nn.ModuleDict(
            {
                output_name: LinearMap(
                    Labels(
                        "central_species",
                        values=torch.tensor(self.all_species).reshape(-1, 1),
                    ),
                    in_features=n_inputs_last_layer,
                    out_features=1,
                )
                for output_name in capabilities.outputs.keys()
            }
        )

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if selected_atoms is not None:
            raise NotImplementedError("SOAP-BPNN does not support selected atoms.")

        for requested_output in outputs.keys():
            if requested_output not in self.capabilities.outputs.keys():
                raise ValueError(
                    f"Requested output {requested_output} is not within "
                    "the model's capabilities."
                )

        soap_features = self.soap_calculator(systems)

        device = soap_features.block(0).values.device
        soap_features = soap_features.keys_to_properties(
            self.neighbor_species_1_labels.to(device)
        )
        soap_features = soap_features.keys_to_properties(
            self.neighbor_species_2_labels.to(device)
        )

        soap_features = self.layernorm(soap_features)

        hidden_features = self.bpnn(soap_features)

        atomic_energies: Dict[str, TensorMap] = {}
        for output_name, output_layer in self.last_layers.items():
            if output_name in outputs:
                atomic_energies[output_name] = apply_composition_contribution(
                    output_layer(hidden_features),
                    self.composition_weights[self.output_to_index[output_name]],
                )

        # Sum the atomic energies coming from the BPNN to get the total energy
        total_energies: Dict[str, TensorMap] = {}
        for output_name, atomic_energy in atomic_energies.items():
            atomic_energy = atomic_energy.keys_to_samples("species_center")
            total_energies[output_name] = metatensor.torch.sum_over_samples(
                atomic_energy, ["center", "species_center"]
            )
            # Change the energy label from _ to (0, 1):
            total_energies[output_name] = TensorMap(
                keys=Labels(
                    names=["lambda", "sigma"],
                    values=torch.tensor([[0, 1]]),
                ),
                blocks=[total_energies[output_name].block()],
            )

        return total_energies

    def set_composition_weights(
        self, output_name: str, input_composition_weights: torch.Tensor
    ) -> None:
        """Set the composition weights for a given output."""
        # all species that are not present retain their weight of zero
        self.composition_weights[self.output_to_index[output_name]][
            self.all_species
        ] = input_composition_weights
