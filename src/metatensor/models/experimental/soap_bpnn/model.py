from typing import Dict, List, Optional

import metatensor.torch
import rascaline.torch
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput, System
from omegaconf import OmegaConf

from ... import ARCHITECTURE_CONFIG_PATH
from ...utils.composition import apply_composition_contribution


ARCHITECTURE_NAME = "experimental.soap_bpnn"
DEFAULT_HYPERS = OmegaConf.to_container(
    OmegaConf.load(ARCHITECTURE_CONFIG_PATH / f"{ARCHITECTURE_NAME}.yaml")
)
DEFAULT_MODEL_HYPERS = DEFAULT_HYPERS["model"]


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: TensorMap) -> TensorMap:
        return x


class MLPMap(torch.nn.Module):
    def __init__(self, all_species: List[int], hypers: dict) -> None:
        super().__init__()

        self.activation_function = torch.nn.SiLU()

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
                module_list.append(self.activation_function)

            nns_per_species.append(torch.nn.Sequential(*module_list))

        # Create a module dict to store the neural networks
        self.layers = torch.nn.ModuleDict(
            {str(species): nn for species, nn in zip(all_species, nns_per_species)}
        )

    def forward(self, features: TensorMap) -> TensorMap:
        # Create a list of the blocks that are present in the features:
        present_blocks = [
            int(features.keys.entry(i).values.item())
            for i in range(features.keys.values.shape[0])
        ]

        new_keys: List[int] = []
        new_blocks: List[TensorBlock] = []
        for species_str, network in self.layers.items():
            species = int(species_str)
            if species in present_blocks:
                new_keys.append(species)
                block = features.block({"center_type": species})
                output_values = network(block.values)
                new_blocks.append(
                    TensorBlock(
                        values=output_values,
                        samples=block.samples,
                        components=block.components,
                        # cannot use Labels.range() here because of torch.jit.save
                        properties=Labels(
                            names=["properties"],
                            values=torch.arange(
                                output_values.shape[1], device=output_values.device
                            ).reshape(-1, 1),
                        ),
                    )
                )
        new_keys_labels = Labels(
            names=["center_type"],
            values=torch.tensor(new_keys, device=new_blocks[0].values.device).reshape(
                -1, 1
            ),
        )

        return TensorMap(keys=new_keys_labels, blocks=new_blocks)


class LayerNormMap(torch.nn.Module):
    def __init__(self, all_species: List[int], n_layer: int) -> None:
        super().__init__()

        # Initialize a layernorm for each species
        layernorm_per_species = []
        for _ in all_species:
            layernorm_per_species.append(torch.nn.LayerNorm((n_layer,)))

        # Create a module dict to store the neural networks
        self.layernorms = torch.nn.ModuleDict(
            {
                str(species): layer
                for species, layer in zip(all_species, layernorm_per_species)
            }
        )

    def forward(self, features: TensorMap) -> TensorMap:
        # Create a list of the blocks that are present in the features:
        present_blocks = [
            int(features.keys.entry(i).values.item())
            for i in range(features.keys.values.shape[0])
        ]

        new_keys: List[int] = []
        new_blocks: List[TensorBlock] = []
        for species_str, layer in self.layernorms.items():
            species = int(species_str)
            if species in present_blocks:
                new_keys.append(species)
                block = features.block({"center_type": species})
                output_values = layer(block.values)
                new_blocks.append(
                    TensorBlock(
                        values=output_values,
                        samples=block.samples,
                        components=block.components,
                        properties=block.properties,
                    )
                )
        new_keys_labels = Labels(
            names=["center_type"],
            values=torch.tensor(new_keys, device=new_blocks[0].values.device).reshape(
                -1, 1
            ),
        )

        return TensorMap(keys=new_keys_labels, blocks=new_blocks)


class LinearMap(torch.nn.Module):
    def __init__(self, all_species: List[int], n_inputs: int) -> None:
        super().__init__()

        # Build a neural network for each species
        layer_per_species = []
        for _ in all_species:
            layer_per_species.append(torch.nn.Linear(n_inputs, 1))

        # Create a module dict to store the neural networks
        self.layers = torch.nn.ModuleDict(
            {
                str(species): layer
                for species, layer in zip(all_species, layer_per_species)
            }
        )

    def forward(self, features: TensorMap) -> TensorMap:
        # Create a list of the blocks that are present in the features:
        present_blocks = [
            int(features.keys.entry(i).values.item())
            for i in range(features.keys.values.shape[0])
        ]

        new_keys: List[int] = []
        new_blocks: List[TensorBlock] = []
        for species_str, layer in self.layers.items():
            species = int(species_str)
            if species in present_blocks:
                new_keys.append(species)
                block = features.block({"center_type": species})
                output_values = layer(block.values)
                new_blocks.append(
                    TensorBlock(
                        values=output_values,
                        samples=block.samples,
                        components=block.components,
                        properties=Labels(
                            names=["energy"],
                            values=torch.zeros(
                                (1, 1), dtype=torch.int32, device=block.values.device
                            ),
                        ),
                    )
                )
        new_keys_labels = Labels(
            names=["center_type"],
            values=torch.tensor(new_keys, device=new_blocks[0].values.device).reshape(
                -1, 1
            ),
        )

        return TensorMap(keys=new_keys_labels, blocks=new_blocks)


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
                    "SOAP-BPNN only supports per-system outputs, "
                    "but a per-atom output was provided"
                )

        self.capabilities = capabilities
        self.all_species = capabilities.atomic_types
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

        self.soap_calculator = rascaline.torch.SoapPowerSpectrum(
            radial_basis={"Gto": {}}, **hypers["soap"]
        )
        soap_size = (
            (len(self.all_species) * (len(self.all_species) + 1) // 2)
            * hypers["soap"]["max_radial"] ** 2
            * (hypers["soap"]["max_angular"] + 1)
        )

        hypers_bpnn = hypers["bpnn"]
        hypers_bpnn["input_size"] = soap_size

        if hypers_bpnn["layernorm"]:
            self.layernorm = LayerNormMap(self.all_species, soap_size)
        else:
            self.layernorm = Identity()

        self.bpnn = MLPMap(self.all_species, hypers_bpnn)

        self.neighbor_species_labels = Labels(
            names=["neighbor_1_type", "neighbor_2_type"],
            values=torch.combinations(
                torch.tensor(self.all_species, dtype=torch.int),
                with_replacement=True,
            ),
        )

        if hypers_bpnn["num_hidden_layers"] == 0:
            n_inputs_last_layer = hypers_bpnn["input_size"]
        else:
            n_inputs_last_layer = hypers_bpnn["num_neurons_per_layer"]

        self.last_layers = torch.nn.ModuleDict(
            {
                output_name: LinearMap(self.all_species, n_inputs_last_layer)
                for output_name in capabilities.outputs.keys()
            }
        )

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:

        soap_features = self.soap_calculator(systems, selected_samples=selected_atoms)

        device = soap_features.block(0).values.device
        soap_features = soap_features.keys_to_properties(
            self.neighbor_species_labels.to(device)
        )

        soap_features = self.layernorm(soap_features)

        hidden_features = self.bpnn(soap_features)

        atomic_energies: Dict[str, TensorMap] = {}
        for output_name, output_layer in self.last_layers.items():
            if output_name in outputs:
                atomic_energies[output_name] = apply_composition_contribution(
                    output_layer(hidden_features),
                    self.composition_weights[  # type: ignore
                        self.output_to_index[output_name]
                    ],
                )

        # Sum the atomic energies coming from the BPNN to get the total energy
        total_energies: Dict[str, TensorMap] = {}
        for output_name, atomic_energy in atomic_energies.items():
            atomic_energy = atomic_energy.keys_to_samples("center_type")
            total_energies[output_name] = metatensor.torch.sum_over_samples(
                atomic_energy, ["atom", "center_type"]
            )

        return total_energies

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
        hypers_bpnn = self.hypers["bpnn"]
        if hypers_bpnn["num_hidden_layers"] == 0:
            n_inputs_last_layer = hypers_bpnn["input_size"]
        else:
            n_inputs_last_layer = hypers_bpnn["num_neurons_per_layer"]
        self.last_layers[output_name] = LinearMap(self.all_species, n_inputs_last_layer)
