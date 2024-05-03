from typing import List

import metatensor.torch
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from .linear import Linear


class CenterEmbedding(torch.nn.Module):

    def __init__(self, all_species, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.species_center_labels = Labels(
            names=["center_type"],
            values=torch.tensor(all_species, dtype=torch.int).unsqueeze(1),
        )
        # TODO: the normalization is wrong here
        self.embeddings = Linear(len(all_species), n_channels)

        self.register_buffer("species_to_species_index", torch.zeros(max(all_species) + 1, dtype=torch.int))
        self.species_to_species_index[all_species] = torch.arange(len(all_species), dtype=torch.int)

    def forward(self, equivariants: TensorMap):

        samples = equivariants.block(0).samples.column("center_type")
        channel_weights = self.embeddings.linear_layer.weight.T[self.species_to_species_index[samples]]

        blocks: List[TensorBlock] = []

        for _, block in equivariants.items():
            assert block.values.shape[-1] % self.n_channels == 0
            n_repeats = block.values.shape[-1] // self.n_channels
            new_block_values = block.values * channel_weights.repeat(
                1, n_repeats
            ).unsqueeze(1)
            blocks.append(
                TensorBlock(
                    values=new_block_values,
                    samples=block.samples,
                    components=block.components,
                    properties=block.properties,
                )
            )

        return TensorMap(
            keys=equivariants.keys,
            blocks=blocks,
        )
