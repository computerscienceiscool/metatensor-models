import torch
from .normalize import Linear, Normalizer
import metatensor.torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from typing import List


# class CenterEmbedding(torch.nn.Module):

#     def __init__(self, irreps, all_species, n_channels):
#         super().__init__()
#         self.species_center_labels = Labels(
#             names = ["center_type"],
#             values = torch.tensor(all_species, dtype=torch.int).unsqueeze(1)
#         )
#         ai_embeddings = {}
#         for L, S in irreps:
#             ai_embeddings[str(L) + "_" + str(S)] = torch.nn.Sequential(
#                 Linear(len(all_species), n_channels),
#                 Normalizer([0, 1])
#             )
#         self.ai_embeddings = torch.nn.ModuleDict(ai_embeddings)

#     def forward(self, equivariants: TensorMap):

#         keys: List[List[int]] = []  # Can perhaps be precomputed or reused from equivariants?
#         blocks: List[TensorBlock] = []

#         for LS_string, LS_embedding in self.ai_embeddings.items():
#             split_LS_string = LS_string.split("_")
#             L, S = int(split_LS_string[0]), int(split_LS_string[1])
#             block = equivariants.block({"o3_lambda": L, "o3_sigma": S})
#             samples = block.samples
#             one_hot_ai = metatensor.torch.one_hot(
#                 samples,
#                 self.species_center_labels
#             )  # Perhaps can be done only once outside the loop
#             ai_channel_weights = LS_embedding(one_hot_ai.to(dtype=block.values.dtype).to(device=block.values.device))
#             new_block_values = block.values * ai_channel_weights.unsqueeze(1)
#             keys.append([L, S])
#             blocks.append(
#                 TensorBlock(
#                     values=new_block_values,
#                     samples=block.samples,
#                     components=block.components,
#                     properties=block.properties
#                 )
#             )

#         return TensorMap(
#             keys=Labels(
#                 names=["o3_lambda", "o3_sigma"],
#                 values=torch.tensor(keys).to(equivariants.keys.values.device)
#             ),
#             blocks=blocks
#         )


class CenterEmbedding(torch.nn.Module):

    def __init__(self, all_species, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.species_center_labels = Labels(
            names = ["center_type"],
            values = torch.tensor(all_species, dtype=torch.int).unsqueeze(1)
        )
        # TODO: the normalization is wrong here
        self.embeddings = torch.nn.Sequential(
            Linear(len(all_species), n_channels),
            Normalizer([0, 1])
        )

    def forward(self, equivariants: TensorMap):

        keys: List[torch.Tensor] = []  # Can perhaps be precomputed or reused from equivariants?
        blocks: List[TensorBlock] = []

        for key, block in equivariants.items():
            assert block.values.shape[-1] % self.n_channels == 0
            n_repeats = block.values.shape[-1] // self.n_channels
            samples = block.samples
            one_hot_ai = metatensor.torch.one_hot(
                samples,
                self.species_center_labels
            )  # TODO: perhaps can be done only once outside the loop
            channel_weights = self.embeddings(one_hot_ai.to(dtype=block.values.dtype).to(device=block.values.device))
            new_block_values = block.values * channel_weights.repeat(1, n_repeats).unsqueeze(1)
            keys.append(key.values)
            blocks.append(
                TensorBlock(
                    values=new_block_values,
                    samples=block.samples,
                    components=block.components,
                    properties=block.properties
                )
            )

        return TensorMap(
            keys=Labels(
                names=["o3_lambda", "o3_sigma"],
                values=torch.stack(keys).to(equivariants.keys.values.device)
            ),
            blocks=blocks
        )
