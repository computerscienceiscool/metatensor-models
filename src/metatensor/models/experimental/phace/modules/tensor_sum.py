from typing import List

import metatensor.torch
import torch


class TensorAdd(torch.nn.Module):
    # tmap_1 MUST be the one with more keys

    def __init__(self, discard: bool = False):
        super().__init__()
        self.discard = discard

    def forward(
        self, tmap_1: metatensor.torch.TensorMap, tmap_2: metatensor.torch.TensorMap
    ):

        new_blocks: List[metatensor.torch.TensorBlock] = []
        for key_1, block_1 in tmap_1.items():
            if key_1 not in tmap_2.keys:
                if self.discard:
                    continue
                else:
                    new_blocks.append(block_1)
            else:
                block_2 = tmap_2.block(key_1)
                new_block = metatensor.torch.TensorBlock(
                    values=block_1.values + block_2.values,
                    samples=block_1.samples,
                    components=block_1.components,
                    properties=block_1.properties,
                )
                new_blocks.append(new_block)

        new_keys = tmap_2.keys if self.discard else tmap_1.keys
        new_map = metatensor.torch.TensorMap(keys=new_keys, blocks=new_blocks)
        return new_map
