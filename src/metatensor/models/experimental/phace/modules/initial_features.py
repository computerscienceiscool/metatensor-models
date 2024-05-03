import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from .labels_to_device import move_labels_to_device


class InitialFeatures(torch.nn.Module):

    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels

        # labels
        self.components = Labels(
            names=["m"],
            values=torch.tensor([[0]], dtype=torch.int32),
        )
        self.properties = Labels(
            names=["properties"],
            values=torch.arange(self.n_channels, dtype=torch.int32).reshape(self.n_channels, 1),
        )
        self.keys = Labels(
            names=["o3_lambda", "o3_sigma"],
            values=torch.tensor([[0, 1]], dtype=torch.int32),
        )

    def forward(self, samples: Labels, dtype: torch.dtype):
        device = samples.device
        self.components = move_labels_to_device(self.components, device)
        self.properties = move_labels_to_device(self.properties, device)
        self.keys = move_labels_to_device(self.keys, device)

        n_atoms = samples.values.shape[0]
        block = TensorBlock(
            samples=samples,
            values=torch.ones(
                (n_atoms, 1, self.n_channels), dtype=dtype, device=device
            ),
            components=[self.components],
            properties=self.properties,
        )
        return TensorMap(
            keys=self.keys,
            blocks=[block],
        )
