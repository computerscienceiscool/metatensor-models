import torch 
from metatensor.torch import Labels, TensorBlock, TensorMap


class InitialFeatures(torch.nn.Module):

    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels

    def forward(self, structures, centers, species, dtype: torch.dtype):
        n_atoms = structures.shape[0]
        block = TensorBlock(
            values=torch.ones((n_atoms, 1, self.n_channels), dtype=dtype, device=structures.device),
            samples=Labels(
                names=["system", "atom", "center_type"],
                values=torch.stack([structures, centers, species], dim=1)
            ).to(device=structures.device),
            components=[
                Labels(
                    names=["m"],
                    values=torch.tensor([[0]], device=structures.device),
                ).to(device=structures.device)
            ],
            properties=Labels.range("properties", self.n_channels).to(device=structures.device)
        )
        return TensorMap(
            keys=Labels(
                names=["o3_lambda", "o3_sigma"],
                values=torch.tensor([[0, 1]], device=structures.device)
            ),
            blocks=[block]
        )
