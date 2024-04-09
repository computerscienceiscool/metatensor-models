import torch
from metatensor.torch import Labels
from .physical_LE import get_physical_le_spliner
from .linear import Linear


class RadialBasis(torch.nn.Module):

    def __init__(self, hypers, all_species) -> None:
        super().__init__()

        lengthscales = torch.zeros((max(all_species)+1))
        for species in all_species:
            lengthscales[species] = 0.0
        self.n_max_l, self.spliner = get_physical_le_spliner(hypers["E_max"], hypers["r_cut"], hypers["normalize"])
        self.lengthscales = torch.nn.Parameter(lengthscales)
        
        self.all_species = all_species
        self.n_max_l = list(self.n_max_l)
        self.l_max = len(self.n_max_l) - 1
        self.n_channels = hypers["n_element_channels"]
        
        self.apply_mlp = False
        if hypers["mlp"]:
            self.apply_mlp = True
            self.radial_mlps = torch.nn.ModuleDict({
                str(l): torch.nn.Sequential(
                    Linear(self.n_max_l[l], 4*self.n_max_l[l]*self.n_channels),
                    torch.nn.SiLU(),
                    Linear(4*self.n_max_l[l]*self.n_channels, self.n_max_l[l]*self.n_channels),
                ) for l in range(self.l_max+1)
            })
        else:  # make torchscript happy
            self.radial_mlps = torch.nn.ModuleDict({})

        self.k_max_l = [self.n_max_l[l]*self.n_channels for l in range(self.l_max+1)]

        self.cutoff_radius = hypers["r_cut"]

    def forward(self, r, samples_metadata: Labels):

        a_i = samples_metadata.column("center_type")
        a_j = samples_metadata.column("species_neighbor")
        x = r/(0.1+torch.exp(self.lengthscales[a_i])+torch.exp(self.lengthscales[a_j]))

        capped_x = torch.where(x < 10.0, x, 5.0)
        radial_functions = torch.where(
            x.unsqueeze(1) < 10.0, self.spliner.compute(capped_x), 0.0
        )
        radial_functions = radial_functions * cutoff_function(r, self.cutoff_radius).unsqueeze(1)

        radial_basis = torch.split(radial_functions, self.n_max_l, dim=1)

        if self.apply_mlp:
            radial_basis_after_mlp = []
            for l_string, radial_mlp_l in self.radial_mlps.items():
                l = int(l_string)
                radial_basis_after_mlp.append(radial_mlp_l(radial_basis[l]))
            radial_basis = radial_basis_after_mlp
        else:
            radial_basis = [radial_basis[l].repeat(1, self.n_channels) for l in range(self.l_max+1)]

        return radial_basis


def cutoff_function(r, r_cut: float):
    return torch.where(
        r < r_cut - 1.0,
        torch.ones_like(r),
        0.5*(1.0 + torch.cos(torch.pi*(r - r_cut + 1.0))),
    )

