import copy
import random

import ase.io
import numpy as np
import rascaline.torch
import torch
from metatensor.learn.data import Dataset
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput
from omegaconf import OmegaConf

from metatensor.models.sparse_gap import DEFAULT_HYPERS, Model, train
from metatensor.models.utils.data import get_all_species
from metatensor.models.utils.data.readers import read_structures, read_targets
from pathlib import Path

from . import DATASET_ETHANOL_PATH, DATASET_PATH


# reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def test_regression_init():
    """Perform a regression test on the model at initialization"""

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        species=[1, 6, 7, 8],
        outputs={
            "U0": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
    )
    Model(capabilities, DEFAULT_HYPERS["model"]).to(torch.float64)


def test_regression_train_and_invariance():
    """Perform a regression test on the model when trained for 2 epoch on a small
    dataset.  We perform also the invariance test here because one needs a trained model
    for this.
    """

    structures = read_structures(DATASET_PATH)
    # PR COMMENT this is a temporary hack until kernel is properly implemented that can
    #            deal with tensor maps with different species pairs
    # for structure in structures:
    #    structure.species = torch.ones(len(structure.species), dtype=torch.int32)

    conf = {
        "U0": {
            "quantity": "energy",
            "read_from": DATASET_PATH,
            "file_format": ".xyz",
            "key": "U0",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets = read_targets(OmegaConf.create(conf))
    dataset = Dataset(structure=structures, U0=targets["U0"])

    hypers = DEFAULT_HYPERS.copy()
    hypers["training"]["num_epochs"] = 2

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        species=get_all_species(dataset),
        outputs={
            "U0": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
    )
    sparse_gap = train([dataset], [dataset], capabilities, hypers)

    # Predict on the first five structures
    output = sparse_gap(structures[:5], {"U0": sparse_gap.capabilities.outputs["U0"]})

    expected_output = torch.tensor(
        [[-40.5891], [-56.7122], [-76.4146], [-77.3364], [-93.4905]]
    )

    assert torch.allclose(output["U0"].block().values, expected_output, rtol=0.3)

    # Tests that the model is rotationally invariant
    structure = ase.io.read(DATASET_PATH)
    # PR COMMENT this is a temporary hack until kernel is properly implemented that can
    #            deal with tensor maps with different species pairs
    structure.numbers = np.ones(len(structure.numbers))

    original_structure = copy.deepcopy(structure)
    structure.rotate(48, "y")

    original_output = sparse_gap(
        [rascaline.torch.systems_to_torch(original_structure)],
        {"U0": sparse_gap.capabilities.outputs["U0"]},
    )
    rotated_output = sparse_gap(
        [rascaline.torch.systems_to_torch(structure)],
        {"U0": sparse_gap.capabilities.outputs["U0"]},
    )

    assert torch.allclose(
        original_output["U0"].block().values,
        rotated_output["U0"].block().values,
    )


def test_ethanol_regression_train_and_invariance():
    """Perform a regression test on the model when trained for 2 epoch on a small
    dataset.  We perform also the invariance test here because one needs a trained model
    for this.
    """

    structures = read_structures(DATASET_ETHANOL_PATH)
    # PR COMMENT this is a temporary hack until kernel is properly implemented that can
    #            deal with tensor maps with different species pairs
    # for structure in structures:
    #    structure.species = torch.ones(len(structure.species), dtype=torch.int32)

    conf = {
        "energy": {
            "quantity": "energy",
            "read_from": DATASET_ETHANOL_PATH,
            "file_format": ".xyz",
            "key": "energy",
            "forces": {
                "read_from": DATASET_ETHANOL_PATH,
                "file_format": ".xyz",
                "key": "forces",
            },
            "stress": False,
            "virial": False,
        }
    }

    targets = read_targets(OmegaConf.create(conf))
    dataset = Dataset(structure=structures, energy=targets["energy"])

    hypers = DEFAULT_HYPERS.copy()
    hypers["model"]["sparse_points"]["points"] = 900

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        species=get_all_species(dataset),
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
    )
    sparse_gap = train([dataset], [dataset], capabilities, hypers)
    # Predict on the first five structures
    output = sparse_gap(
        structures[:5], {"energy": sparse_gap.capabilities.outputs["energy"]}
    )
    # taken from the file ethanol_reduced_100.xyz
    data = ase.io.read(DATASET_ETHANOL_PATH, ":5", format="extxyz")
    expected_output = torch.tensor([[i.info["energy"]] for i in data])
    # np.savetxt(
    #    "/Users/davidetisi/Documents/Work/Software/metatensor-models/expected_forces.dat",
    #    -output["energy"].block().gradient("positions").values.reshape(45, 3),
    # )
    expected_forces = torch.vstack([torch.Tensor(i.arrays["forces"]) for i in data])
    # expected_forces = np.loadtxt(
    #    str(Path(__file__).parent.resolve() / "expected_forces.dat")
    # )
    assert torch.allclose(output["energy"].block().values, expected_output, rtol=0.1)
    assert torch.allclose(
        -output["energy"].block().gradient("positions").values.reshape(-1),
        torch.Tensor(expected_forces.reshape(-1)),
        rtol=20,
    )
    # breakpoint()
    # Tests that the model is rotationally invariant
    structure = ase.io.read(DATASET_ETHANOL_PATH)
    # PR COMMENT this is a temporary hack until kernel is properly implemented that can
    #            deal with tensor maps with different species pairs
    # structure.numbers = np.ones(len(structure.numbers))

    original_structure = copy.deepcopy(structure)
    structure.rotate(48, "y")

    original_output = sparse_gap(
        [rascaline.torch.systems_to_torch(original_structure)],
        {"energy": sparse_gap.capabilities.outputs["energy"]},
    )
    rotated_output = sparse_gap(
        [rascaline.torch.systems_to_torch(structure)],
        {"energy": sparse_gap.capabilities.outputs["energy"]},
    )

    assert torch.allclose(
        original_output["energy"].block().values,
        rotated_output["energy"].block().values,
    )