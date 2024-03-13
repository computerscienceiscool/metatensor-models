from pathlib import Path

import torch
from metatensor.learn.data import Dataset
from omegaconf import OmegaConf

from metatensor.models.utils.data import (
    DatasetInfo,
    TargetInfo,
    collate_fn,
    get_all_species,
    read_systems,
    read_targets,
)


RESOURCES_PATH = Path(__file__).parent.resolve() / ".." / ".." / "resources"


def test_dataset_info():
    """Tests the DatasetInfo class."""

    dataset_info = DatasetInfo(
        length_unit="angstrom",
        targets={
            "energy": TargetInfo(quantity="energy", unit="kcal/mol"),
            "U0": TargetInfo(quantity="energy", unit="kcal/mol"),
        },
    )

    assert dataset_info.length_unit == "angstrom"
    assert dataset_info.targets["energy"].quantity == "energy"
    assert dataset_info.targets["energy"].unit == "kcal/mol"
    assert dataset_info.targets["U0"].quantity == "energy"
    assert dataset_info.targets["U0"].unit == "kcal/mol"


def test_dataset():
    """Tests the readers and the dataset class."""

    systems = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")

    filename = str(RESOURCES_PATH / "qm9_reduced_100.xyz")
    conf = {
        "energy": {
            "quantity": "energy",
            "read_from": filename,
            "file_format": ".xyz",
            "key": "U0",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets = read_targets(OmegaConf.create(conf))
    dataset = Dataset(system=systems, energy=targets["energy"])
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=10, collate_fn=collate_fn
    )

    for batch in dataloader:
        assert batch[1]["energy"].block().values.shape == (10, 1)


def test_species_list():
    """Tests that the species list is correctly computed with get_all_species."""

    systems = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")
    conf = {
        "energy": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "qm9_reduced_100.xyz"),
            "file_format": ".xyz",
            "key": "U0",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    systems_2 = read_systems(RESOURCES_PATH / "ethanol_reduced_100.xyz")
    conf_2 = {
        "energy": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "ethanol_reduced_100.xyz"),
            "file_format": ".xyz",
            "key": "energy",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets = read_targets(OmegaConf.create(conf))
    targets_2 = read_targets(OmegaConf.create(conf_2))
    dataset = Dataset(system=systems, **targets)
    dataset_2 = Dataset(system=systems_2, **targets_2)
    assert get_all_species(dataset) == [1, 6, 7, 8]
    assert get_all_species(dataset_2) == [1, 6, 8]
    assert get_all_species([dataset, dataset_2]) == [1, 6, 7, 8]
