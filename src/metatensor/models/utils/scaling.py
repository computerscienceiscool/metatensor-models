import metatensor.torch
import torch
from metatensor.torch import TensorMap

from .evaluate_model import evaluate_model
from .metrics import RMSEAccumulator


def calculate_scaling(model, train_dataloader, device):
    """Calculate the scaling factors for a model.

    :param model: Model to calculate the scaling factors for.
    :param dataloader: Dataloader to calculate the scaling factors for.
    :returns: Scaling factors for the model.
    """

    model.scalings = torch.zeros_like(model.scalings)
    train_rmse_calculator = RMSEAccumulator()
    for batch in train_dataloader:
        systems, targets = batch
        systems = [system.to(device=device) for system in systems]
        targets = {key: value.to(device=device) for key, value in targets.items()}
        predictions = evaluate_model(
            model,
            systems,
            {name: [] for name, _ in targets.items()},
            is_training=False,
        )
        train_rmse_calculator.update(predictions, targets)
    standard_deviations = train_rmse_calculator.finalize()

    model.scalings = torch.ones_like(model.scalings)
    train_rmse_calculator = RMSEAccumulator()
    for batch in train_dataloader:
        systems, targets = batch
        systems = [system.to(device=device) for system in systems]
        targets = {key: value.to(device=device) for key, value in targets.items()}
        predictions = evaluate_model(
            model,
            systems,
            {name: [] for name, _ in targets.items()},
            is_training=False,
        )
        train_rmse_calculator.update(predictions, targets)
    rmses = train_rmse_calculator.finalize()

    for name in standard_deviations.keys():
        name_without_rmse = name.replace(" RMSE", "")
        model.scalings[model.output_to_index[name_without_rmse]] = (
            standard_deviations[name] / rmses[name]
        )


def apply_scaling(atomic_energies: TensorMap, scaling_factor: float) -> TensorMap:
    """Scales the atomic energies by the scaling factors."""

    return metatensor.torch.multiply(atomic_energies, scaling_factor)
