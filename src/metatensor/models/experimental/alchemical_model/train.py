import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from metatensor.learn.data import DataLoader
from metatensor.learn.data.dataset import Dataset
from metatensor.torch import TensorMap
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput

from ...utils.composition import calculate_composition_weights
from ...utils.data import (
    CombinedDataLoader,
    DatasetInfo,
    check_datasets,
    collate_fn,
    get_all_species,
    get_all_targets,
)
from ...utils.evaluate_model import evaluate_model
from ...utils.extract_targets import get_outputs_dict
from ...utils.io import is_exported, load, save
from ...utils.logging import MetricLogger
from ...utils.loss import TensorMapDictLoss
from ...utils.merge_capabilities import merge_capabilities
from ...utils.metrics import RMSEAccumulator
from ...utils.neighbors_lists import get_system_with_neighbors_lists
from ...utils.per_atom import average_block_by_num_atoms
from .model import DEFAULT_HYPERS, Model
from .utils.normalize import (
    get_average_number_of_atoms,
    get_average_number_of_neighbors,
)


logger = logging.getLogger(__name__)


def train(
    train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
    validation_datasets: List[Union[Dataset, torch.utils.data.Subset]],
    dataset_info: DatasetInfo,
    devices: List[torch.device],
    hypers: Dict = DEFAULT_HYPERS,
    continue_from: Optional[str] = None,
    output_dir: str = ".",
):
    all_species = get_all_species(train_datasets + validation_datasets)
    outputs = {
        key: ModelOutput(
            quantity=value.quantity,
            unit=value.unit,
            per_atom=False,
        )
        for key, value in dataset_info.targets.items()
    }
    new_capabilities = ModelCapabilities(
        length_unit=dataset_info.length_unit,
        outputs=outputs,
        atomic_types=all_species,
        supported_devices=["cpu", "cuda"],
    )

    if continue_from is None:
        model = Model(
            capabilities=new_capabilities,
            hypers=hypers["model"],
        )
        novel_capabilities = new_capabilities
    else:
        model = load(continue_from)
        if is_exported(model):
            raise ValueError("model is already exported and can't be used for continue")

        filtered_new_dict = {k: v for k, v in hypers["model"].items() if k != "restart"}
        filtered_old_dict = {k: v for k, v in model.hypers.items() if k != "restart"}
        if filtered_new_dict != filtered_old_dict:
            logger.warning(
                "The hyperparameters of the model have changed since the last "
                "training run. The new hyperparameters will be discarded."
            )
        # merge the model's capabilities with the requested capabilities
        merged_capabilities, novel_capabilities = merge_capabilities(
            model.capabilities, new_capabilities
        )
        model.capabilities = merged_capabilities
        # make the new model capable of handling the new outputs
        for output_name in novel_capabilities.outputs.keys():
            model.add_output(output_name)

    model_capabilities = model.capabilities

    # Perform canonical checks on the datasets:
    logger.info("Checking datasets for consistency")
    try:
        check_datasets(train_datasets, validation_datasets)
    except ValueError as err:
        if continue_from is not None:
            logger.warning(err)
        else:
            # only error if we are not continuing
            raise ValueError(err) from err

    # Calculating the neighbors lists for the training and validation datasets:
    logger.info("Calculating neighbors lists for the datasets")
    requested_neighbor_lists = model.requested_neighbors_lists()
    for dataset in train_datasets + validation_datasets:
        for i in range(len(dataset)):
            system = dataset[i].system
            # The following line attached the neighbors lists to the system,
            # and doesn't require to reassign the system to the dataset:
            _ = get_system_with_neighbors_lists(system, requested_neighbor_lists)

    # Calculate the average number of atoms and neighbors in the training datasets:
    average_number_of_atoms = get_average_number_of_atoms(train_datasets)
    average_number_of_neighbors = get_average_number_of_neighbors(train_datasets)

    # Given that currently multiple datasets are not supported, we can assume that:
    average_number_of_atoms = average_number_of_atoms[0]
    average_number_of_neighbors = average_number_of_neighbors[0]

    # Set the normalization factors for the model:
    model.set_normalization_factor(average_number_of_atoms)
    model.set_basis_normalization_factor(average_number_of_neighbors)

    device = devices[0]  # only one device, as we don't support multi-gpu for now
    dtype = train_datasets[0][0].system.positions.dtype

    logger.info(f"training on device {device} with dtype {dtype}")
    model.to(device=device, dtype=dtype)

    # Calculate and set the composition weights for all targets:
    for target_name in novel_capabilities.outputs.keys():
        # TODO: warn in the documentation that capabilities that are already
        # present in the model won't recalculate the composition weights
        # find the datasets that contain the target:
        train_datasets_with_target = []
        for dataset in train_datasets:
            if target_name in get_all_targets(dataset):
                train_datasets_with_target.append(dataset)
        if len(train_datasets_with_target) == 0:
            raise ValueError(
                f"Target {target_name} in the model's new capabilities is not "
                "present in any of the training datasets."
            )
        composition_weights, species = calculate_composition_weights(
            train_datasets_with_target, target_name
        )
        model.set_composition_weights(composition_weights.unsqueeze(0), species)

    hypers_training = hypers["training"]

    logger.info("Setting up data loaders")

    # Create dataloader for the training datasets:
    train_dataloaders = []
    for dataset in train_datasets:
        train_dataloaders.append(
            DataLoader(
                dataset=dataset,
                batch_size=hypers_training["batch_size"],
                shuffle=True,
                collate_fn=collate_fn,
            )
        )
    train_dataloader = CombinedDataLoader(train_dataloaders, shuffle=True)

    # Create dataloader for the validation datasets:
    validation_dataloaders = []
    for dataset in validation_datasets:
        validation_dataloaders.append(
            DataLoader(
                dataset=dataset,
                batch_size=hypers_training["batch_size"],
                shuffle=False,
                collate_fn=collate_fn,
            )
        )
    validation_dataloader = CombinedDataLoader(validation_dataloaders, shuffle=False)

    # Extract all the possible outputs and their gradients from the training set:
    outputs_dict = get_outputs_dict(train_datasets)
    for output_name in outputs_dict.keys():
        if output_name not in model_capabilities.outputs:
            raise ValueError(
                f"Output {output_name} is not in the model's capabilities."
            )

    # Create a loss weight dict:
    loss_weights_dict = {}
    for output_name, value_or_gradient_list in outputs_dict.items():
        loss_weights_dict[output_name] = {
            value_or_gradient: 1.0 for value_or_gradient in value_or_gradient_list
        }

    # Create a loss function:
    loss_fn = TensorMapDictLoss(loss_weights_dict)

    # Create an optimizer:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hypers_training["learning_rate"]
    )

    # Create a scheduler:
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=hypers_training["scheduler_factor"],
        patience=hypers_training["scheduler_patience"],
    )

    # counters for early stopping:
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    # per-atom targets:
    per_atom_targets = hypers_training["per_atom_targets"]

    # Train the model:
    logger.info("Starting training")
    for epoch in range(hypers_training["num_epochs"]):
        train_rmse_calculator = RMSEAccumulator()
        validation_rmse_calculator = RMSEAccumulator()

        train_loss = 0.0
        for batch in train_dataloader:
            optimizer.zero_grad()

            systems, targets = batch
            assert len(systems[0].known_neighbors_lists()) > 0
            systems = [system.to(device=device) for system in systems]
            targets = {key: value.to(device=device) for key, value in targets.items()}
            predictions = evaluate_model(
                model,
                systems,
                {
                    name: tensormap.block().gradients_list()
                    for name, tensormap in targets.items()
                },
                is_training=True,
            )

            # average by the number of atoms (if requested)
            num_atoms = torch.tensor(
                [len(s) for s in systems], device=device
            ).unsqueeze(-1)
            for pa_target in per_atom_targets:
                predictions[pa_target] = TensorMap(
                    predictions[pa_target].keys,
                    [
                        average_block_by_num_atoms(
                            predictions[pa_target].block(), num_atoms
                        )
                    ],
                )
                targets[pa_target] = TensorMap(
                    targets[pa_target].keys,
                    [average_block_by_num_atoms(targets[pa_target].block(), num_atoms)],
                )

            train_loss_batch = loss_fn(predictions, targets)
            train_loss += train_loss_batch.item()
            train_loss_batch.backward()
            optimizer.step()
            train_rmse_calculator.update(predictions, targets)
        finalized_train_info = train_rmse_calculator.finalize()

        validation_loss = 0.0
        for batch in validation_dataloader:
            systems, targets = batch
            assert len(systems[0].known_neighbors_lists()) > 0
            systems = [system.to(device=device) for system in systems]
            targets = {key: value.to(device=device) for key, value in targets.items()}
            predictions = evaluate_model(
                model,
                systems,
                {
                    name: tensormap.block().gradients_list()
                    for name, tensormap in targets.items()
                },
                is_training=False,
            )

            # average by the number of atoms (if requested)
            num_atoms = torch.tensor(
                [len(s) for s in systems], device=device
            ).unsqueeze(-1)
            for pa_target in per_atom_targets:
                predictions[pa_target] = TensorMap(
                    predictions[pa_target].keys,
                    [
                        average_block_by_num_atoms(
                            predictions[pa_target].block(), num_atoms
                        )
                    ],
                )
                targets[pa_target] = TensorMap(
                    targets[pa_target].keys,
                    [average_block_by_num_atoms(targets[pa_target].block(), num_atoms)],
                )

            validation_loss_batch = loss_fn(predictions, targets)
            validation_loss += validation_loss_batch.item()
            validation_rmse_calculator.update(predictions, targets)
        finalized_validation_info = validation_rmse_calculator.finalize()

        lr_scheduler.step(validation_loss)

        # Now we log the information:
        finalized_train_info = {"loss": train_loss, **finalized_train_info}
        finalized_validation_info = {
            "loss": validation_loss,
            **finalized_validation_info,
        }

        if epoch == 0:
            metric_logger = MetricLogger(
                model_capabilities=model_capabilities,
                initial_metrics=[finalized_train_info, finalized_validation_info],
                names=["train", "validation"],
            )
        if epoch % hypers_training["log_interval"] == 0:
            metric_logger.log(
                metrics=[finalized_train_info, finalized_validation_info],
                epoch=epoch,
            )

        if epoch % hypers_training["checkpoint_interval"] == 0:
            save(
                model,
                Path(output_dir) / f"model_{epoch}.pt",
            )

        # early stopping criterion:
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= hypers_training["early_stopping_patience"]:
                logger.info(
                    "Early stopping criterion reached after "
                    f"{hypers_training['early_stopping_patience']} epochs "
                    "without improvement."
                )
                break

    return model
