import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import rascaline
import torch
from metatensor.torch.atomistic import ModelCapabilities

from ..utils.composition import calculate_composition_weights
from ..utils.compute_loss import compute_model_loss
from ..utils.data import (
    Dataset,
    check_datasets,
    collate_fn,
    combine_dataloaders,
    get_all_targets,
)
from ..utils.info import finalize_aggregated_info, update_aggregated_info
from ..utils.logging import MetricLogger
from ..utils.loss import TensorMapDictLoss
from ..utils.merge_capabilities import merge_capabilities
from ..utils.model_io import load_model, save_model
from .model import DEFAULT_HYPERS, Model


logger = logging.getLogger(__name__)

# disable rascaline logger
rascaline.set_logging_callback(lambda x, y: None)

# Filter out the second derivative warning from rascaline-torch
warnings.filterwarnings("ignore", category=UserWarning, message="second derivative")


def train(
    train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
    validation_datasets: List[Union[Dataset, torch.utils.data.Subset]],
    requested_capabilities: ModelCapabilities,
    hypers: Dict = DEFAULT_HYPERS,
    continue_from: Optional[str] = None,
    output_dir: str = ".",
) -> torch.nn.Module:
    # Create the model:
    if continue_from is None:
        model = Model(
            capabilities=requested_capabilities,
            hypers=hypers["model"],
        )
        new_capabilities = requested_capabilities
    else:
        model = load_model(continue_from)
        filtered_new_dict = {k: v for k, v in hypers["model"].items() if k != "restart"}
        filtered_old_dict = {k: v for k, v in model.hypers.items() if k != "restart"}
        if filtered_new_dict != filtered_old_dict:
            logger.warn(
                "The hyperparameters of the model have changed since the last "
                "training run. The new hyperparameters will be discarded."
            )
        # merge the model's capabilities with the requested capabilities
        merged_capabilities, new_capabilities = merge_capabilities(
            model.capabilities, requested_capabilities
        )
        model.capabilities = merged_capabilities
        # make the new model capable of handling the new outputs
        for output_name in new_capabilities.outputs.keys():
            model.add_output(output_name)

    model_capabilities = model.capabilities

    # Perform checks on the datasets:
    logger.info("Checking datasets for consistency")
    check_datasets(
        train_datasets,
        validation_datasets,
        model_capabilities,
    )

    # Calculate and set the composition weights for all targets:
    logger.info("Calculating composition weights")
    for target_name in new_capabilities.outputs.keys():
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
        composition_weights = calculate_composition_weights(
            train_datasets_with_target, target_name
        )
        model.set_composition_weights(target_name, composition_weights)

    hypers_training = hypers["training"]

    logger.info("Setting up data loaders")

    # Create dataloader for the training datasets:
    train_dataloaders = []
    for dataset in train_datasets:
        train_dataloaders.append(
            torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=hypers_training["batch_size"],
                shuffle=True,
                collate_fn=collate_fn,
            )
        )
    train_dataloader = combine_dataloaders(train_dataloaders, shuffle=True)

    # Create dataloader for the validation datasets:
    validation_dataloaders = []
    for dataset in validation_datasets:
        validation_dataloaders.append(
            torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=hypers_training["batch_size"],
                shuffle=False,
                collate_fn=collate_fn,
            )
        )
    validation_dataloader = combine_dataloaders(validation_dataloaders, shuffle=False)

    # Extract all the possible outputs and their gradients from the training set:
    outputs_dict = _get_outputs_dict(train_datasets)
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

    # counters for early stopping:
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    # Train the model:
    logger.info("Starting training")
    for epoch in range(hypers_training["num_epochs"]):
        # aggregated information holders:
        aggregated_train_info: Dict[str, Tuple[float, int]] = {}
        aggregated_validation_info: Dict[str, Tuple[float, int]] = {}

        train_loss = 0.0
        for batch in train_dataloader:
            optimizer.zero_grad()
            structures, targets = batch
            loss, info = compute_model_loss(loss_fn, model, structures, targets)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            aggregated_train_info = update_aggregated_info(aggregated_train_info, info)
        finalized_train_info = finalize_aggregated_info(aggregated_train_info)

        validation_loss = 0.0
        for batch in validation_dataloader:
            structures, targets = batch
            # TODO: specify that the model is not training here to save some autograd
            loss, info = compute_model_loss(loss_fn, model, structures, targets)
            validation_loss += loss.item()
            aggregated_validation_info = update_aggregated_info(
                aggregated_validation_info, info
            )
        finalized_validation_info = finalize_aggregated_info(aggregated_validation_info)

        # Now we log the information:
        if epoch == 0:
            metric_logger = MetricLogger(
                model_capabilities,
                train_loss,
                validation_loss,
                finalized_train_info,
                finalized_validation_info,
            )
        if epoch % hypers_training["log_interval"] == 0:
            metric_logger.log(
                epoch,
                train_loss,
                validation_loss,
                finalized_train_info,
                finalized_validation_info,
            )

        if epoch % hypers_training["checkpoint_interval"] == 0:
            save_model(
                model,
                Path(output_dir) / f"model_{epoch}.pt",
            )

        # early stopping criterion:
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 50:
                logger.info(
                    "Early stopping criterion reached after 50 "
                    "epochs without improvement."
                )
                break

    return model


def _get_outputs_dict(datasets: List[Union[Dataset, torch.utils.data.Subset]]):
    """
    This is a helper function that extracts all the possible outputs and their gradients
    from a list of datasets.

    :param datasets: A list of Datasets or Subsets.

    :returns: A dictionary mapping output names to a list of "values" (always)
        and possible gradients.
    """

    outputs_dict = {}
    for dataset in datasets:
        sample_batch = next(iter(dataset))
        targets = sample_batch[1]  # this is a dictionary of TensorMaps
        for target_name, target_tmap in targets.items():
            if target_name not in outputs_dict:
                outputs_dict[target_name] = [
                    "values"
                ] + target_tmap.block().gradients_list()

    return outputs_dict
