import logging
import warnings
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
from ...utils.metrics import MAEAccumulator, RMSEAccumulator
from ...utils.neighbors_lists import get_system_with_neighbors_lists
from ...utils.per_atom import average_block_by_num_atoms
from ...utils.scaling import calculate_scaling
from .model import DEFAULT_HYPERS, Model


logger = logging.getLogger(__name__)


# Filter out the second derivative and device warnings from rascaline-torch
warnings.filterwarnings("ignore", category=UserWarning, message="second derivative")
warnings.filterwarnings(
    "ignore", category=UserWarning, message="Systems data is on device"
)


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

    # Create the model:
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

    torch.jit.set_fusion_strategy([("DYNAMIC", 0)])
    model = torch.jit.script(model)

    logger.info(
        f"Model k_max_l: {model.k_max_l}"
    )

    model_capabilities = model.capabilities

    # Perform checks on the datasets:
    logger.info("Checking datasets for consistency")
    try:
        check_datasets(train_datasets, validation_datasets)
    except ValueError as err:
        if continue_from is not None:
            logger.warning(err)
        else:
            # only error if we are not continuing
            raise ValueError(err) from err

    device = devices[0]  # only one device, as we don't support multi-gpu for now
    dtype = train_datasets[0][0].system.positions.dtype

    logger.info(f"training on device {device} with dtype {dtype}")
    model.to(device=device, dtype=dtype)

    hypers_training = hypers["training"]

    # Calculate and set the composition weights for all targets:
    logger.info("Calculating composition weights")
    for target_name in novel_capabilities.outputs.keys():
        # TODO: warn in the documentation that capabilities that are already
        # present in the model won't recalculate the composition weights
        # find the datasets that contain the target:

        if target_name in hypers_training["fixed_composition_weights"].keys():
            logger.info(
                f"For {target_name}, model will proceed with "
                "user-supplied composition weights"
            )
            cur_weight_dict = hypers_training["fixed_composition_weights"][target_name]
            species = []
            num_species = len(cur_weight_dict)
            fixed_weights = torch.zeros(num_species, dtype=dtype, device=device)

            for ii, (key, weight) in enumerate(cur_weight_dict.items()):
                species.append(key)
                fixed_weights[ii] = weight

            all_species = []
            for dataset in train_datasets:
                all_species += get_all_species(dataset)

            if not set(species) == set(all_species):
                raise ValueError(
                    "Values were not supplied for all "
                    "the species in present in the dataset"
                )
            model.set_composition_weights(target_name, fixed_weights, species)

        else:
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
            model.set_composition_weights(target_name, composition_weights, species)

    logger.info("Setting up data loaders")

    import tqdm

    # Calculate NLs:
    logger.info("Calculating neighbors lists for the datasets")
    requested_neighbor_lists = model.requested_neighbors_lists()
    for dataset in train_datasets + validation_datasets:
        for i in tqdm.tqdm(range(len(dataset))):
            system = dataset[i].system
            # The following line attached the neighbors lists to the system,
            # and doesn't require to reassign the system to the dataset:
            _ = get_system_with_neighbors_lists(system, requested_neighbor_lists)

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
    # train_dataloader = train_dataloaders[0]

    # scaling:
    # TODO: this will work sub-optimally if the model is restarting with
    # new targets (but it will still work)
    calculate_scaling(model, train_dataloader, device)

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
    loss_fn = TensorMapDictLoss(loss_weights_dict, reduction="sum")

    # special_params = [
    #     param for name, param in model.named_parameters() if ('lengthscales' in name or 'embeddings' in name or 'last_layers' in name)
    # ]
    # other_params = [
    #     param for name, param in model.named_parameters()
    #     if ('lengthscales' not in name and 'embeddings' not in name and 'last_layers' not in name)
    # ]

    # lr = hypers_training["learning_rate"]

    # param_groups = [
    #     {'params': special_params, 'lr': lr/100.0},
    #     {'params': other_params, 'lr': lr}
    # ]


    # Create an optimizer and a scheduler:
    optimizer = torch.optim.AdamW(model.parameters(), lr=hypers_training["learning_rate"], amsgrad=True, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=hypers_training["scheduler_factor"], patience=hypers_training["scheduler_patience"])

    # per-atom targets:
    per_atom_targets = hypers_training["per_atom_targets"]

    # Train the model:
    logger.info("Starting training")

    from torch.profiler import profile, ProfilerActivity

    best_validation_loss = float("inf")
    n_epochs_without_improvement = 0
    for epoch in range(hypers_training["num_epochs"]):
        train_rmse_calculator = RMSEAccumulator()
        validation_rmse_calculator = RMSEAccumulator()
        train_mae_calculator = MAEAccumulator()
        validation_mae_calculator = MAEAccumulator()

        train_loss = 0.0
        for batch in train_dataloader:
            optimizer.zero_grad()
            systems, targets = batch

            systems = [system.to(device=device) for system in systems]
            targets = {
                key: value.to(device=device) for key, value in targets.items()
            }
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
            predictions = evaluate_model(
                model,
                systems,
                {
                    name: tensormap.block().gradients_list()
                    for name, tensormap in targets.items()
                },
                is_training=True,
            )

            # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
            # exit()
            
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
                    [
                        average_block_by_num_atoms(
                            targets[pa_target].block(), num_atoms
                        )
                    ],
                )

            train_rmse_calculator.update(predictions, targets)
            train_mae_calculator.update(predictions, targets)
            loss = loss_fn(predictions, targets)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        finalized_train_info = {
            **train_rmse_calculator.finalize(),
            **train_mae_calculator.finalize(),
        }

        validation_loss = 0.0
        for batch in validation_dataloader:
            systems, targets = batch
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

            loss = loss_fn(predictions, targets)
            validation_loss += loss.item()
            validation_rmse_calculator.update(predictions, targets)
            validation_mae_calculator.update(predictions, targets)
        finalized_validation_info = {
            **validation_rmse_calculator.finalize(),
            **validation_mae_calculator.finalize(),
        }

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
            non_scripted_model = Model(
                capabilities=model_capabilities,
                hypers=hypers["model"],
            ).to(device=device, dtype=dtype)
            non_scripted_model.hypers = hypers["model"]
            non_scripted_model.load_state_dict(model.state_dict())
            save(
                non_scripted_model,
                Path(output_dir) / f"model_{epoch}.ckpt",
            )

        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step(validation_loss)
        lr_after = optimizer.param_groups[0]["lr"]
        if lr_before != lr_after:
            logger.info(f"Learning rate changed from {lr_before} to {lr_after}")
        if lr_after < 1e-6:
            logger.info("Training has converged, stopping")
            break

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            n_epochs_without_improvement = 0
        else:
            n_epochs_without_improvement += 1
            if n_epochs_without_improvement >= hypers_training["early_stopping_patience"]:
                logger.info(
                    f"Stopping early after {n_epochs_without_improvement} epochs without improvement"
                )
                break

    non_scripted_model = Model(
        capabilities=model_capabilities,
        hypers=hypers["model"],
    ).to(device=device, dtype=dtype)
    non_scripted_model.load_state_dict(model.state_dict())
    non_scripted_model.hypers = hypers["model"]

    return non_scripted_model
