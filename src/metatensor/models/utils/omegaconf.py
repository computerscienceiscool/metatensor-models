from pathlib import Path
from typing import Union

from omegaconf import DictConfig, OmegaConf


def file_format(_parent_):
    """Custom OmegaConf resolver to find the file format.

    File format is obtained based on the suffix of the ``read_from`` field in the same
    section."""
    return Path(_parent_["read_from"]).suffix


# Register custom resolvers
OmegaConf.register_new_resolver("file_format", file_format)


def _resolve_single_str(config):
    if isinstance(config, str):
        return OmegaConf.create({"read_from": config})
    else:
        return config


# BASE CONFIGURATIONS
CONF_STRUCTURES = OmegaConf.create(
    {
        "read_from": "${..read_from}",
        "file_format": "${file_format:}",
        "key": None,
        "length_unit": None,
    }
)

CONF_TARGET_FIELDS = OmegaConf.create(
    {
        "quantity": "energy",
        "read_from": "${...structures.read_from}",
        "file_format": "${file_format:}",
        "key": None,
        "unit": None,
    }
)

CONF_GRADIENTS = OmegaConf.create({"forces": False, "stress": False, "virial": False})
CONF_GRADIENT = OmegaConf.create(
    {
        "read_from": "${..read_from}",
        "file_format": "${file_format:}",
        "key": None,
    }
)

KNWON_GRADIENTS = list(CONF_GRADIENTS.keys())

# merge configs to get default configs for energies and other targets
CONF_TARGET = OmegaConf.merge(CONF_TARGET_FIELDS, CONF_GRADIENTS)
CONF_ENERGY = CONF_TARGET.copy()
CONF_ENERGY["forces"] = CONF_GRADIENT.copy()
CONF_ENERGY["stress"] = CONF_GRADIENT.copy()


def expand_dataset_config(conf: Union[str, DictConfig]) -> DictConfig:
    """Expands shorthand notations in a dataset configuration to their full formats.

    This function takes a dataset configuration, either as a string or a DictConfig, and
    expands it into a detailed configuration format. It processes structures, targets,
    and gradient sections, setting default values and inferring missing information.
    Unknown keys are ignored, allowing for flexibility.

    The function performs the following steps:

    - Loads base configurations for structures, targets, and gradients from predefined
      YAML files.
    - Merges and interpolates the input configuration with the base configurations.
    - Expands shorthand notations like file paths or simple true/false settings to full
      dictionary structures.
    - Handles special cases, such as the mandatory nature of the 'energy' section for MD
      simulations and the mutual exclusivity of 'stress' and 'virial' sections.
    - Validates the final expanded configuration, particularly for gradient-related
      settings, to ensure consistency and prevent conflicts during training.

    :param conf: The dataset configuration, either as a file path string or a DictConfig
        object.
    :returns: The fully expanded dataset configuration.
    :raises ValueError: If both ``virial`` and ``stress`` sections are enabled in the
        'energy' target, as this is not permissible for training.
    """
    if isinstance(conf, str):
        read_from = conf
        conf = OmegaConf.create(
            {"structures": read_from, "targets": {"energy": read_from}}
        )

    if type(conf["structures"]) is str:
        conf["structures"] = _resolve_single_str(conf["structures"])

    conf["structures"] = OmegaConf.merge(CONF_STRUCTURES, conf["structures"])

    for target_key, target in conf["targets"].items():
        if type(target) is str:
            target = _resolve_single_str(target)

        # Add default gradients "energy" target section
        if target_key == "energy":
            # For special case of the "energy" we add the section for force and stress
            # gradient by default
            target = OmegaConf.merge(CONF_ENERGY, target)
        else:
            target = OmegaConf.merge(CONF_TARGET, target)

        if target["key"] is None:
            target["key"] = target_key

        # Update DictConfig to allow for config node interpolation
        conf["targets"][target_key] = target

        # merge and interpolate possibly present gradients with default gradient config
        for gradient_key, gradient_conf in conf["targets"][target_key].items():
            if gradient_key in KNWON_GRADIENTS:
                if gradient_conf is True:
                    gradient_conf = CONF_GRADIENT.copy()
                elif type(gradient_conf) is str:
                    gradient_conf = _resolve_single_str(gradient_conf)

                if isinstance(gradient_conf, DictConfig):
                    gradient_conf = OmegaConf.merge(CONF_GRADIENT, gradient_conf)

                    if gradient_conf["key"] is None:
                        gradient_conf["key"] = gradient_key

                    conf["targets"][target_key][gradient_key] = gradient_conf

        # If user sets the virial gradient and leaves the stress section untouched,
        # we disable the by default enabled stress gradient section.
        base_stress_gradient_conf = CONF_GRADIENT.copy()
        base_stress_gradient_conf["key"] = "stress"

        if (
            target_key == "energy"
            and conf["targets"][target_key]["virial"]
            and conf["targets"][target_key]["stress"] == base_stress_gradient_conf
        ):
            conf["targets"][target_key]["stress"] = False

        if (
            conf["targets"][target_key]["stress"]
            and conf["targets"][target_key]["virial"]
        ):
            raise ValueError(
                f"Cannot perform training with respect to virials and stress as in "
                f"section {target_key}. Set either `virials: off` or `stress: off`."
            )

    return conf