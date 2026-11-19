"""Module for defining constants related to data handling."""

from pathlib import Path
from typing import Final

import yaml

from eor_limits._datatypes import DataSet
from eor_limits.sensitivities import KNOWN_SENSITIVITIES
from eor_limits.theory import KNOWN_THEORIES

__all__ = [
    "_resolve_dataset_path",
    "load_limit_data",
    "load_sensitivity_data",
    "load_theory_model",
]


def load_theory_model(name: str | Path) -> DataSet:
    """
    Load a theory model from the known theories or from a YAML file.

    Parameters
    ----------
    name : str | Path
        The name of the theory model to load
        (see ``KNOWN_THEORIES`` for available models)
        or a path to a YAML file containing the model.

    Returns
    -------
    DataSet
        The loaded theory model as a DataSet object.

    """
    path = _resolve_dataset_path(name, KNOWN_THEORIES)
    return DataSet.load(path)


def load_limit_data(name: str | Path) -> DataSet:
    """
    Load observational limits data from known limits or from a YAML file.

    This function wraps and provides the same functionality
    as :func:`eor_limits.DataSet.load`, see docs there for more details.

    Parameters
    ----------
    name : str | Path
        The name of the limits data to load
        (see ``KNOWN_LIMITS`` for available limits)
        or a path to a YAML file containing the dataset.

    Returns
    -------
    DataSet
        The loaded observational limits as a DataSet object.
    """
    return DataSet.load(name)


# The unit of k that sensitivity files must declare in their ``k_unit`` entry.
SENSITIVITY_K_UNIT: Final = "h/Mpc"


def load_sensitivity_data(name: str | Path) -> DataSet:
    """
    Load a sensitivity estimate from the known sensitivities or from a YAML file.

    The YAML file uses the same format as the files of observational limits
    (see :func:`eor_limits.DataSet.load`). However, the |k| values must be given
    in the default units of |h/Mpc| since the plotting functions put both on the
    same axis without conversion.
    21cmSense v3 writes |k| in units of ``1/Mpc`` (see the ``k_unit`` attribute of
    its HDF5 output), so those values must be divided by the dimensionless Hubble
    parameter :math:`h` first. The YAML file must also contain the entry
    ``k_unit: h/Mpc``. This entry is not part of the limits format; it ensures
    that the |k| values are in the correct units, and the file is rejected if it is
    missing or different.
    Each file holds one kind of sensitivity (e.g. thermal noise only, or
    thermal noise and sample variance).


    Parameters
    ----------
    name : str | Path
        The name of a known sensitivity estimate (see ``KNOWN_SENSITIVITIES``),
        or a path to a YAML file containing the sensitivity.

    Returns
    -------
    DataSet
        The sensitivity estimate as a DataSet object.
    """
    path = _resolve_dataset_path(name, KNOWN_SENSITIVITIES)
    with path.open("r") as fl:
        yaml_data = yaml.safe_load(fl)
    k_unit = yaml_data.get("k_unit") if isinstance(yaml_data, dict) else None
    if k_unit != SENSITIVITY_K_UNIT:
        raise ValueError(
            f"Sensitivity file '{path.name}' must contain "
            f"'k_unit: {SENSITIVITY_K_UNIT}' (found {k_unit!r}). Give k in h/Mpc, "
            "as for the limits. Values in 1/Mpc, as written by 21cmSense v3, must "
            "be divided by h."
        )
    return DataSet.load(path)


def _resolve_dataset_path(name: str | Path, known: dict[str, Path]) -> Path:
    """
    Return the file for a known dataset name, otherwise treat ``name`` as a path.

    Parameters
    ----------
    name : str | Path
        A key of ``known``, or a path to a YAML file (``.yaml`` or ``.yml``).
    known : dict[str, Path]
        The registry of datasets to look ``name`` up in, e.g. ``KNOWN_LIMITS``,
        ``KNOWN_THEORIES`` or ``KNOWN_SENSITIVITIES``.

    Returns
    -------
    Path
        The path to the dataset's YAML file.

    Raises
    ------
    FileNotFoundError
        If ``name`` is neither a key of ``known`` nor an existing file.
    ValueError
        If ``name`` is a file that is not a YAML file.
    """
    if isinstance(name, str) and name in known:
        return known[name]
    path = Path(name)
    if not path.is_file():
        raise FileNotFoundError(
            f"'{name}' is neither a known name nor an existing file. "
            f"Known names: {list(known)}"
        )
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(
            f"Dataset files must be YAML (.yaml or .yml), got '{path.name}'."
        )
    return path
