"""Module for defining constants related to data handling."""

from pathlib import Path
from typing import Final

import yaml

from eor_limits._datatypes import DataSet
from eor_limits.data import DATA_PATH, KNOWN_LIMITS
from eor_limits.sensitivities import KNOWN_SENSITIVITIES
from eor_limits.theory import KNOWN_THEORIES, __all_theories__

__all__ = [
    "_normalize_dataset_name",
    "load_limit_data",
    "load_sensitivity_data",
    "load_theory_model",
]


def load_theory_model(name: str) -> DataSet:
    """
    Load a theory model from the known theories.

    Parameters
    ----------
    name : str
        The name of the theory model to load
        (see ``KNOWN_THEORIES`` for available models).

    Returns
    -------
    DataSet
        The loaded theory model as a DataSet object.

    """
    if name not in KNOWN_THEORIES:
        raise ValueError(
            f"Theory '{name}' not found. Available theories: {KNOWN_THEORIES.keys()}"
        )
    return __all_theories__[name].load_as_dataset()


def load_limit_data(name: str | Path) -> DataSet:
    """
    Load observational limits data from existing datasets or from a YAML file.

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

    ``name`` is either the name of a sensitivity estimate distributed with the
    package (see ``KNOWN_SENSITIVITIES``, e.g. ``'Breitman2026_AA4_deep1080h'``)
    or a path to a YAML file. Only names in ``KNOWN_SENSITIVITIES`` are looked
    up; any other ``name`` is used as a path.

    The YAML file uses the same format as the files of observational limits
    (see :func:`eor_limits.DataSet.load`), with the expected sensitivity given as
    |dsq| (``delta_squared``) or |delta| (``delta``) for each redshift |z| and
    scale |k|. For example:

    .. code-block:: yaml

        telescope: MyArray
        author: Doe
        year: 2026
        doi: unpublished
        k_unit: h/Mpc
        data:
          delta_squared: [[.inf, 20.0, 15.0, 40.0]]
          k: [[0.1, 0.2, 0.4, 0.8]]
          z: [8.0]

    The |k| values must be given in units of |h/Mpc|, the unit used for the limits,
    since the plotting functions put both on the same axis without conversion.
    21cmSense v3 writes |k| in units of ``1/Mpc`` (see the ``k_unit`` attribute of
    its HDF5 output), so those values must be divided by the dimensionless Hubble
    parameter :math:`h` first. The file must also contain the entry
    ``k_unit: h/Mpc``. This entry is not part of the limits format; it states the
    unit of |k| in each file, and the file is rejected if it is missing or
    different. The sensitivity is given in units of |mK^2|.
    Each file holds one kind of sensitivity (e.g. thermal noise only, or
    thermal noise and sample variance). Non-finite values (e.g. ``.inf`` for |k|
    bins without sensitivity) are allowed; they can be removed with
    :meth:`eor_limits.DataSet.drop_nan`.

    Parameters
    ----------
    name : str | Path
        The name of a known sensitivity estimate (see ``KNOWN_SENSITIVITIES``),
        or a path to a YAML file (``.yaml`` or ``.yml``) containing the sensitivity.

    Returns
    -------
    DataSet
        The sensitivity estimate as a DataSet object.

    Raises
    ------
    FileNotFoundError
        If ``name`` is neither a known sensitivity nor an existing file.
    ValueError
        If the file is not a YAML file, or if it does not contain
        ``k_unit: h/Mpc``.
    """
    if isinstance(name, str) and name in KNOWN_SENSITIVITIES:
        path = KNOWN_SENSITIVITIES[name]
    else:
        path = Path(name)
    if not path.is_file():
        raise FileNotFoundError(
            f"Sensitivity file '{path}' not found. "
            f"Known sensitivities: {list(KNOWN_SENSITIVITIES)}"
        )
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(
            f"Sensitivity files must be YAML files (.yaml or .yml), got '{path.name}'."
        )

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


def _normalize_dataset_name(path: str | Path, /) -> Path:
    if isinstance(path, str) and not path.endswith(".yaml"):
        path = DATA_PATH / (path + ".yaml")
    elif isinstance(path, str):
        path = Path(path)

    if not path.exists():
        path = DATA_PATH / path

    if not path.exists():
        raise ValueError(
            f"Dataset file '{path.name}' not found. "
            f"Available datasets: {KNOWN_LIMITS.keys()}"
        )

    return path
