"""Module for defining constants related to data handling."""

from pathlib import Path

from eor_limits._datatypes import DataSet
from eor_limits.data import DATA_PATH, KNOWN_LIMITS
from eor_limits.theory import KNOWN_THEORIES, __all_theories__

__all___ = [
    "load_theory_model",
    "load_limit_data",
    "_normalize_dataset_name",
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
