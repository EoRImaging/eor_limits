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
    """Get the theory data processor for a given theory name."""
    if name not in KNOWN_THEORIES or name not in __all_theories__:
        raise ValueError(
            f"Theory '{name}' not found. Available theories: {KNOWN_THEORIES.keys()}"
        )
    return __all_theories__[name].load_as_dataset()


def load_limit_data(name: str | Path, /) -> DataSet:
    """Load the limit data for a given paper name."""
    return DataSet.load(name)


def _normalize_dataset_name(path: str | Path) -> Path:
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
