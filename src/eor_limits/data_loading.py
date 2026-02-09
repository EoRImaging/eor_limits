"""Module for defining constants related to data handling."""

from pathlib import Path

from eor_limits.datatypes import DataSet

from ._paths import DATA_PATH
from .theory import __all_theories__

__all___ = [
    "KNOWN_PAPERS",
    "KNOWN_THEORIES",
    "load_theory_model",
    "load_limit_data",
    "_normalize_dataset_name",
]


KNOWN_PAPERS = {p.stem: p for p in DATA_PATH.glob("*.yaml") if p.is_file()}
KNOWN_THEORIES = tuple(__all_theories__.keys())


def load_theory_model(name: str) -> DataSet:
    """Get the theory data processor for a given theory name."""
    if name not in __all_theories__:
        raise ValueError(
            f"Theory '{name}' not found. Available theories: {KNOWN_THEORIES}"
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
            f"Available datasets: {KNOWN_PAPERS.keys()}"
        )

    return path
