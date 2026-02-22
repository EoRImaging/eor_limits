"""Abstract base class for theory processors."""

from abc import ABC, abstractmethod
from pathlib import Path

from eor_limits._datatypes import Data, DataSet

# Import paths and theory registry
THEORY_PATH = Path(__file__).parent.resolve()
KNOWN_THEORIES = {}  # populated in __init__.py
__all_theories__ = {}  # populated in __init__.py


class BaseTheoryProcessor(ABC):
    """Abstract base class for theory processors."""

    simulator: str
    author: str
    year: int
    doi: str
    _datapath: Path

    def __init_subclass__(cls) -> None:
        """Register subclasses in the __all_theories__ dictionary."""
        __all_theories__[cls.__name__] = cls
        KNOWN_THEORIES[cls.__name__] = cls._datapath
        return super().__init_subclass__()

    @classmethod
    @abstractmethod
    def _load_data(cls) -> Data:
        """Process the data."""
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    def load_as_dataset(cls) -> DataSet:
        """Load the data as a DataSet."""
        return DataSet(
            telescope=cls.simulator,
            author=cls.author,
            year=cls.year,
            doi=cls.doi,
            data=cls._load_data(),
            key=cls.__name__,
        )
