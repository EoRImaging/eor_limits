"""Utility processors for different raw theory data formats."""

from eor_limits.datatypes import DataSet

from . import pagano_2020 as pagano_2020
from ._base import BaseTheoryProcessor as BaseTheoryProcessor
from ._base import __all_theories__
from .mesinger_2016 import Mesinger2016Bright as Mesinger2016Bright
from .mesinger_2016 import Mesinger2016Faint as Mesinger2016Faint
from .munoz_2018_fdm3 import Munoz2018FDM3 as Munoz2018FDM3
from .munoz_2021 import Munoz2022AllGalaxies as Munoz2022AllGalaxies
from .munoz_2021 import Munoz2022Optimistic as Munoz2022Optimistic

ALL_THEORIES = tuple(__all_theories__.keys())


def get_theory_data(theory_name: str) -> DataSet:
    """Get the theory data processor for a given theory name."""
    if theory_name not in __all_theories__:
        raise ValueError(
            f"Theory '{theory_name}' not found. Available theories: {__all_theories__}"
        )
    return __all_theories__[theory_name].load_as_dataset()
