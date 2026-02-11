"""Module for dealing with theory datasets."""

from ._base import KNOWN_THEORIES, THEORY_PATH, __all_theories__

# Import the individual theory processors to populate the __all_theories__ dictionary
from .Mesinger2016 import mesinger_2016
from .Munoz2018 import munoz_2018
from .Munoz2022 import munoz_2022
from .Pagano2020 import pagano_2020

__all__ = [
    "THEORY_PATH",
    "KNOWN_THEORIES",
    __all_theories__,
    "mesinger_2016",
    "munoz_2018",
    "munoz_2022",
    "pagano_2020",
]
