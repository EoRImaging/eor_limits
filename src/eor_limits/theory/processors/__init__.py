"""Utility processors for different raw theory data formats."""

# Import each module so that the theory
# processors are registered in __all_theories__.
from . import mesinger_2016 as mesinger_2016
from . import munoz_2018_fdm3 as munoz_2018_fdm3
from . import munoz_2022 as munoz_2022
from . import pagano_2020 as pagano_2020
from ._base import __all_theories__ as __all_theories__
