"""
eor-limits: A package for plotting and comparing 21cm EoR limits and theories.

Copyright (c) 2019 Nichole Barry, Bryna Hazelton
Licensed under the 2-clause BSD License
"""

from ._data_loading import (
    load_limit_data,
    load_sensitivity_data,
    load_theory_model,
)
from ._datatypes import Data, DataSet
from ._plot import get_latex_label, plot_vs_k, plot_vs_z
from .data import KNOWN_LIMITS
from .sensitivities import KNOWN_SENSITIVITIES
from .theory import KNOWN_THEORIES

__all__ = [
    "KNOWN_LIMITS",
    "KNOWN_SENSITIVITIES",
    "KNOWN_THEORIES",
    "Data",
    "DataSet",
    "get_latex_label",
    "load_limit_data",
    "load_sensitivity_data",
    "load_theory_model",
    "plot_vs_k",
    "plot_vs_z",
]
