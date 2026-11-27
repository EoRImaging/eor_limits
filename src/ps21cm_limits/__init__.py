"""
ps21cm-limits: A 21-cm power spectrum limits compendium and plotting package.

This package provides tools for loading and plotting the latest observational
limits on the 21-cm power spectrum from the Epoch of Reionization (EoR) and
Cosmic Dawn, alongside theoretical models and sensitivities of current and
future experiments.
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
