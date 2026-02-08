"""
eor-limits: A package for plotting and comparing 21cm EoR limits and theories.

Copyright (c) 2019 Nichole Barry, Bryna Hazelton
Licensed under the 2-clause BSD License
"""

from . import plots
from .data import DATA_PATH, KNOWN_PAPERS, THEORY_PATH
from .datatypes import DataSet

__all__ = [
    "DATA_PATH",
    "KNOWN_PAPERS",
    "THEORY_PATH",
    "DataSet",
    "plots",
]
