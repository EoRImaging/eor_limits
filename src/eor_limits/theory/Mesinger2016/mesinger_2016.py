#! /usr/bin/env python
# Copyright (c) 2019 Nichole Barry, Bryna Hazelton
# Licensed under the 2-clause BSD License
"""Process Mesinger et al. 2016 EOS spectra for plotting."""

from pathlib import Path

import numpy as np

from eor_limits._datatypes import Data
from eor_limits.theory._base import THEORY_PATH, BaseTheoryProcessor

faint_path = THEORY_PATH / "Mesinger2016" / "faint_galaxies/"
bright_path = THEORY_PATH / "Mesinger2016" / "bright_galaxies/"


def _parse_filename(filename: str) -> tuple[float, float]:
    """Parse the filename to extract the neutral fraction and redshift."""
    parts = filename.split("_")
    nf = None
    z = None
    for part in parts:
        if part.startswith("nf"):
            nf = float(part[2:])
        elif part.startswith("z"):
            z = float(part[1:])
    return nf, z


def _read_file(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the data from the file and return k and delta_squared arrays."""
    data_array = np.loadtxt(filepath, dtype=float)
    k_vals = data_array[:, 0]
    delta_squared = data_array[:, 1]
    return k_vals, delta_squared


class Mesinger2016Faint(BaseTheoryProcessor):
    """Processor for Mesinger et al. 2016 EOS spectra with faint galaxies."""

    simulator: str = "21cmFASTv2 (faint galaxies)"
    author: str = "Mesinger"
    year: int = 2016
    doi: str = "10.1093/mnras/stw831"
    _datapath = faint_path

    @classmethod
    def _load_data(cls) -> Data:
        """Process the data."""
        all_files = sorted(cls._datapath.glob("*"))
        data_list = []
        for f in all_files:
            z = _parse_filename(f.name)[1]
            k_vals, delta_squared = _read_file(f)
            data_list.append((z, k_vals, delta_squared))

        return Data(
            z=[d[0] for d in data_list],
            k=[d[1] for d in data_list],
            delta_squared=[d[2] for d in data_list],
        )


class Mesinger2016Bright(Mesinger2016Faint):
    """Processor for Mesinger et al. 2016 EOS spectra with bright galaxies."""

    simulator: str = "21cmFASTv2 (bright galaxies)"
    _datapath = bright_path
