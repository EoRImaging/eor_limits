#! /usr/bin/env python
# Copyright (c) 2019 Nichole Barry, Bryna Hazelton
# Licensed under the 2-clause BSD License
"""Process Pagano and Liu 2020 spectra for plotting."""

import numpy as np

from eor_limits._paths import THEORY_PATH
from eor_limits.datatypes import Data

from ._base import BaseTheoryProcessor

pagano_file = THEORY_PATH / "pagano_liu_2020.npz"

# load all possible betas
with np.load(pagano_file) as data:
    _betas = data["betas"]


def _pagano_factory(betaidx: float):
    beta = _betas[betaidx]

    def load_data(cls) -> Data:
        with np.load(pagano_file) as data:
            delta_squared_arr = data["power_spectra"]
            k_arr = data["bins_k"]
            redshifts = data["redshifts"]
        data.close()

        return Data(
            z=redshifts,
            k=(k_arr,) * len(redshifts),
            delta_squared=tuple(dsq for dsq in delta_squared_arr[betaidx, :, :]),
        )

    # Create the class dynamically
    return type(
        f"PaganoLiu2020Beta{beta:.2f}",
        (BaseTheoryProcessor,),
        {
            "simulator": f"21cmFASTv2 (beta={beta:.2f})",
            "author": "Pagano and Liu",
            "year": 2020,
            "doi": "10.1093/mnras/staa2118",
            "_load_data": classmethod(load_data),
        },
    )


for betidx in range(len(_betas)):
    cls = _pagano_factory(betidx)
    globals()[cls.__name__] = cls
