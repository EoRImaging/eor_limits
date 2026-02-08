#! /usr/bin/env python
# Copyright (c) 2019 Nichole Barry, Bryna Hazelton
# Licensed under the 2-clause BSD License
# (Slightly) modified by Julian Munoz in 2021
"""Process Munoz et al. 2021 spectra for plotting."""

import numpy as np

from eor_limits.data import THEORY_PATH
from eor_limits.datatypes import Data

from ._base import BaseTheoryProcessor


class Munoz2022AllGalaxies(BaseTheoryProcessor):
    """The AllGalaxies model from Munoz et al. 2022."""

    author: str = "Muñoz"
    year: int = 2022
    doi: str = "10.1093/mnras/stac185"

    _datapath = THEORY_PATH / "munoz_2021_allgalaxies"

    @classmethod
    def _get_paths(cls):
        munoz_file_k = cls._datapath / "1pt5Gpc_EOS_coeval_pow_kbins.bin"
        munoz_file_z = cls._datapath / "1pt5Gpc_EOS_coeval_pow_zlist.bin"
        munoz_file_p21 = cls._datapath / "1pt5Gpc_EOS_coeval_pow_P21.bin"
        return munoz_file_k, munoz_file_z, munoz_file_p21

    @classmethod
    def _load_data(cls) -> Data:
        """
        Get the AllGalaxies model.

        Parameters
        ----------
        model : str
            Which model to use. Options are 'EOS' for standard EOS2021 (AllGalaxies) or
            OPT for 'optimistic' (More PopIII stars)
        redshift :  float
            Which redshift to get a model for (the closest match).

        """
        munoz_file_k, munoz_file_z, munoz_file_p21 = cls._get_paths()

        redshifts = np.fromfile(munoz_file_z)
        k_arr = np.fromfile(munoz_file_k)
        delta_squared_arr = np.fromfile(munoz_file_p21).reshape((
            redshifts.size,
            k_arr.size,
        ))

        # Sort in order of ascending redshift
        order = np.argsort(redshifts)
        redshifts = redshifts[order]
        delta_squared_arr = delta_squared_arr[order]

        return Data(
            z=redshifts,
            k=tuple(k_arr for _ in redshifts),
            delta_squared=tuple(dsq for dsq in delta_squared_arr),
        )


class Munoz2022Optimistic(Munoz2022AllGalaxies):
    """The Optimistic model from Munoz et al. 2022, which has more PopIII stars."""

    simulator: str = "21cmFASTv3 (EOS2022, optimistic)"
    _datapath = THEORY_PATH / "munoz_2021_optimistic"

    @classmethod
    def _get_paths(cls):
        d = cls._datapath
        munoz_file_k = d / "600Mpc_pt0_coeval_pow_kbins.bin"
        munoz_file_z = d / "600Mpc_pt0_coeval_pow_zlist.bin"
        munoz_file_p21 = d / "600Mpc_pt0_coeval_pow_P21.bin"
        return munoz_file_k, munoz_file_z, munoz_file_p21
