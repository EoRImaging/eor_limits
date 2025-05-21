#! /usr/bin/env python
# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Nichole Barry, Bryna Hazelton
# Licensed under the 2-clause BSD License
# (Slightly) modified by Julian Munoz in 2021
"""Process Munoz et al. 2021 spectra for plotting."""

import os

import numpy as np

from eor_limits.data import DATA_PATH


def get_munoz_2021_line(model="EOS", redshift=None, linewidth=1.0, nf=None):
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
    if model not in ["EOS", "OPT"]:
        raise ValueError("Model must be either 'EOS' or 'OPT'.")

    paper_dict = {
        "author": r"Mu\~noz",
        "year": 2021,
        "model": " ",
        "doi": "arXiv: 2110.13919",
        "type": "line",
        "marker": ".",
        "linewidth": linewidth,
    }

    if model == "EOS":
        munoz_file_k = os.path.join(
            DATA_PATH,
            "theory",
            "munoz_2021_allgalaxies/1pt5Gpc_EOS_coeval_pow_kbins.bin",
        )
        munoz_file_z = os.path.join(
            DATA_PATH,
            "theory",
            "munoz_2021_allgalaxies/1pt5Gpc_EOS_coeval_pow_zlist.bin",
        )
        munoz_file_p21 = os.path.join(
            DATA_PATH, "theory", "munoz_2021_allgalaxies/1pt5Gpc_EOS_coeval_pow_P21.bin"
        )
        # munoz_file_errP21 = os.path.join(DATA_PATH, "theory",
        # "munoz_2021_allgalaxies/1pt5Gpc_EOS_coeval_pow_errP21.bin")
        paper_dict["linestyle"] = "solid"
        paper_dict["model"] = "AllGalaxies"
    else:  # model == "OPT"
        munoz_file_k = os.path.join(
            DATA_PATH, "theory", "munoz_2021_optimistic/600Mpc_pt0_coeval_pow_kbins.bin"
        )
        munoz_file_z = os.path.join(
            DATA_PATH, "theory", "munoz_2021_optimistic/600Mpc_pt0_coeval_pow_zlist.bin"
        )
        munoz_file_p21 = os.path.join(
            DATA_PATH, "theory", "munoz_2021_optimistic/600Mpc_pt0_coeval_pow_P21.bin"
        )
        # munoz_file_errP21 = os.path.join(DATA_PATH, "theory",
        # "munoz_2021_allgalaxies/600Mpc_pt0_coeval_pow_errP21.bin")
        paper_dict["linestyle"] = "dashdot"
        paper_dict["model"] = "AllGalaxies (OPT)"

    redshifts = np.fromfile(munoz_file_z)
    k_arr = np.fromfile(munoz_file_k)
    delta_squared_arr = np.fromfile(munoz_file_p21).reshape(
        (redshifts.size, k_arr.size)
    )

    # Sort in order of ascending redshift
    order = np.argsort(redshifts)
    redshifts = redshifts[order]
    delta_squared_arr = delta_squared_arr[order]

    redshift_ind = np.argmin(np.abs(redshifts - redshift))

    paper_dict["k"] = k_arr.tolist()
    # redshift is the 1st index, k is the 2nd
    paper_dict["delta_squared"] = delta_squared_arr[redshift_ind, :]
    paper_dict["redshift"] = np.asarray(redshifts)[redshift_ind].tolist()

    return paper_dict
