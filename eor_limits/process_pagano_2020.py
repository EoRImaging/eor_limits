#! /usr/bin/env python
# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Nichole Barry, Bryna Hazelton
# Licensed under the 2-clause BSD License
"""Process Pagano and Liu 2020 spectra for plotting."""

import os

import numpy as np

from eor_limits.data import DATA_PATH

pagano_file = os.path.join(DATA_PATH, "theory", "pagano_liu_2020.npz")


def get_pagano_2020_line(beta=1, redshift=8, linewidth=1.0):
    """
    Get a model from Mesinger et al. 2016 library.

    Parameters
    ----------
    beta : float
        Which beta  to get a model for (the closest match). beta is the correlation
        between the density and ionization field, beta=1 is an inside-out scenario
        while beta=-1 is an outside-in scenario.
    redshift :  float
        Which redshift to get a model for (the closest match).

    """
    if beta > 1 or beta < -1:
        raise ValueError("beta must be between -1 and 1.")

    model_name = "beta_" + str(beta) + "_z_" + str(redshift)

    paper_dict = {
        "author": "Pagano and Liu",
        "year": 2020,
        "model": " ".join(model_name.split("_")),
        "doi": "10.1093/mnras/staa2118",
        "type": "line",
        "marker": ".",
        "linewidth": linewidth,
        "linestyle": "-",
    }

    with np.load(pagano_file) as data:
        delta_squared_arr = data["power_spectra"]
        betas = data["betas"]
        k_arr = data["bins_k"]
        redshifts = data["redshifts"]
    data.close()

    beta_ind = np.argmin(np.abs(betas - beta))
    redshift_ind = np.argmin(np.abs(redshifts - redshift))

    paper_dict["k"] = k_arr.tolist()
    # beta is the first index, redshift is the second index
    paper_dict["delta_squared"] = delta_squared_arr[beta_ind, redshift_ind, :]
    paper_dict["redshift"] = np.asarray(redshifts)[redshift_ind].tolist()

    return paper_dict
