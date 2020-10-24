#! /usr/bin/env python
# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Nichole Barry, Bryna Hazelton
# Licensed under the 2-clause BSD License
"""Process Mesinger et al. 2016 EOS spectra for plotting."""

import glob
import os

import numpy as np

from eor_limits.data import DATA_PATH

faint_path = os.path.join(DATA_PATH, "theory", "mesinger_2016_faint_galaxies_all")
bright_path = os.path.join(DATA_PATH, "theory", "mesinger_2016_bright_galaxies_all")


def get_mesinger_2016_line(model="faint", nf=None, redshift=None, linewidth=1.0):
    """
    Get a model from Mesinger et al. 2016 library.

    Parameters
    ----------
    model : str
        Which model to use. Options are 'faint' for faint galaxies, which is the
        fiducial model, or 'bright' for bright galaxies.
    nf : float
        Which neutral fraction to get a model for (the closest match).
        If both `redshift` and `nf` are set, `nf` is used first and `redshift` is only
        used if there are multiple models that match the `nf`. If neither `redshift` nor
        `nf` is set, the default is to find the max power across all the models.
    redshift :  float
        Which redshift to get a model for (the closest match).
        If both `redshift` and `nf` are set, `nf` is used first and `redshift` is only
        used if there are multiple models that match the `nf`. If neither `redshift` nor
        `nf` is set, the default is to find the max power across all the models.

    """
    if model not in ["faint", "bright"]:
        raise ValueError("Model must be either 'faint' or 'bright'.")

    paper_dict = {
        "author": "Mesinger",
        "year": 2016,
        "doi": "10.1093/mnras/stw831",
        "type": "line",
        "marker": ".",
        "linewidth": linewidth,
    }
    if model == "faint":
        model_files = glob.glob(os.path.join(faint_path, "*"))
        paper_dict["linestyle"] = "--"
    else:
        model_files = glob.glob(os.path.join(bright_path, "*"))
        paper_dict["linestyle"] = ":"

    if nf is None and redshift is None:
        paper_dict["linewidth"] = 0
        model_name = model + "_galaxies_max"
        for ind, this_file in enumerate(model_files):
            data_array = np.loadtxt(this_file, dtype=np.float)
            if ind == 0:
                # first column is k
                k_vals = data_array[:, 0]
                # second is power
                delta_squared = np.reshape(data_array[:, 1], (data_array[:, 1].size, 1))
            else:
                assert np.allclose(k_vals, data_array[:, 0])
                this_ds = np.reshape(data_array[:, 1], (data_array[:, 1].size, 1))
                delta_squared = np.concatenate((delta_squared, this_ds), axis=1)
        max_ds = np.max(delta_squared, axis=1)
        paper_dict["delta_squared"] = max_ds.tolist()
        paper_dict["k"] = k_vals.tolist()
        paper_dict["redshift"] = None

    else:
        filenames = [os.path.splitext(os.path.basename(f))[0] for f in model_files]
        nf_vals = [float(f.partition("nf")[2].partition("_")[0]) for f in filenames]
        redshifts = [float(f.partition("z")[2].partition("_")[0]) for f in filenames]

        if nf is not None:
            diffs = np.abs(np.asarray(nf_vals) - nf)
            closest_ind = np.nonzero(diffs == np.min(diffs))[0]
            if closest_ind.size > 1:
                model_name = model + "_galaxies_nf_" + str(nf) + "_z_" + str(redshift)
                if redshift is not None:
                    redshift_diffs = np.abs(
                        np.asarray(redshifts[closest_ind]) - redshift
                    )
                    closest_ind = np.atleast_1d(closest_ind[np.argmin(redshift_diffs)])
                else:
                    raise ValueError(
                        "More than one model matches the specified neutral fraction. "
                        "Please specify redshift as well."
                    )
            else:
                model_name = model + "_galaxies_nf_" + str(nf)
        else:
            model_name = model + "_galaxies_z_" + str(redshift)
            redshift_diffs = np.abs(np.asarray(redshifts) - redshift)
            closest_ind = np.atleast_1d(np.argmin(redshift_diffs))

        data_array = np.loadtxt(np.asarray(model_files)[closest_ind][0], dtype=np.float)
        # first column is k
        paper_dict["k"] = data_array[:, 0].tolist()
        # second is power
        paper_dict["delta_squared"] = data_array[:, 1].tolist()
        paper_dict["redshift"] = np.asarray(redshifts)[closest_ind].tolist()

    paper_dict["model"] = " ".join(model_name.split("_"))

    return paper_dict
