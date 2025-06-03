#! /usr/bin/env python
# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Nichole Barry, Bryna Hazelton
# Licensed under the 2-clause BSD License
"""Code for plotting EoR Limits."""

import os

import copy
import glob
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import yaml
from typing import Tuple, Optional
import json
import h5py

from eor_limits.data import DATA_PATH

default_theory_params = {
    "munoz_2021_AllGalaxies_z8.5": {
        "paper": "munoz_2021",
        "model": "EOS",
        "redshift": 8.5,
        "linewidth": 3,
    },
    "mesinger_2016_faint_nf0.8": {
        "paper": "mesinger_2016",
        "model": "faint",
        "nf": 0.8,
        "linewidth": 2,
    },
    "mesinger_2016_bright_nf0.8": {
        "paper": "mesinger_2016",
        "model": "bright",
        "nf": 0.8,
        "linewidth": 2,
    },
    "mesinger_2016_faint_nf0.5": {
        "paper": "mesinger_2016",
        "model": "faint",
        "nf": 0.5,
        "linewidth": 3,
    },
    "mesinger_2016_bright_nf0.5": {
        "paper": "mesinger_2016",
        "model": "bright",
        "nf": 0.5,
        "linewidth": 2,
    },
    "pagano_beta1_z8.5": {"paper": "pagano_liu_2020", "beta": 1, "redshift": 8.5},
    "pagano_beta-1_z8.5": {"paper": "pagano_liu_2020", "beta": -1, "redshift": 8.5},
}


def read_data_yaml(paper_name, theory=False):
    """
    Read in the data from a paper yaml file.

    Parameters
    ----------
    paper_name : str
        Short name of paper (usually author_year) which corresponds to a file
        in the data directory named <paper_name>.yaml
    theory : bool
        Flag that this is a theory paper and so is in the theory folder.

    Returns
    -------
    dict
        Dictionary with the parsed yaml for use in the plotting code.

    """
    if theory:
        file_name = os.path.join(DATA_PATH, "theory", paper_name + ".yaml")
    else:
        file_name = os.path.join(DATA_PATH, paper_name + ".yaml")

    with open(file_name, "r") as pfile:
        paper_dict = yaml.safe_load(pfile)

    if isinstance(paper_dict["delta_squared"][0], (str,)):
        try:
            paper_dict["delta_squared"] = [
                float(val) for val in paper_dict["delta_squared"]
            ]
        except ValueError:
            val_list = []
            for val in paper_dict["delta_squared"]:
                if "**" in val:
                    val_split = val.split("**")
                    val_list.append(float(val_split[0]) ** float(val_split[1]))
                else:
                    val_list.append(float(val))
            paper_dict["delta_squared"] = val_list
    elif isinstance(paper_dict["delta_squared"][0], (list,)) and isinstance(
        paper_dict["delta_squared"][0][0], (str,)
    ):
        for ind, elem in enumerate(paper_dict["delta_squared"]):
            try:
                paper_dict["delta_squared"][ind] = [float(val) for val in elem]
            except ValueError:
                val_list = []
                for val in paper_dict["delta_squared"][ind]:
                    if "**" in val:
                        val_split = val.split("**")
                        val_list.append(float(val_split[0]) ** float(val_split[1]))
                    else:
                        val_list.append(float(val))
                paper_dict["delta_squared"][ind] = val_list

    return paper_dict


def make_plot(
    papers=None,
    include_theory=True,
    theory_legend=True,
    theory_params=default_theory_params,
    paper_redshifts=None,
    plot_as_points=["patil_2017", "mertens_2020"],
    delta_squared_range=None,
    redshift_range=None,
    k_range=None,
    shade_limits="generational",
    shade_theory="flat",
    colormap="Spectral_r",
    linewidths=None,
    bold_papers=None,
    fontsize=15,
    plot_filename=None,
    markersize=150,
    fig_ratio=None,
    sensitivities: Optional[dict] = None,
    sensitivity_style: Optional[dict] = None,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the current EoR Limits as a function of k and redshift.

    Parameters
    ----------
    papers : list of str
        List of papers to include in the plot (specified as 'author_year',
        must be present in the data folder).
        Defaults to `None` meaning include all papers in the data folder.
    include_theory : bool
        Flag to include theory lines on plots.
    theory_params : dict
        Dictionary specifying theory lines to include on the plot. Dictionary
        parameters depend on the theory paper. E.g. for lines from Mesinger et al. 2016,
        the options are 'model' which can be 'bright' or 'faint', 'nf' which specifies
        a neutral fraction and 'redshift'. See the paper specific modules for more
        examples. Only used if `include_theory` is True.
    theory_legend : bool
        Option to exclude theory lines from the legend. Used by some users who prefer
        to add the annotations on the lines by hand to improve readability.
    paper_redshifts : dict
        Dict specifying which redshifts to plot per paper. This can help simplify the
        plot so that it's not so busy. Default of 'None' means all redshifts are plotted
        for all papers. The keys should be paper names (specified as 'author_year'),
        values should be lists of redshifts to include in the plot, e.g.
        paper_redshifts = {"trott_2020": [6.5, 7.8], "li_2019": [6.5]}.
    plot_as_points : list of str
        List of papers that have a line type data model to be plotted as points rather
        that a line. This can help simplify the plot so that it's not so busy.
    delta_squared_range : list of float
        Range of delta squared values to include in plot (yaxis range). Must be
        length 2 with second element greater than first element. Defaults to [1e3, 1e6]
        if include_theory is False and [1e0, 1e6] otherwise.
    redshift_range : list of float
        Range of redshifts to include in the plot. Must be length 2 with the second
        element greater than the first element.
    k_range : list of float
        Range of ks to include in the plot. Must be length 2 with the second element
        greater than the first element.
    shade_limits : {'generational', 'alpha', False}
        How to shade above plotted limits. 'generational' shading shades dark grey for
        all generation 1 papers and light grey for later generation papers. 'alpha'
        shading shades all papers with semi-transparent grey. Setting this to False
        results in no shading.
    shade_theory : {'flat', 'alpha', False}
        How to shade below theory lines. 'flat' shading shades light grey below all
        theory lines. 'alpha' shading shades below all theory lines with
        semi-transparent grey. Setting this to False results in no shading.
    colormap : str
        Matplotlib colormap to use for redshift.
    linewidths : dict
        Dict specifying line widths to use for specific papers, to override the line
        widths specified in the paper yamls. Keys are paper names, values are desired
        line widths. Default of 'None' means use the values in the paper yamls.
    bold_papers : list of str
        List of papers to bold in caption.
    fig, ax
        The matplotlib figure and axis on which to draw the plot. If not given, will
        create one for you.
    fontsize : float
        Font size to use on plot.
    plot_filename : str
        File name to save plot to.
    markersize : int
        Size of the markers to use for point plots.
    fig_ratio : float
        Ratio of figure height to width. If None, defaults to 0.5 if theory is not
        included, and 1 if theory is included.
    sensitivities : dict
        Dictionary of sensitivities to plot on the figure. The keys are labels for each
        sensitivity estimate, and the values are the file names of the
        sensitivities to plot, which must be outputs from 21cmSense v2+.
    sensitivity_style : dict
        Dictionary of style parameters for plotting sensitivities. The keys are
        labels for each sensitivity estimate, and the values are dictionaries with
        style parameters for plotting, e.g. {'color': 'k', 'ls': '--', 'lw': 3}.
        An additional key 'sensitivity_kind' can be used to specify which kind of
        sensitivity to plot, e.g. 'sample+thermal', 'sample' or 'thermal'.
    """
    if papers is None:
        # use all the papers. This gives weird ordering which we will fix later
        papers_sorted = False
        papers = [
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(DATA_PATH, "*.yaml"))
        ]
    else:
        # if a list is passed in by hand, don't reorder it
        papers_sorted = True

    if delta_squared_range is None:
        if include_theory:
            delta_squared_range = [1e0, 1e6]
        else:
            delta_squared_range = [1e3, 1e6]

    if bold_papers is None:
        bold_papers = []
    if linewidths is None:
        linewidths = {}
    if paper_redshifts is None:
        paper_redshifts = {}
    generation1 = [
        "paciga_2013",
        "dillon_2014",
        "dillon_2015",
        "beardsley_2016",
        "patil_2017",
        "kolopanis_2019",
    ]
    paper_list = []
    for paper_name in papers:
        paper_dict = read_data_yaml(paper_name)
        paper_dict["name"] = paper_name
        if paper_name in bold_papers:
            paper_dict["bold"] = True
        else:
            paper_dict["bold"] = False
        if paper_name in plot_as_points:
            paper_dict["plot_as_point"] = True
        else:
            paper_dict["plot_as_point"] = False
        if paper_name in generation1:
            paper_dict["generation1"] = True
        else:
            paper_dict["generation1"] = False
        if paper_name in linewidths:
            paper_dict["linewidth"] = linewidths[paper_name]
        paper_list.append(paper_dict)
    if not papers_sorted:
        paper_list.sort(key=lambda paper_list: paper_list["year"])

    if include_theory:
        theory_paper_list = []
        for _, theory in theory_params.items():
            theory_paper_yamls = [
                os.path.splitext(os.path.basename(p))[0]
                for p in glob.glob(os.path.join(DATA_PATH, "theory", "*.yaml"))
            ]
            if theory["paper"] in theory_paper_yamls:
                paper_dict = read_data_yaml(theory["paper"], theory=True)
            elif theory["paper"] == "mesinger_2016":
                from eor_limits.process_mesinger_2016 import get_mesinger_2016_line

                dict_use = copy.deepcopy(theory)
                dict_use.pop("paper")
                paper_dict = get_mesinger_2016_line(**dict_use)
            elif theory["paper"] == "pagano_liu_2020":
                from eor_limits.process_pagano_2020 import get_pagano_2020_line

                dict_use = copy.deepcopy(theory)
                dict_use.pop("paper")
                paper_dict = get_pagano_2020_line(**dict_use)
            elif theory["paper"] == "munoz_2021":
                from eor_limits.process_munoz_2021 import get_munoz_2021_line

                dict_use = copy.deepcopy(theory)
                dict_use.pop("paper")
                paper_dict = get_munoz_2021_line(**dict_use)
            else:
                raise ValueError(
                    f"Theory paper {theory['paper']}  is not a yaml in the data/theory "
                    f"folder and is not a paper with a known processing module."
                )

            theory_paper_list.append(paper_dict)

    if redshift_range is not None:
        if len(redshift_range) != 2:
            raise ValueError(
                "redshift range must have 2 elements with the second element greater "
                "than the first element."
            )
        if redshift_range[0] >= redshift_range[1]:
            raise ValueError(
                "redshift range must have 2 elements with the second element greater "
                "than the first element."
            )

        norm = colors.Normalize(vmin=redshift_range[0], vmax=redshift_range[1])
    else:
        redshift_list = []
        for paper in paper_list:
            if paper["type"] == "point":
                delta_array = np.array(paper["delta_squared"])
                redshift_array = np.array(paper["redshift"])
                if redshift_array.size == 1 and delta_array.size > 1:
                    redshift_array = np.repeat(redshift_array[0], delta_array.size)
                if k_range is not None:
                    k_vals = np.asarray(paper["k"])
                    inds_use = np.nonzero(
                        (delta_array <= delta_squared_range[1])
                        & (k_vals <= k_range[1])
                        & (k_vals >= k_range[0])
                    )[0]
                else:
                    inds_use = np.nonzero(delta_array <= delta_squared_range[1])[0]
                if len(paper["redshift"]) == 1 and inds_use.size > 0:
                    inds_use = np.asarray([0])
                redshift_list += list(redshift_array[inds_use])
            else:
                if not isinstance(paper["k"][0], list):
                    redshifts = [paper["redshift"][0]]
                    k_vals = [paper["k"]]
                    delta_squared = [paper["delta_squared"]]
                else:
                    redshifts = list(np.squeeze(paper["redshift"]))
                    k_vals = paper["k"]
                    delta_squared = paper["delta_squared"]
                for ind, elem in enumerate(redshifts):
                    delta_array = np.asarray(delta_squared[ind])
                    if k_range is not None:
                        k_array = np.asarray(k_vals[ind])
                        if np.nanmin(delta_array) <= delta_squared_range[1] or (
                            np.min(k_array) <= k_range[1]
                            and np.max(k_array) >= k_range[0]
                        ):
                            redshift_list.append(elem)
                    else:
                        if np.nanmin(delta_array) <= delta_squared_range[1]:
                            redshift_list.append(elem)

        redshift_list = sorted(set(redshift_list))
        if np.min(redshift_list) < np.max(redshift_list):
            redshift_range_use = [redshift_list[0], redshift_list[-1]]
        else:
            # if only 1 redshift and no range specified, use a range of 2 centered on
            # redshift of data.
            redshift_range_use = [redshift_list[0] - 1, redshift_list[0] + 1]

        norm = colors.Normalize(vmin=redshift_range_use[0], vmax=redshift_range_use[1])
    scalar_map = cmx.ScalarMappable(norm=norm, cmap=colormap)

    fig_width = 25 if theory_legend else 20
    if include_theory:
        fig_height = fig_width * (fig_ratio or 1)
    else:
        fig_height = fig_width * (fig_ratio or 0.5)

    if fig is None or ax is None:
        fig = plt.figure(figsize=(fig_width, fig_height))
    elif ax is not None:
        plt.sca(ax)

    legend_names = []
    lines = []
    paper_ks = []
    skipped_papers = []
    for paper_i, paper in enumerate(paper_list):
        print("Processing", paper["author"], paper["year"], end="")
        if paper["bold"]:
            label_start = " $\\bf{"
        else:
            label_start = " $\\rm{"
        label_end = "}$"
        label = (
            label_start
            + r"\ ".join(paper["telescope"].split(" "))
            + r"\ ("
            + paper["author"]
            + r",\ "
            + str(paper["year"])
            + ")"
            + label_end
        )
        if paper["type"] == "point":
            if len(paper["redshift"]) == 1 and len(paper["delta_squared"]) > 1:
                paper["redshift"] = paper["redshift"] * len(paper["delta_squared"])
            elif len(paper["redshift"]) != len(paper["delta_squared"]):
                raise ValueError(f"{label} has the wrong number of redshift values.")
            delta_squared = np.asarray(paper["delta_squared"])
            if redshift_range is not None:
                redshift_array = np.asarray(paper["redshift"])
                if len(redshift_array) != len(delta_squared):
                    raise ValueError(
                        f"Paper {paper['author']} ({paper['year']}) has "
                        f"{len(redshift_array)} redshifts, but {len(delta_squared)} "
                        "power spectrum values!"
                    )

                points_use = np.where(
                    (redshift_array >= redshift_range[0])
                    & (redshift_array <= redshift_range[1])
                    & (delta_squared >= delta_squared_range[0])
                    & (delta_squared <= delta_squared_range[1])
                )[0]
            else:
                points_use = np.where(
                    (delta_squared >= delta_squared_range[0])
                    & (delta_squared <= delta_squared_range[1])
                )[0]

            if len(paper_redshifts) > 0 and paper["name"] in paper_redshifts:
                new_points_use = []
                redshift_array = np.asarray(paper["redshift"])
                for point in points_use:
                    if redshift_array[point] in paper_redshifts[paper["name"]]:
                        new_points_use.append(point)
                points_use = np.array(new_points_use, dtype=int)

            if points_use.size == 0:
                skipped_papers.append(paper)
                if redshift_range is None:
                    print(
                        ";  skipped since its outside delta^2 range "
                        f"[{delta_squared_range[0]:1.0e} < Δ² < "
                        f"{delta_squared_range[1]:1.0e}]"
                    )
                else:
                    print(
                        ";  skipped since its outside redshift/delta^2 range "
                        f"[{redshift_range[0]} < z < {redshift_range[1]}] & "
                        f"[{delta_squared_range[0]:1.0e} < Δ² < "
                        f"{delta_squared_range[1]:1.0e}]"
                    )
                continue
            else:
                plural = len(points_use) > 1
                print(f";  using {len(points_use)} point{'s' if plural > 1 else ''}.")
                paper_ks.extend(list(np.asarray(paper["k"])[points_use]))
                delta_squared = np.asarray(paper["delta_squared"])[points_use]
                line = plt.scatter(
                    np.asarray(paper["k"])[points_use],
                    delta_squared,
                    marker=paper["marker"],
                    c=np.asarray(paper["redshift"])[points_use].tolist(),
                    cmap=colormap,
                    norm=norm,
                    edgecolors="black",
                    label=label,
                    s=markersize,
                    zorder=10,
                )
                if shade_limits is not False:
                    if shade_limits == "generational":
                        if paper["generation1"]:
                            color_use = "grey"
                            zorder = 1
                            alpha = 1
                        else:
                            color_use = "lightgrey"
                            zorder = 0
                            alpha = 1
                    else:
                        color_use = "grey"
                        zorder = 0
                        alpha = 0.5
                    for index in points_use:
                        k_edges = [paper["k_lower"][index], paper["k_upper"][index]]
                        delta_edges = [
                            paper["delta_squared"][index],
                            paper["delta_squared"][index],
                        ]
                        plt.fill_between(
                            k_edges,
                            delta_edges,
                            delta_squared_range[1],
                            color=color_use,
                            alpha=alpha,
                            zorder=zorder,
                        )

                lines.append(line)
                legend_names.append(label)

        else:
            if not isinstance(paper["k"][0], list):
                redshifts = [paper["redshift"][0]]
                k_vals = [paper["k"]]
                k_lower = [paper["k_lower"]]
                k_upper = [paper["k_upper"]]
                delta_squared = [paper["delta_squared"]]
            else:
                redshifts = list(np.squeeze(paper["redshift"]))
                k_vals = paper["k"]
                k_lower = paper["k_lower"]
                k_upper = paper["k_upper"]
                delta_squared = paper["delta_squared"]

            if redshift_range is not None:
                redshift_array = np.asarray(redshifts)
                lines_use = np.where(
                    (redshift_array >= redshift_range[0])
                    & (redshift_array <= redshift_range[1])
                )[0]
                if lines_use.size == 0:
                    skipped_papers.append(paper)
                    print(
                        f";  skipped since its outside redshift/delta^2 range ["
                        f"{redshift_range[0]} < z < {redshift_range[1]}] & ["
                        f"{delta_squared_range[0]:1.0e} < Δ² < "
                        f"{delta_squared_range[1]:1.0e}]"
                    )
                    continue
            else:
                lines_use = np.arange(len(redshifts))

            print(
                f";  using {len(lines_use)} point{'s' if len(lines_use) > 1 else ''}."
            )

            if len(paper_redshifts) > 0 and paper["name"] in paper_redshifts:
                new_lines_use = []
                redshift_array = np.asarray(redshifts)
                for line in lines_use:
                    if redshift_array[line] in paper_redshifts[paper["name"]]:
                        new_lines_use.append(line)
                lines_use = np.array(new_lines_use, dtype=int)

            print(
                f";  using {len(lines_use)} point{'s' if len(lines_use) > 1 else ''}."
            )

            for ind, redshift in enumerate(np.asarray(redshifts)[lines_use]):
                paper_ks.extend(k_vals[ind])

                # Some lines have overlapping edges (since window functions can overlap)
                # That looks ugly, so we make sure we meet towards the right.
                max_right_edges = np.concatenate((k_vals[ind][1:], [np.inf]))
                min_left_edges = np.concatenate(([0], max_right_edges[:-1]))

                k_edges = np.stack(
                    (
                        np.maximum(np.asarray(k_lower[ind]), min_left_edges),
                        np.minimum(np.asarray(k_upper[ind]), max_right_edges),
                    )
                ).T.flatten()
                delta_edges = np.stack(
                    (np.asarray(delta_squared[ind]), np.asarray(delta_squared[ind]))
                ).T.flatten()
                if paper["plot_as_point"]:
                    line = plt.scatter(
                        k_vals[ind],
                        delta_squared[ind],
                        marker=paper["marker"],
                        c=np.zeros(len(k_vals[ind])) + redshift,
                        cmap=colormap,
                        norm=norm,
                        edgecolors="black",
                        label=label,
                        s=markersize,
                        zorder=10,
                    )
                else:
                    color_val = scalar_map.to_rgba(redshift)
                    # make black outline by plotting thicker black line first
                    plt.plot(
                        k_edges,
                        delta_edges,
                        c="black",
                        linewidth=paper["linewidth"] + 2,
                        zorder=2,
                    )

                    (line,) = plt.plot(
                        k_edges,
                        delta_edges,
                        c=color_val,
                        linewidth=paper["linewidth"],
                        label=label,
                        zorder=2,
                    )
                if shade_limits is not False:
                    if shade_limits == "generational":
                        if paper["generation1"]:
                            color_use = "grey"
                            zorder = 1
                            alpha = 1
                        else:
                            color_use = "lightgrey"
                            zorder = 0
                            alpha = 1
                    else:
                        color_use = "grey"
                        zorder = 0
                        alpha = 0.5
                    plt.fill_between(
                        k_edges,
                        delta_edges,
                        delta_squared_range[1],
                        color=color_use,
                        alpha=alpha,
                        zorder=zorder,
                    )
                if ind == min(lines_use):
                    lines.append(line)
                    legend_names.append(label)

    if len(skipped_papers) == len(paper_list):
        raise ValueError("No papers in specified redshift and/or delta squared range.")

    theory_line_inds = []
    if include_theory:
        # we want to supress legend labels for theories with linewidth=0
        # which are only used for shading
        # fix ordering to put them at the end
        linewidths = np.asarray([paper["linewidth"] for paper in theory_paper_list])
        ordering = np.argsort(linewidths == 0)
        theory_paper_list = [theory_paper_list[p] for p in ordering]

        for paper in theory_paper_list:
            label_start = " $\\bf{Theory:} \\rm{ "
            label_end = "}$"
            label = (
                label_start
                + r"\ ".join(paper["model"].split(" "))
                + r"\ ("
                + r"\ ".join(paper["author"].split(" "))
                + r",\ "
                + str(paper["year"])
                + ")"
                + label_end
            )
            k_vals = paper["k"]
            delta_squared = paper["delta_squared"]

            (line,) = plt.plot(
                k_vals,
                delta_squared,
                c="lightsteelblue",
                linewidth=paper["linewidth"],
                linestyle=paper["linestyle"],
                zorder=2,
            )
            if shade_theory is not False:
                if shade_theory == "flat":
                    color_use = "aliceblue"
                    zorder = 0
                    alpha = 1
                else:
                    color_use = "lightsteelblue"
                    zorder = 0
                    alpha = 1.0 / len(theory_paper_list)
                plt.fill_between(
                    k_vals,
                    delta_squared,
                    delta_squared_range[0],
                    color=color_use,
                    alpha=alpha,
                    zorder=zorder,
                )
            theory_line_inds.append(len(lines))
            lines.append(line)
            if paper["linewidth"] > 0 and theory_legend:
                legend_names.append(label)

    point_size = 1 / 72.0  # typography standard (points/inch)
    font_inch = fontsize * point_size

    sensitivity_style = sensitivity_style or {}
    sensitivities = sensitivities or {}
    for indx, (name, fname) in enumerate(sensitivities.items()):
        # Set the style.
        if name in sensitivity_style:
            style = sensitivity_style[name]
        else:
            style = sensitivity_style

        sense_kind = style.get("sensitivity_kind", "sample+thermal")

        if sense_kind not in ["sample+thermal", "sample", "thermal"]:
            raise ValueError(
                f"Invalid sensitivity kind '{sense_kind}' for {name}. "
                "Must be one of 'sample+thermal', 'sample', or 'thermal'."
            )

        # These must be outputs from 21cmSense v2+
        with h5py.File(fname, "r") as fl:
            if "k" not in fl.keys():
                raise ValueError(
                    f"{fname} is not a valid 21cmSense output: no key 'k' found"
                )
            if sense_kind not in fl.keys():
                raise IOError(f"{fname} has no key {sense_kind} for sensitivity data. ")
            ks = fl["k"][:]
            sense = fl[sense_kind][:]

        ks = ks[~np.isinf(sense)]
        sense = sense[~np.isinf(sense)]

        plt.plot(
            ks,
            sense,
            color=style.get("color", "k"),
            ls=style.get("ls", ["--", ":", "-."][indx % 3]),
            lw=style.get("lw", [3, 2, 4][indx // 3]),
        )

        # Put the instrument name right on the plot.
        # We know the sensitivity will go up to the right, so we put it at about 2/3
        # of the way, and align it to top.
        k_ind = int(len(ks) * (0.8 - 0.1 * indx))
        plt.text(
            ks[k_ind], sense[k_ind], name, fontsize=fontsize, verticalalignment="top"
        )

    plt.rcParams.update({"font.size": fontsize})
    plt.xlabel(r"k ($h Mpc^{-1}$)", fontsize=fontsize)
    plt.ylabel(r"$\Delta^2$ ($mK^2$)", fontsize=fontsize)  # noqa
    plt.yscale("log")
    plt.xscale("log")
    plt.ylim(*delta_squared_range)

    if k_range is None:
        k_range = [np.min(paper_ks), np.max(paper_ks)]
        min_factor = 10 ** np.ceil(np.log10(k_range[0]) * -1)
        max_factor = 10 ** np.ceil(np.log10(k_range[1]) * -1)
        k_range = [
            np.floor(k_range[0] * min_factor) / min_factor,
            np.ceil(k_range[1] * max_factor) / max_factor,
        ]
    plt.xlim(*k_range)

    plt.tick_params(labelsize=fontsize)
    cb = plt.colorbar(
        scalar_map, ax=plt.gca(), fraction=0.1, pad=0.08, label="Redshift"
    )
    cb.ax.yaxis.set_label_position("left")
    cb.ax.yaxis.set_ticks_position("left")
    cb.set_label(label="Redshift", fontsize=fontsize)
    plt.grid(axis="y")

    if fontsize > 25:
        leg_columns = 1
    elif fontsize > 20:
        leg_columns = 2
    else:
        leg_columns = 3

    leg_rows = int(np.ceil(len(legend_names) / leg_columns))

    legend_height = (2 * leg_rows) * font_inch

    legend_height_norm = legend_height / fig_height  # 0.25

    axis_height = 3 * fontsize * point_size
    axis_height_norm = axis_height / fig_height
    plot_bottom = legend_height_norm + axis_height_norm

    leg = plt.legend(
        lines,
        legend_names,
        bbox_to_anchor=(0.48, legend_height_norm / 2.0),
        loc="center",
        bbox_transform=fig.transFigure,
        ncol=leg_columns,
        frameon=False,
    )

    for ind in range(len(leg.legend_handles)):
        if ind not in theory_line_inds:
            leg.legend_handles[ind].set_color("gray")
    plt.subplots_adjust(bottom=plot_bottom)
    fig.tight_layout()
    if plot_filename is not None:
        plt.savefig(plot_filename)

    return fig, plt.gca()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--papers",
        type=str,
        nargs="+",
        default=None,
        help="Papers to include on plot "
        "(must be in data directory). Defaults to all papers "
        "in the data directory.",
    )
    parser.add_argument(
        "--no_theory",
        action="store_true",
        help="Flag to not plot theory lines. If True, default range is modified.",
    )
    parser.add_argument(
        "--theories",
        type=str,
        nargs="+",
        default=None,
        help="Theories to plot. Theory-specific options can be set to control which "
        "lines are drawn.",
    )
    parser.add_argument(
        "--theory_model",
        nargs="+",
        type=str,
        default=None,
        help="Model type to select from theories (e.g. 'bright' or 'faint' for "
        "Mesinger et al. 2016).",
    )
    parser.add_argument(
        "--theory_nf",
        nargs="+",
        type=str,
        default=None,
        help="Neutral fractions to select from theories.",
    )
    parser.add_argument(
        "--theory_redshift",
        nargs="+",
        type=str,
        default=None,
        help="Redshifts to select from theories.",
    )
    parser.add_argument(
        "--theory_linewidth",
        nargs="+",
        type=float,
        default=None,
        help="Linewidths for theory lines.",
    )
    parser.add_argument(
        "--no_theory_legend",
        action="store_true",
        help="Flag to exclude theory lines from the legend. Used by some users who "
        "prefer to add the annotations on the lines by hand to improve readability.",
    )
    parser.add_argument(
        "--aspoints",
        type=str,
        nargs="+",
        default=["patil_2017", "mertens_2020"],
        help="Papers to plot as points rather than lines to help simplify the plot.",
    )
    parser.add_argument(
        "--range",
        type=float,
        help="Range of Delta Squared to include on plot (yaxis range). "
        "Defaults to [1e3, 1e6] if include_theory is false and [1e0, 1e6] otherwise",
        default=None,
        nargs="+",
    )
    parser.add_argument(
        "--redshift",
        type=float,
        help="Range of redshifts to include on plot.",
        default=None,
        nargs="+",
    )
    parser.add_argument(
        "--k_range",
        type=float,
        help="Range of k values to include on plot (xaxis range).",
        default=None,
        nargs="+",
    )
    parser.add_argument(
        "--shade_limits",
        type=str,
        default="generational",
        help="Type of shading above limits to apply, one of: 'generational', 'alpha' "
        "or False.",
    )
    parser.add_argument(
        "--shade_theory",
        type=str,
        default="flat",
        help="Type of shading below theories to apply, one of: 'flat', 'alpha' "
        "or False.",
    )
    parser.add_argument(
        "--colormap", type=str, help="Matplotlib colormap to use.", default="Spectral_r"
    )
    parser.add_argument(
        "--bold",
        type=str,
        nargs="+",
        help="List of papers to bold in caption.",
        default=None,
    )
    parser.add_argument(
        "--sensitivities",
        type=str,
        nargs="+",
        default=None,
        help=(
            "List of names:::files for which to include sensitivities. "
            "Files must be 21cmSense v2+ outputs, which can be generated with "
            "``sense calc-sense --fname out.h5`` (see 21cmSense docs for more info)."
        ),
    )
    parser.add_argument(
        "--sensitivity-style",
        type=str,
        nargs="+",
        default=None,
        help=(
            "style parameters for plotting sensitivities. "
            "Format should be name:::{key:val} or just {key:val}. "
            "Each 'name' (if given) should correspond to a label in the sensitivities "
            "argument, and the values should be a JSON string with style parameters "
            "for plotting, e.g. {'color': 'k', 'ls': '--', 'lw': 3}. "
            "An additional key 'sensitivity_kind' can be used to specify which kind of "
            "sensitivity to plot, e.g. 'sample+thermal', 'sample' or 'thermal'. "
            "If no name is given, the style will be applied to all sensitivities. "
            "If no style is given, the default style will be used.",
        ),
    )
    parser.add_argument("--fontsize", type=int, help="Font size to use.", default=15)
    parser.add_argument(
        "--height-ratio",
        type=float,
        help="defines the height of the figure",
        default=None,
    )
    parser.add_argument(
        "--markersize", type=int, default=150, help="size of the markers"
    )
    parser.add_argument(
        "--file",
        type=str,
        dest="filename",
        help="Filename to save plot to.",
        default="eor_limits.pdf",
    )

    args = parser.parse_args()

    if args.shade_limits == "False":
        args.shade_limits = False
    if args.shade_theory == "False":
        args.shade_theory = False

    if args.theories is not None:
        if args.theory_nf is None:
            args.theory_nf = [None]
        else:
            args.theory_nf = [
                float(val) if val != "None" else None for val in args.theory_nf
            ]
        if args.theory_redshift is None:
            args.theory_redshift = [None]
        if args.theory_model is None:
            args.theory_model = [None]

        theory_params = {}
        num_theories = len(args.theories)
        num_models = len(args.theory_model)
        num_nf = len(args.theory_nf)
        num_redshift = len(args.theory_redshift)
        num_theory_lines = max([num_theories, num_models, num_nf, num_redshift])
        if num_theory_lines > 1:
            if num_theories == 1:
                args.theories = args.theories * num_theory_lines
            elif num_theories != num_theory_lines:
                raise ValueError(
                    "Number of theories must be one or match the max length of "
                    "theory_model, theory_nf or theory_redshift."
                )
            if num_models == 1:
                args.theory_model = args.theory_model * num_theory_lines
            elif num_models != num_theory_lines:
                raise ValueError(
                    "Number of theory_models must be one or match the max length of "
                    "theories, theory_nf or theory_redshift."
                )
            if num_nf == 1:
                args.theory_nf = args.theory_nf * num_theory_lines
            elif num_nf != num_theory_lines:
                raise ValueError(
                    "Number of theory_nfs must be one or match the max length of "
                    "theories, theory_model or theory_redshift."
                )
            if num_redshift == 1:
                args.theory_redshift = args.theory_redshift * num_theory_lines
            elif num_redshift != num_theory_lines:
                raise ValueError(
                    "Number of theory_redshifts must be one or match the max length of "
                    "theories, theory_model or theory_nf."
                )

            # Ensure it's interpreted as a number
            args.theory_redshift = [float(z) for z in args.theory_redshift]

            if args.theory_linewidth is not None:
                if len(args.theory_linewidth) == 1:
                    args.theory_linewidth = args.theory_linewidth * num_theory_lines
                elif len(args.theory_linewidth) != num_theory_lines:
                    raise ValueError(
                        "Number of theory lines must be one or match the max length of "
                        "theories, theory_model, theory_nf or theory_redshift."
                    )
        for index, (theory, model, nf, redshift) in enumerate(
            zip(args.theories, args.theory_model, args.theory_nf, args.theory_redshift)
        ):
            name = f"{theory}_{model}_nf_{nf}_z_{redshift}"
            theory_params[name] = {
                "paper": theory,
                "model": model,
                "nf": float(nf) if nf is not None else None,
                "redshift": float(redshift) if redshift is not None else None,
            }
            if args.theory_linewidth is not None:
                theory_params[name]["linewidth"] = args.theory_linewidth[index]
    else:
        if args.theory_nf or args.theory_redshift or args.theory_model:
            raise ValueError(
                "You passed a theory nf/redshift/model but no theory itself!"
            )

        theory_params = default_theory_params

    # Process sensitivity arguments.
    if args.sensitivities:
        sensitivities = dict(param.split(":::") for param in args.sensitivities)
    else:
        sensitivities = None

    if args.sensitivity_style:
        if args.sensitivity_style.strip().startswith("{"):
            sensitivity_style = json.loads(args.sensitivity_style)
        else:
            sensitivity_style = dict(
                param.split(":::") for param in args.sensitivity_style
            )
            sensitivity_style = {k: json.loads(v) for k, v in sensitivity_style.items()}
    else:
        sensitivity_style = None

    fig, ax = make_plot(
        papers=args.papers,
        include_theory=not args.no_theory,
        theory_legend=not args.no_theory_legend,
        theory_params=theory_params,
        plot_as_points=args.aspoints,
        delta_squared_range=args.range,
        redshift_range=args.redshift,
        k_range=args.k_range,
        shade_limits=args.shade_limits,
        shade_theory=args.shade_theory,
        colormap=args.colormap,
        bold_papers=args.bold,
        fontsize=args.fontsize,
        sensitivities=sensitivities,
        sensitivity_style=sensitivity_style,
        markersize=args.markersize,
        fig_ratio=args.height_ratio,
    )

    fig.savefig(args.filename)
