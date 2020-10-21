#! /usr/bin/env python
# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Nichole Barry, Bryna Hazelton
# Licensed under the 2-clause BSD License
"""Code for plotting EoR Limits."""

import glob
import os

import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

from eor_limits.data import DATA_PATH


def read_data_yaml(paper_name):
    """
    Read in the data from a paper yaml file.

    Parameters
    ----------
    paper_name : str
        Short name of paper (usually author_year) which corresponds to a file
        in the data directory named <paper_name>.yaml

    Returns
    -------
    dict
        Dictionary with the parsed yaml for use in the plotting code.

    """
    file_name = os.path.join(DATA_PATH, paper_name + ".yaml")

    with open(file_name, "r") as pfile:
        paper_dict = yaml.safe_load(pfile)

    if isinstance(paper_dict["delta_squared"][0], (str,)):
        try:
            paper_dict["delta_squared"] = [
                float(val) for val in paper_dict["delta_squared"]
            ]
        except (ValueError):
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
            except (ValueError):
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
    plot_as_points=["patil_2017", "mertens_2020"],
    plot_filename="eor_limits.pdf",
    delta_squared_range=[1e3, 1e6],
    redshift_range=None,
    k_range=None,
    shade_limits="generational",
    colormap="Spectral_r",
    bold_papers=None,
    fontsize=15,
):
    """
    Plot the current EoR Limits as a function of k and redshift.

    Parameters
    ----------
    papers : list of str
        List of papers to include in the plot (specified as 'author_year',
        must be present in the data folder).
        Defaults to `None` meaning include all papers in the data folder.
    plot_as_points : list of str
        List of papers that have a line type data model to be plotted as points rather
        that a line.
    delta_squared_range : list of float
        Range of delta squared values to include in plot (yaxis range). Must be
        length 2 with second element greater than first element.
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
    colormap : str
        Matplotlib colormap to use for redshift.
    plot_filename : str
        File name to save plot to.
    bold_papers : list of str
        List of papers to bold in caption.

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

    if bold_papers is None:
        bold_papers = []
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
        paper_list.append(paper_dict)
    if not papers_sorted:
        paper_list.sort(key=lambda paper_list: paper_list["year"])

    if redshift_range is not None:
        norm = colors.Normalize(vmin=redshift_range[0], vmax=redshift_range[1])
    else:
        redshift_list = []
        for paper in paper_list:
            if paper["type"] == "point":
                delta_array = np.array(paper["delta_squared"])
                paper_redshifts = np.array(paper["redshift"])
                if paper_redshifts.size == 1 and delta_array.size > 1:
                    paper_redshifts = np.repeat(paper_redshifts[0], delta_array.size)
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
                redshift_list += list(paper_redshifts[inds_use])
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
        norm = colors.Normalize(vmin=redshift_list[0], vmax=redshift_list[-1])
    scalar_map = cmx.ScalarMappable(norm=norm, cmap=colormap)

    fig_height = 10
    fig_width = 20
    fig = plt.figure(figsize=(fig_width, fig_height))
    legend_names = []
    lines = []
    paper_ks = []
    skipped_papers = []
    for paper_i, paper in enumerate(paper_list):
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
            delta_squared = np.asarray(paper["delta_squared"])
            if redshift_range is not None:
                redshift_array = np.asarray(paper["redshift"])
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

            if points_use.size == 0:
                skipped_papers.append(paper)
                continue
            else:
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
                    s=150,
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
                    continue
            else:
                lines_use = np.arange(len(redshifts))

            for ind, redshift in enumerate(np.asarray(redshifts)[lines_use]):
                paper_ks.extend(k_vals[ind])

                k_edges = np.stack(
                    (np.asarray(k_lower[ind]), np.asarray(k_upper[ind]))
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
                        s=150,
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
                if ind == 0:
                    lines.append(line)
        legend_names.append(label)

    point_size = 1 / 72.0  # typography standard (points/inch)
    font_inch = fontsize * point_size

    plt.rcParams.update({"font.size": fontsize})
    plt.xlabel("k ($h Mpc^{-1}$)", fontsize=fontsize)
    plt.ylabel("$\Delta^2$ ($mK^2$)", fontsize=fontsize)  # noqa
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
    cb = plt.colorbar(fraction=0.1, pad=0.08, label="Redshift")
    cb.ax.yaxis.set_label_position("left")
    cb.ax.yaxis.set_ticks_position("left")
    cb.set_label(label="Redshift", fontsize=fontsize)
    plt.grid(axis="y")

    if fontsize > 20:
        leg_columns = 2
    else:
        leg_columns = 3

    leg_rows = int(np.ceil(len(paper_list) / leg_columns))

    legend_height = (2 * leg_rows) * font_inch

    legend_height_norm = legend_height / fig_height  # 0.25

    axis_height = 3 * fontsize * point_size
    axis_height_norm = axis_height / fig_height
    plot_bottom = legend_height_norm + axis_height_norm

    leg = plt.legend(
        lines,
        legend_names,
        bbox_to_anchor=(0.45, legend_height_norm / 2.0),
        loc="center",
        bbox_transform=fig.transFigure,
        ncol=leg_columns,
        frameon=False,
    )

    for ind in range(len(leg.legendHandles)):
        leg.legendHandles[ind].set_color("gray")
    plt.subplots_adjust(bottom=plot_bottom)
    fig.tight_layout()
    plt.savefig(plot_filename)


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
        "--file",
        type=str,
        dest="filename",
        help="Filename to save plot to.",
        default="eor_limits.pdf",
    )
    parser.add_argument(
        "--aspoints",
        type=str,
        nargs="+",
        default=["patil_2017", "mertens_2020"],
        help="Papers to plot as points rather than lines.",
    )
    parser.add_argument(
        "--range",
        type=float,
        help="Range of Delta Squared to include on plot (yaxis range).",
        default=[1e3, 1e6],
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
        "--shading",
        type=str,
        default="generational",
        help="Type of shading to apply, one of: 'generational', 'alpha' or False.",
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
    parser.add_argument("--fontsize", type=int, help="Font size to use.", default=15)

    args = parser.parse_args()

    if args.shading == "False":
        args.shading = False

    make_plot(
        papers=args.papers,
        plot_as_points=args.aspoints,
        delta_squared_range=args.range,
        redshift_range=args.redshift,
        k_range=args.k_range,
        shade_limits=args.shading,
        colormap=args.colormap,
        plot_filename=args.filename,
        bold_papers=args.bold,
        fontsize=args.fontsize,
    )
