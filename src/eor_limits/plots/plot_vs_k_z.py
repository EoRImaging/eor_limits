"""Module defining a plotting function for EoR limits vs k and redshift."""

import copy
from typing import Any

import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from eor_limits.data import KNOWN_PAPERS, KNOWN_THEORIES

from ..datatypes import read_data_yaml

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


def make_plot(
    papers: list[str] | None = None,
    include_theory: bool = True,
    theory_legend: bool = True,
    theory_params: dict[str, dict[str, Any]] = default_theory_params,
    paper_redshifts: dict | None = None,
    plot_as_points: list[str] | None = None,
    delta_squared_range: tuple[float, float] | None = None,
    redshift_range: tuple[float, float] | None = None,
    k_range: tuple[float, float] | None = None,
    shade_limits: str = "generational",
    shade_theory: str = "flat",
    colormap: str = "Spectral_r",
    linewidths: dict | None = None,
    bold_papers: list[str] | None = None,
    fontsize: int = 15,
) -> plt.Figure:
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
    fontsize : float
        Font size to use on plot.
    plot_filename : str
        File name to save plot to.

    """
    if plot_as_points is None:
        plot_as_points = ["patil_2017", "mertens_2020"]

    if papers is None:
        # use all the papers. This gives weird ordering which we will fix later
        papers_sorted = False
        papers = list(KNOWN_PAPERS.keys())
    else:
        # if a list is passed in by hand, don't reorder it
        papers_sorted = True

    if delta_squared_range is None:
        if include_theory:
            delta_squared_range = (1e0, 1e6)
        else:
            delta_squared_range = (1e3, 1e6)

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
    paper_list = build_paper_info(
        papers, plot_as_points, linewidths, bold_papers, generation1
    )
    if not papers_sorted:
        paper_list.sort(key=lambda paper_list: paper_list["year"])

    if include_theory:
        theory_paper_list = load_theory_data(theory_params)

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
        redshift_list = determine_redshifts(delta_squared_range, k_range, paper_list)

        if np.min(redshift_list) < np.max(redshift_list):
            redshift_range_use = [redshift_list[0], redshift_list[-1]]
        else:
            # if only 1 redshift and no range specified, use a range of 2 centered on
            # redshift of data.
            redshift_range_use = [redshift_list[0] - 1, redshift_list[0] + 1]

        norm = colors.Normalize(vmin=redshift_range_use[0], vmax=redshift_range_use[1])
    scalar_map = cmx.ScalarMappable(norm=norm, cmap=colormap)

    fig_height = 20 if include_theory else 10
    fig_width = 25 if theory_legend else 20

    fig = plt.figure(figsize=(fig_width, fig_height))
    legend_names, lines, paper_ks, skipped_papers = plot_limit_papers(
        paper_redshifts,
        delta_squared_range,
        redshift_range,
        shade_limits,
        colormap,
        paper_list,
        norm,
        scalar_map,
    )

    if len(skipped_papers) == len(paper_list):
        raise ValueError("No papers in specified redshift and/or delta squared range.")

    if include_theory:
        theory_line_inds = plot_theory_lines(
            theory_legend,
            delta_squared_range,
            shade_theory,
            theory_paper_list,
            legend_names,
            lines,
        )
    else:
        theory_line_inds = []

    point_size = 1 / 72.0  # typography standard (points/inch)
    font_inch = fontsize * point_size

    plt.rcParams.update({"font.size": fontsize})
    plt.xlabel(r"k ($h Mpc^{-1}$)", fontsize=fontsize)
    plt.ylabel(r"$\Delta^2$ ($mK^2$)", fontsize=fontsize)
    plt.yscale("log")
    plt.xscale("log")
    plt.ylim(*delta_squared_range)

    if k_range is None:
        k_range = (np.min(paper_ks), np.max(paper_ks))
        min_factor = 10 ** np.ceil(np.log10(k_range[0]) * -1)
        max_factor = 10 ** np.ceil(np.log10(k_range[1]) * -1)
        k_range = (
            np.floor(k_range[0] * min_factor) / min_factor,
            np.ceil(k_range[1] * max_factor) / max_factor,
        )
    plt.xlim(*k_range)

    plt.tick_params(labelsize=fontsize)
    cb = plt.colorbar(
        scalar_map, ax=plt.gca(), fraction=0.1, pad=0.08, label="Redshift"
    )
    cb.ax.yaxis.set_label_position("left")
    cb.ax.yaxis.set_ticks_position("left")
    cb.set_label(label="Redshift", fontsize=fontsize)
    plt.grid(axis="y")

    leg_columns = 2 if fontsize > 20 else 3
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
    return fig


def plot_theory_lines(
    theory_legend,
    delta_squared_range,
    shade_theory,
    theory_paper_list,
    legend_names,
    lines,
):
    """Plot theory lines on the current plot."""
    theory_line_inds = []

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
    return theory_line_inds


def plot_limit_papers(
    paper_redshifts,
    delta_squared_range,
    redshift_range,
    shade_limits,
    colormap,
    paper_list,
    norm,
    scalar_map,
):
    """Plot limit papers on the current plot."""
    legend_names = []
    lines = []
    paper_ks = []
    skipped_papers = []
    for paper in paper_list:
        label_start = " $\\bf{" if paper["bold"] else " $\\rm{"
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
            skip_this_paper, these_ks, line = plot_limit_paper_as_points(
                paper_redshifts,
                delta_squared_range,
                redshift_range,
                shade_limits,
                colormap,
                norm,
                paper,
                label,
            )
        else:
            skip_this_paper, these_ks, line = plot_limit_paper_lines(
                paper_redshifts,
                delta_squared_range,
                redshift_range,
                shade_limits,
                colormap,
                norm,
                scalar_map,
                paper,
                label,
            )

        if skip_this_paper:
            skipped_papers.append(paper)
        else:
            paper_ks.extend(these_ks)
            lines.append(line)
            legend_names.append(label)

    return legend_names, lines, paper_ks, skipped_papers


def plot_limit_paper_lines(
    paper_redshifts,
    delta_squared_range,
    redshift_range,
    shade_limits,
    colormap,
    norm,
    scalar_map,
    paper,
    label,
):
    """Plot a limit paper with line data on the current plot."""
    skip_this_paper = False
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
    else:
        lines_use = np.arange(len(redshifts))

    if lines_use.size > 0:
        if len(paper_redshifts) > 0 and paper["name"] in paper_redshifts:
            redshift_array = np.asarray(redshifts)
            new_lines_use = [
                line
                for line in lines_use
                if redshift_array[line] in paper_redshifts[paper["name"]]
            ]
            lines_use = np.array(new_lines_use, dtype=int)

        for ind in lines_use:
            redshift = np.asarray(redshifts)[ind]
            these_ks = k_vals[ind]

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
                (
                    np.asarray(delta_squared[ind]),
                    np.asarray(delta_squared[ind]),
                )
            ).T.flatten()
            if paper["plot_as_point"]:
                this_line = plt.scatter(
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

                (this_line,) = plt.plot(
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
                line = this_line
                out_ks = these_ks

    else:
        skip_this_paper = True
        line = None
        out_ks = []

    return skip_this_paper, out_ks, line


def plot_limit_paper_as_points(
    paper_redshifts,
    delta_squared_range,
    redshift_range,
    shade_limits,
    colormap,
    norm,
    paper,
    label,
):
    """Plot a limit paper with point data on the current plot."""
    skip_this_paper = False
    if len(paper["redshift"]) == 1 and len(paper["delta_squared"]) > 1:
        paper["redshift"] = paper["redshift"] * len(paper["delta_squared"])
    elif len(paper["redshift"]) != len(paper["delta_squared"]):
        raise ValueError(f"{label} has the wrong number of redshift values.")
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

    if len(paper_redshifts) > 0 and paper["name"] in paper_redshifts:
        redshift_array = np.asarray(paper["redshift"])
        new_points_use = [
            point
            for point in points_use
            if redshift_array[point] in paper_redshifts[paper["name"]]
        ]
        points_use = np.array(new_points_use, dtype=int)

    if points_use.size > 0:
        these_ks = list(np.asarray(paper["k"])[points_use])

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
        if shade_limits:
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
    else:
        skip_this_paper = True
        these_ks = []
        line = None

    return skip_this_paper, these_ks, line


def determine_redshifts(delta_squared_range, k_range, paper_list):
    """Determine the redshifts from all limits."""
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
                        np.min(k_array) <= k_range[1] and np.max(k_array) >= k_range[0]
                    ):
                        redshift_list.append(elem)
                else:
                    if np.nanmin(delta_array) <= delta_squared_range[1]:
                        redshift_list.append(elem)
    return sorted(set(redshift_list))


def load_theory_data(theory_params):
    """Load the theory data."""
    theory_paper_list = []
    for theory in theory_params.values():
        if theory["paper"] in KNOWN_THEORIES:
            paper_dict = read_data_yaml(theory["paper"], theory=True)
        elif theory["paper"] == "mesinger_2016":
            from ..processors.process_mesinger_2016 import get_mesinger_2016_line

            dict_use = copy.deepcopy(theory)
            dict_use.pop("paper")
            paper_dict = get_mesinger_2016_line(**dict_use)
        elif theory["paper"] == "pagano_liu_2020":
            from ..processors.process_pagano_2020 import get_pagano_2020_line

            dict_use = copy.deepcopy(theory)
            dict_use.pop("paper")
            paper_dict = get_pagano_2020_line(**dict_use)
        elif theory["paper"] == "munoz_2021":
            from ..processors.process_munoz_2021 import get_munoz_2021_line

            dict_use = copy.deepcopy(theory)
            dict_use.pop("paper")
            paper_dict = get_munoz_2021_line(**dict_use)
        else:
            raise ValueError(
                "Theory paper " + theory["paper"] + " is not a yaml in the "
                "data/theory folder and is not a paper with a known processing "
                "module."
            )

        theory_paper_list.append(paper_dict)
    return theory_paper_list


def build_paper_info(papers, plot_as_points, linewidths, bold_papers, generation1):
    """Generate a dictionary of paper info for all papers to be plotted."""
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
    return paper_list
