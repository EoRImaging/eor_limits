"""Module defining a plotting function for EoR limits vs k and redshift."""

import logging
from itertools import chain
from typing import Any, Literal
from pathlib import Path

import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from ._data_loading import load_limit_data, load_theory_model
from ._datatypes import DataSet
from .data import KNOWN_LIMITS

DEFAULT_TELESCOPE_MARKERS = {
    "PAPER": "o",
    "MWA": "s",
    "LOFAR": "D",
    "HERA": "H",
    "GMRT": "v",
}

app = App()

logger = logging.getLogger("eor_limits")

@app.command
def make_plot(
    limits: list[str] | None = None,
    delta_squared_range: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
    k_range: tuple[float, float] | None = None,
    base_limit_style: dict[str, Any] | None = None,
    limit_styles: dict[str, dict[str, Any]] | None = None,
    limit_linewidths: dict | None = None,
    bold_limits: list[str] | None = None,
    shade_limits: float | None = 0.5,
    
    theories: list[str] | None = None,
    theory_redshifts: dict[str, list[float]] | None = None,
    theory_linewidths: dict[str, float] | None = None,
    shade_theory_alpha: float | None = 0.5,
    
    sensitivities: dict | None = None,
    sensitivity_style: dict | None = None,

    colormap: str = "Spectral_r",
    fontsize: int = 15,
    fig_ratio: float | None = None,
    aspoints: list[str] | None = None,
    aslines: list[str] | None = None,
    nk_for_lines: int = 10,
    
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    out: Path | None = None,
) -> plt.Figure:
    """
    Plot the current EoR Limits as a function of k and redshift.

    Parameters
    ----------
    limits : list of str
        List of limits to include in the plot (specified as 'AuthorYear').
        These must be present in the data folder.
        Defaults to `None` meaning include all papers in the data folder.
    theories
        Theories to plot, (specified as a list of theory paper keys.
        These must be present in the theory folder.
        Defaults to `None` meaning no theories are plotted.
    theory_redshifts : list of float
        List specifying which redshifts to plot for theory lines. The default is to use
        the closest redshift to the centre of the prescribed redshift range. Multiple
        redshifts can be used.
    delta_squared_range : list of float
        Range of delta squared values to include in plot (yaxis range). Must be
        length 2 with second element greater than first element. Defaults to [1e3, 1e6]
        if theories are not included and [1e0, 1e6] otherwise.
    z_range : list of float
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
    bold_limits : list of str
        List of limits to bold in caption.
        List of papers to bold in caption.
    fontsize : float
        Font size to use on plot.
    plot_filename : str
        File name to save plot to.

    """
    
    if aspoints is None:
        aspoints = []
    if aslines is None:
        aslines = []

    if limits is None:
        # use all the papers. This gives weird ordering which we will fix later
        limits_sorted = False
        limits = list(KNOWN_LIMITS.keys())
    else:
        # if a list is passed in by hand, don't reorder it
        limits_sorted = True

    if delta_squared_range is None:
        if theories is not None:
            delta_squared_range = (1e0, 1e6)
        else:
            delta_squared_range = (1e3, 1e6)

    if bold_limits is None:
        bold_limits = []
    if limit_linewidths is None:
        limit_linewidths = {}
    if theory_linewidths is None:
        theory_linewidths = {}

    limits = [load_limit_data(p).drop_nan() for p in papers]

    if not limits_sorted:
        limits.sort(key=lambda limit: limit.year)

    limits = select_k_and_z_ranges(limits, z_range, k_range, delta_squared_range)
    actual_z_range = get_total_z_range(limits, z_range)

    if not limits:
        raise ValueError(
            "No limits in specified redshift, k and/or delta squared range."
        )

    # Build styles for each paper, which we will use for plotting and the legend.
    # This also applies any overrides specified by the user.
    points_or_lines = get_points_or_lines(limits, nk_for_lines, aspoints, aslines)
    limit_styles = build_limit_styles(
        limits, points_or_lines, base_limit_style, limit_styles
    )

    norm = set_cmap_norm_via_zrange(actual_z_range)
    scalar_map = cmx.ScalarMappable(norm=norm, cmap=colormap)

    fig_width = 25
    if theories is not None:
        fig_height = fig_width * (fig_ratio or 1)
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
    legend_names, lines, paper_ks = plot_limit_papers(
        limits,
        limit_styles,
        bold_limits,
        shade_limits,
        colormap,
        norm,
        scalar_map,
        points_or_lines,
        delta_squared_range,
    )

    if theories is not None:
        theory_data = [load_theory_model(theory) for theory in theories]

        # Each theory has _all_ redshifts in it. We need to downselect
        # to those specified by the user, or the closest redshift to the centre of the
        # redshift range if no redshifts are specified.
        if theory_redshifts is not None:
            # theory_redshifts is now a dict mapping theory names to lists of redshifts
            new_theory_data = []
            for data in theory_data:
                if data.key in theory_redshifts:
                    for z in theory_redshifts[data.key]:
                        new_theory_data.append(data.select_closest_z(z))
                else:
                    # Use closest to centre if no redshifts specified for this theory
                    z_centre = 0.5 * (actual_z_range[0] + actual_z_range[1])
                    new_theory_data.append(data.select_closest_z(z_centre))
            theory_data = new_theory_data
        else:
            z_centre = 0.5 * (actual_z_range[0] + actual_z_range[1])
            theory_data = [data.select_closest_z(z_centre) for data in theory_data]

        theory_styles = build_theory_styles(theory_data, theory_linewidths)

        theory_lines, theory_labels = plot_theory_lines(
            delta_squared_range,
            shade_theory_alpha,
            theory_data,
            theory_styles,
        )
    else:
        theory_lines, theory_labels = [], []

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

    plt.legend(
        lines + theory_lines,
        legend_names + theory_labels,
        bbox_to_anchor=(0.48, legend_height_norm / 2.0),
        loc="center",
        bbox_transform=fig.transFigure,
        ncol=leg_columns,
        frameon=False,
    )

    plt.subplots_adjust(bottom=plot_bottom)
    fig.tight_layout()
    
    if out is not None:
        fig.savefig(out)
        return None
    else:
        return fig


def select_k_and_z_ranges(
    limits: list[DataSet],
    z_range: tuple[float, float] | None,
    k_range: tuple[float, float] | None,
    delta_squared_range: tuple[float, float] | None,
) -> list[DataSet]:
    """Select the specified k and redshift ranges from the limits."""
    new_limits = []
    for limit in limits:
        if z_range is not None:
            try:
                limit = limit.select_z_range(*z_range)
            except ValueError:
                logger.info(
                    f"{limit.key} skipped since its outside redshift range "
                    f"[{z_range[0]} < z < {z_range[1]}]"
                )
                continue

        if k_range is not None:
            try:
                limit = limit.select_k_range(*k_range)
            except ValueError:
                logger.info(
                    f"{limit.key} skipped since its outside k range "
                    f"[{k_range[0]} < k < {k_range[1]}]"
                )
                continue

        if delta_squared_range is not None:
            try:
                limit = limit.select_delta_squared_range(*delta_squared_range)
            except ValueError:
                logger.info(
                    f"{limit.key} skipped since its outside delta squared range "
                    f"[{delta_squared_range[0]} < delta^2 < {delta_squared_range[1]}]"
                )
                continue

        new_limits.append(limit)
    return new_limits


def set_cmap_norm_via_zrange(z_range):
    """Set the colormap normalization based on redshift range."""
    if len(z_range) != 2:
        raise ValueError(
            "redshift range must have 2 elements with the second element greater "
            "than the first element."
        )
    if z_range[0] > z_range[1]:
        raise ValueError(
            "redshift range must have 2 elements with the second element greater "
            "than the first element."
        )

    if z_range[0] == z_range[1]:
        z_range_use = [z_range[0] - 1, z_range[0] + 1]
    else:
        z_range_use = z_range

    return colors.Normalize(vmin=z_range_use[0], vmax=z_range_use[1])


def plot_sensitivities(fontsize, sensitivities, sensitivity_style):
    """Plot the sensitivity curves."""
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
            if "k" not in fl:
                raise ValueError(
                    f"{fname} is not a valid 21cmSense output: no key 'k' found"
                )
            if sense_kind not in fl:
                raise OSError(f"{fname} has no key {sense_kind} for sensitivity data. ")
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


def plot_theory_lines(
    delta_squared_range: tuple[float, float],
    shade_theory_alpha: float,
    theory_paper_list: list[DataSet],
    theory_styles: dict[str, dict[str, Any]],
):
    """Plot theory lines on the current plot."""
    labels = []
    lines = []
    if shade_theory_alpha is None:
        shade_theory_alpha = 1.0 / len(theory_paper_list)
    for paper in theory_paper_list:
        label = get_latex_theory_label(paper)

        (line,) = plt.plot(
            paper.data.k[0],
            paper.data.delta_squared[0],
            zorder=2,
            **theory_styles.get(paper.key, {}),
        )

        if shade_theory_alpha:
            color_use = theory_styles.get(paper.key, {})["c"]
            zorder = 0
            alpha = shade_theory_alpha
            plt.fill_between(
                paper.data.k[0],
                paper.data.delta_squared[0],
                delta_squared_range[0],
                color=color_use,
                alpha=alpha,
                zorder=zorder,
            )

        lines.append(line)
        labels.append(label)

    return lines, labels


def build_limit_styles(
    limits: list[DataSet],
    lines_or_points: dict[str, Literal["points", "lines"]],
    base_override: dict[str, Any] | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build a dictionary of styles to use for each limit paper."""
    styles = {}
    for limit in limits:
        if lines_or_points[limit.key] == "points":
            style = {"s": 150}
            style |= base_override if base_override else {}

            style["marker"] = DEFAULT_TELESCOPE_MARKERS.get(limit.telescope, "o")

            style |= overrides.get(f"{limit.key}", {}) if overrides else {}

        else:
            style = {
                "linewidth": 2,
                "linestyle": "-",
            }
            style |= base_override if base_override else {}
            style |= overrides.get(f"{limit.key}", {}) if overrides else {}

        styles[f"{limit.key}"] = style
    return styles


def build_theory_styles(
    theory_papers: list[DataSet],
    linewidths: dict | None = None,
) -> dict[str, dict[str, Any]]:
    """Build a dictionary of styles to use for each theory paper."""
    styles = {}
    for paper in theory_papers:
        style = {
            "linewidth": 2,
            "linestyle": "--",
            "c": "lightsteelblue",
        }
        if linewidths and paper.key in linewidths:
            style["linewidth"] = linewidths[paper.key]
        styles[paper.key] = style
    return styles


def get_latex_paper_label(paper: DataSet, bold: bool = False) -> str:
    """Get a LaTeX label for a paper."""
    label_start = " $\\bf{" if bold else " $\\rm{"
    label_end = "}$"
    return (
        label_start
        + r"\ ".join(paper.telescope.split(" "))
        + r"\ ("
        + paper.author
        + r",\ "
        + str(paper.year)
        + ")"
        + label_end
    )


def get_latex_theory_label(paper: DataSet) -> str:
    """Get a LaTeX label for a theory paper."""
    return (
        " $\\bf{Theory:} \\rm{ "
        + r"\ ".join(paper.telescope.split(" "))
        + r"\ ("
        + r"\ ".join(paper.author.split(" "))
        + r",\ "
        + str(paper.year)
        + ")}$"
    )


def get_points_or_lines(
    limits,
    nk_for_lines: int,
    aspoints: list[str] | None,
    aslines: list[str] | None,
) -> dict[str, Literal["points", "lines"]]:
    """Determine whether to plot each limit paper as points or lines."""
    out = {}
    for limit in limits:
        if limit.key in aspoints:
            out[limit.key] = "points"
        elif limit.key in aslines:
            out[limit.key] = "lines"
        else:
            maxlen = max(len(k) for k in limit.data.k)
            if maxlen <= nk_for_lines:
                out[limit.key] = "points"
            else:
                out[limit.key] = "lines"

    return out


def plot_limit_papers(
    limits: list[DataSet],
    limit_styles: dict[str, dict[str, Any]],
    bold_limits: list[str],
    shade_limits,
    colormap,
    norm,
    scalar_map,
    points_or_lines: dict[str, Literal["points", "lines"]],
    delta_squared_range,
):
    """Plot limit papers on the current plot."""
    legend_names = []
    lines = []
    paper_ks = []
    for paper in limits:
        logger.info(f"Plotting {paper.author} {paper.year}")
        label = get_latex_paper_label(paper, bold=paper.key in bold_limits)
        if points_or_lines[paper.key] == "points":
            these_ks, line = plot_limit_paper_as_points(
                shade_limits,
                colormap,
                norm,
                paper,
                label,
                limit_style=limit_styles.get(paper.key, {}),
                delta_squared_range=delta_squared_range,
            )
        else:
            these_ks, line = plot_limit_paper_lines(
                paper,
                shade_limits=shade_limits,
                scalar_map=scalar_map,
                label=label,
                style=limit_styles[paper.key],
                delta_squared_range=delta_squared_range,
            )

        paper_ks.extend(these_ks)
        lines.append(line)
        legend_names.append(label)

    return legend_names, lines, paper_ks


def plot_limit_paper_lines(
    limit: DataSet,
    shade_limits: float,
    scalar_map,
    label: str,
    style: dict[str, Any],
    delta_squared_range: tuple[float, float],
):
    """Plot a limit paper with line data on the current plot."""
    for ind, redshift in enumerate(limit.data.z):
        k_vals = limit.data.k[ind]
        delta_squared = limit.data.delta_squared[ind]
        k_lower = (
            limit.data.k_lower[ind]
            if limit.data.k_lower is not None
            else [0] * len(k_vals)
        )
        k_upper = (
            limit.data.k_upper[ind]
            if limit.data.k_upper is not None
            else [np.inf] * len(k_vals)
        )

        # Some lines have overlapping edges (since window functions can overlap)
        # That looks ugly, so we make sure we meet towards the right.
        max_right_edges = np.concatenate((k_vals[1:], [np.inf]))
        min_left_edges = np.concatenate(([0], max_right_edges[:-1]))

        k_edges = np.stack((
            np.maximum(np.asarray(k_lower), min_left_edges),
            np.minimum(np.asarray(k_upper), max_right_edges),
        )).T.flatten()
        delta_edges = np.stack((
            np.asarray(delta_squared),
            np.asarray(delta_squared),
        )).T.flatten()

        color_val = scalar_map.to_rgba(redshift)
        # make black outline by plotting thicker black line first
        plt.plot(
            k_edges,
            delta_edges,
            c="black",
            linewidth=style["linewidth"] + 2,
            zorder=2,
        )

        (this_line,) = plt.plot(
            k_edges,
            delta_edges,
            c=color_val,
            linewidth=style["linewidth"],
            label=label,
            zorder=2,
        )
        if shade_limits:
            color_use = "grey"
            zorder = 0
            alpha = shade_limits
            plt.fill_between(
                k_edges,
                delta_edges,
                delta_squared_range[1],
                color=color_use,
                alpha=alpha,
                zorder=zorder,
            )

        if ind == 0:
            line = this_line
            out_ks = k_vals

    return out_ks, line


def plot_limit_paper_as_points(
    shade_limits: bool,
    colormap: str,
    norm,
    paper: DataSet,
    label: str,
    limit_style: dict[str, Any],
    delta_squared_range: tuple[float, float],
):
    """Plot a limit paper with point data on the current plot."""
    # flatten all the data (remember that at each z, the size of k might be different)
    k = np.concatenate(paper.data.k)
    dsq = np.concatenate(paper.data.delta_squared)
    z = list(
        chain.from_iterable(
            ([z] * len(kk) for z, kk in zip(paper.data.z, paper.data.k, strict=True)),
        )
    )

    line = plt.scatter(
        k,
        dsq,
        c=z,
        cmap=colormap,
        norm=norm,
        label=label,
        zorder=10,
        **limit_style,
    )

    if (
        shade_limits
        and paper.data.k_lower is not None
        and paper.data.k_upper is not None
    ):
        color_use = "grey"
        zorder = 0
        alpha = 0.5

        for klow, khi, dsq in zip(
            paper.data.k_lower,
            paper.data.k_upper,
            paper.data.delta_squared,
            strict=True,
        ):
            k_edges = np.concatenate((klow, [khi[-1]]))
            delta_edges = np.concatenate((dsq, dsq[-1:]))
            plt.fill_between(
                k_edges,
                delta_edges,
                delta_squared_range[1],
                color=color_use,
                alpha=alpha,
                zorder=zorder,
            )

    return k, line


def get_total_z_range(paper_list, z_range):
    """Get the total redshift range across all papers."""
    if z_range is not None:
        return z_range
    minz = min(min(paper.data.z) for paper in paper_list)
    maxz = max(max(paper.data.z) for paper in paper_list)
    return (minz, maxz)
