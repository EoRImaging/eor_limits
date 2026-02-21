"""Module defining a plotting function for EoR limits vs k and redshift."""

import logging
from itertools import chain
from pathlib import Path
from typing import Annotated, Any

import h5py
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from cyclopts import Parameter

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


def make_plot(
    # Limit plotting options
    limits: list[str] | None = None,
    base_limit_style: dict[str, Any] | None = None,
    limit_styles: dict[str, dict[str, Any]] | None = None,
    bold_limits: list[str] | None = None,
    shade_limits: float | None = 0.5,
    aspoints: list[str] | None = None,
    aslines: list[str] | None = None,
    nk_for_lines: int = 10,
    # Limits selection options
    z_range: tuple[float, float] | None = None,
    k_range: tuple[float, float] | None = None,
    delta_squared_range: tuple[float, float] | None = None,
    # Theory plotting options
    theories: list[str] | None = None,
    theory_redshifts: dict[str, list[float]] | None = None,
    base_theory_style: dict[str, Any] | None = None,
    theory_styles: dict[str, dict[str, Any]] | None = None,
    bold_theories: list[str] | None = None,
    shade_theories: float | None = 0.5,
    # Sensitivity plotting options
    sensitivities: dict | None = None,
    sensitivity_style: dict | None = None,
    # General plotting options
    colormap: str = "Spectral_r",
    fontsize: int = 15,
    fig_ratio: float | None = None,
    # Output options
    fig: Annotated[plt.Figure | None, Parameter(show=False)] = None,
    ax: Annotated[plt.Axes | None, Parameter(show=False)] = None,
    out: str | Path | None = None,
) -> plt.Figure:
    """
    Plot the current EoR Limits as a function of k and redshift.

    Parameters
    ----------
    limits : list of str
        List of limits to include in the plot. See `KNOWN_LIMITS` for available limits.
        Defaults to `None` meaning include all papers in the data folder.
    base_limit_style : dict
        Base style parameters for plotting limits, applied to all limits before any
        individual overrides. For example, `{'alpha': 0.7}` to make all limits slightly
        transparent.
    limit_styles : dict of dict
        Dictionary of style parameters for plotting limits. The keys are the limit
        keys (e.g. `'Paciga2013'`), and the values are dictionaries with style
        parameters for plotting, e.g. `{'color': 'C0', 's': 100}` for points or
        `{'color': 'C0', 'linewidth': 3}` for lines.
    bold_limits : list of str
        List of limits to bold in the legend (specified as limit keys,
        e.g. `'Paciga2013'`).
    shade_limits : float
        If not `None`, the alpha value to use for shading the area above each limit line
        (or points, if plotted as points). If `None`, no shading is applied.
    aspoints : list of str
        List of limits to plot as points instead of lines (specified as limit keys,
        e.g. `'Paciga2013'`).
        If not specified, the function will automatically determine whether to plot as
        points or lines based on the number of k values (see `nk_for_lines`).
    aslines : list of str
        List of limits to plot as lines instead of points (specified as limit keys,
        e.g. `'Paciga2013'`).
        If not specified, the function will automatically determine whether to plot as
        points or lines based on the number of k values (see `nk_for_lines`).
    nk_for_lines : int
        Threshold for the number of k values to determine whether to plot a limit as
        points or lines if not specified in `aspoints` or `aslines`. If a limit has
        more than this number of k values, it will be plotted as a line by default;
        otherwise, it will be plotted as points by default.
    z_range : tuple of float
        Tuple specifying the redshift range to include in the plot, in the form
        `(z_min, z_max)`. If not specified, all redshifts will be included.
    k_range : tuple of float
        Tuple specifying the k range to include in the plot, in the form
        `(k_min, k_max)`. If not specified, all k values will be included.
    delta_squared_range : tuple of float
        Tuple specifying the delta squared range to include in the plot, in the form
        `(delta_squared_min, delta_squared_max)`. If not specified, the range will be
         set to `[1e0, 1e6]` if theories are plotted and `[1e3, 1e6]` otherwise.
    theories : list of str
        List of theories to include in the plot. See `KNOWN_THEORIES` for available
        theories and their keys. Defaults to `None` meaning no theories are plotted.
    theory_redshifts : dict of list
        Dictionary specifying which redshifts to plot for each theory. The keys are the
         theory keys (e.g. `'Mesinger2016Faint'`), and the values are lists of redshifts
         to plot for that theory.
         If not specified, the function will plot the line closest to the center of
         the redshift range.
    base_theory_style : dict
        Base style parameters for plotting theories, applied to all theories before any
        individualoverrides. For example, `{'alpha': 0.7}` to make all theories
        slightly transparent.
    theory_styles : dict of dict
        Dictionary of style parameters for plotting theories. The keys are the theory
        keys (e.g. `'Mesinger2016Faint'`), and the values are dictionaries with
        style parameters for plotting, e.g. `{'color': 'C1', 'linestyle': '--'}`.
    bold_theories : list of str
        List of theories to bold in the legend.
    shade_theories : float
        If not `None`, the alpha value to use for shading the area below each theory
        line. If `None`, defaults to 1 divided by the number of theories.
    sensitivities : dict
        Dictionary of sensitivities to plot on the figure. The keys are labels for each
        sensitivity estimate, and the values are the file names of the
        sensitivities to plot, which must be outputs from 21cmSense v2+.
    sensitivity_style : dict
        Dictionary of style parameters for plotting sensitivities. The keys are
        labels for each sensitivity estimate, and the values are dictionaries with
        style parameters for plotting,
        e.g. `{'color': 'k', 'linestyle': '--', 'linewidth': 3}`.
        An additional key 'sensitivity_kind' can be used to specify which kind of
        sensitivity to plot, e.g. `'sample+thermal'`, `'sample'` or `'thermal'`.
    colormap : str
        Matplotlib colormap to use for coloring limits by redshift.
        Defaults to `'Spectral_r'`.
    fontsize : int
        Font size to use in the legend and axis labels. Defaults to `15`.
    fig_ratio : float
        Height to width ratio of the figure. If not specified, the height will be 1
        times the width if theories are plotted, and 0.5 times the width if no theories
        are plotted.
    fig : matplotlib.figure.Figure
        If specified, the figure to plot on. If not specified, a new figure
        will be created.
    ax : matplotlib.axes.Axes
        If specified, the axis to plot on. If not specified, a new axis
        will be created.
    out : str or Path or None
        If specified, the file name to save the figure to.
    """
    ###################################################################################
    # Set up the figure and axis
    fig_width = 25
    if theories is not None:
        fig_height = fig_width * (fig_ratio or 1)
    else:
        fig_height = fig_width * (fig_ratio or 0.5)

    if fig is None or ax is None:
        fig = plt.figure(figsize=(fig_width, fig_height))
    elif ax is not None:
        plt.sca(ax)

    ###################################################################################
    # OBSERVATIONAL LIMITS

    # Load data for limits and sort by year.
    if limits is None:
        limits = list(KNOWN_LIMITS.keys())
        limits = [load_limit_data(l).drop_nan() for l in limits]
        limits.sort(key=lambda limit: limit.year)
    else:
        limits = [load_limit_data(l).drop_nan() for l in limits]
    
    # Select the specified k and z ranges from the limits
    def _get_z_range_from_limits(limits):
        z_min = min(min(limit.data.z) for limit in limits)
        z_max = max(max(limit.data.z) for limit in limits)
        return (z_min, z_max)

    def _get_k_range_from_limits(limits):
        k_min = min(min(k) for limit in limits for k in limit.data.k)
        k_max = max(max(k) for limit in limits for k in limit.data.k)
        min_factor = 10 ** np.ceil(np.log10(k_min) * -1)
        max_factor = 10 ** np.ceil(np.log10(k_max) * -1)
        return (
            np.floor(k_min * min_factor) / min_factor,
            np.ceil(k_max * max_factor) / max_factor,
        )

    z_range = z_range or _get_z_range_from_limits(limits)
    k_range = k_range or _get_k_range_from_limits(limits)

    if delta_squared_range is None:
        if theories is not None:
            delta_squared_range = (1e0, 1e6)
        else:
            delta_squared_range = (1e3, 1e6)

    limits = select_k_and_z_ranges(limits, z_range, k_range, delta_squared_range)

    z_range = _get_z_range_from_limits(limits)  # again, since we removed some limits
    k_range = _get_k_range_from_limits(limits)  # again, since we removed some limits

    # Set up colormap for redshift.
    if z_range[0] == z_range[1]:
        z_range_use = [z_range[0] - 1, z_range[0] + 1]
    else:
        z_range_use = z_range
    norm = colors.Normalize(vmin=z_range_use[0], vmax=z_range_use[1])
    scalar_map = cmx.ScalarMappable(norm=norm, cmap=colormap)

    # Building plotting styles for each limit.
    limit_styles = build_limit_styles(
        limits, aspoints, aslines, nk_for_lines, base_limit_style, limit_styles
    )

    # Whether to bold each limit in the legend
    bold_limits = (
        bold_limits or []
    )  # equivalent to: if bold_limits is None: bold_limits = []
    limit_labels = [
        get_latex_limit_label(limit, bold=(limit.key in bold_limits))
        for limit in limits
    ]

    # Plotting the limits as points or lines, depending on the number of k values
    # or user specifications.
    limit_lines = plot_limits(
        limits,
        limit_styles,
        limit_labels,
        shade_limits,
        delta_squared_range,
        scalar_map,
    )

    ###################################################################################
    # THEORY MODELS

    # Loading data for theories
    theories = theories or []  # equivalent to: if theories is None: theories = []
    theories = [load_theory_model(theory) for theory in theories]

    # Downselecting to specified redshifts for theories,
    # or closest redshift to centre of redshift range if no redshifts specified.
    theory_redshifts = (
        theory_redshifts or {}
    )  # equivalent to: if theory_redshifts is None: theory_redshifts = {}
    new_theories = []
    for theory in theories:
        if theory.key not in theory_redshifts:
            theory_redshifts[theory.key] = [0.5 * (z_range[0] + z_range[1])]
        new_theories.extend([
            theory.select_closest_z(z) for z in theory_redshifts[theory.key]
        ])
    theories = new_theories

    # Build styles for theory lines, applying any overrides specified by the user.
    theory_styles = build_theory_styles(theories, base_theory_style, theory_styles)

    # Whether to bold each theory in the legend
    bold_theories = (
        bold_theories or []
    )  # equivalent to: if bold_theories is None: bold_theories = []
    theory_labels = [
        get_latex_theory_label(theory, bold=(theory.key in bold_theories))
        for theory in theories
    ]

    # Plotting the theory lines.
    theory_lines = plot_theories(
        theories,
        theory_styles,
        theory_labels,
        shade_theories,
        delta_squared_range,
    )

    ###################################################################################
    # SENSITIVITIES

    # If sensitivities are specified, build styles and plot them.
    sensitivities = (
        sensitivities or {}
    )  # equivalent to: if sensitivities is None: sensitivities = {}

    # Build styles for sensitivity lines, applying any overrides specified by the user.
    sensitivity_style = build_sensitivity_styles(sensitivities, sensitivity_style)

    # Plot the sensitivity curves.
    plot_sensitivities(sensitivities, sensitivity_style, fontsize)

    ###################################################################################
    # PLOT ADJUSTMENTS

    plt.rcParams.update({"font.size": fontsize})
    plt.xlabel(r"k ($h Mpc^{-1}$)", fontsize=fontsize)
    plt.ylabel(r"$\Delta^2$ ($mK^2$)", fontsize=fontsize)
    plt.yscale("log")
    plt.xscale("log")
    plt.ylim(*delta_squared_range)
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

    leg_rows = int(np.ceil(len(limit_labels) / leg_columns))

    point_size = 1 / 72.0  # typography standard (points/inch)
    font_inch = fontsize * point_size
    legend_height = (2 * leg_rows) * font_inch

    legend_height_norm = legend_height / fig_height  # 0.25

    axis_height = 3 * fontsize * point_size
    axis_height_norm = axis_height / fig_height
    plot_bottom = legend_height_norm + axis_height_norm

    plt.legend(
        limit_lines + theory_lines,
        limit_labels + theory_labels,
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
            if len(z_range) != 2:
                raise ValueError(
                    "redshift range must have 2 elements with the second element "
                    "greater than the first element."
                )
            if z_range[0] > z_range[1]:
                raise ValueError(
                    "redshift range must have 2 elements with the second element "
                    "greater than the first element."
                )
            try:
                limit = limit.select_z_range(*z_range)
            except ValueError:
                logging.getLogger("eor_limits").info(
                    f"{limit.key} skipped since its outside redshift range "
                    f"[{z_range[0]} < z < {z_range[1]}]"
                )
                continue

        if k_range is not None:
            if len(k_range) != 2:
                raise ValueError(
                    "k range must have 2 elements with the second element "
                    "greater than the first element."
                )
            if k_range[0] > k_range[1]:
                raise ValueError(
                    "k range must have 2 elements with the second element "
                    "greater than the first element."
                )
            try:
                limit = limit.select_k_range(*k_range)
            except ValueError:
                logging.getLogger("eor_limits").info(
                    f"{limit.key} skipped since its outside k range "
                    f"[{k_range[0]} < k < {k_range[1]}]"
                )
                continue

        if delta_squared_range is not None:
            if len(delta_squared_range) != 2:
                raise ValueError(
                    "delta squared range must have 2 elements with the second element "
                    "greater than the first element."
                )
            if delta_squared_range[0] > delta_squared_range[1]:
                raise ValueError(
                    "delta squared range must have 2 elements with the second element "
                    "greater than the first element."
                )
            try:
                limit = limit.select_delta_squared_range(*delta_squared_range)
            except ValueError:
                logging.getLogger("eor_limits").info(
                    f"{limit.key} skipped since its outside delta squared range "
                    f"[{delta_squared_range[0]} < delta^2 < {delta_squared_range[1]}]"
                )
                continue

        new_limits.append(limit)

    if not new_limits:
        raise ValueError(
            "No limits in specified redshift, k and/or delta squared range."
        )

    return new_limits


def build_limit_styles(
    limits: list[DataSet],
    aspoints: list[str] | None,
    aslines: list[str] | None,
    nk_for_lines: int,
    base_override: dict[str, Any] | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build a dictionary of styles to use for each limit paper."""
    aspoints = aspoints or []  # equivalent to: if aspoints is None: aspoints = []
    aslines = aslines or []  # equivalent to: if aslines is None: aslines = []
    styles = {}
    for limit in limits:
        style = {}

        # Determine whether to plot as points or lines, based on user specifications
        # and number of k values.
        if limit.key in aspoints:
            style["as_line"] = False
        elif limit.key in aslines:
            style["as_line"] = True
        else:
            maxlen = max(len(k) for k in limit.data.k)
            if maxlen <= nk_for_lines:
                style["as_line"] = False
            else:
                style["as_line"] = True

        # Set the style, if plotting as points, applying any overrides.
        if not style["as_line"]:
            style["s"] = 150
            style |= base_override if base_override else {}
            style["marker"] = DEFAULT_TELESCOPE_MARKERS.get(limit.telescope, "o")
            style |= overrides.get(f"{limit.key}", {}) if overrides else {}

        # Set the style, if plotting as lines, applying any overrides.
        else:
            style["linewidth"] = 2
            style["linestyle"] = "-"
            style |= base_override if base_override else {}
            style |= overrides.get(f"{limit.key}", {}) if overrides else {}

        styles[f"{limit.key}"] = style

    return styles


def build_theory_styles(
    theories: list[DataSet],
    base_override: dict[str, Any] | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build a dictionary of styles to use for each theory paper."""
    styles = {}
    for theory in theories:
        style = {
            "linewidth": 2,
            "linestyle": "--",
            "color": "lightsteelblue",
        }
        style |= base_override if base_override else {}
        style |= overrides.get(f"{theory.key}", {}) if overrides else {}
        styles[f"{theory.key}"] = style
    return styles


def build_sensitivity_styles(
    sensitivities: dict[str, str],
    sensitivity_style: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Build a dictionary of styles to use for each sensitivity curve."""
    styles = {}
    for name in sensitivities:
        style = {
            "sensitivity_kind": "sample+thermal",
            "linestyle": "--",
            "linewidth": 3,
            "color": "k",
        }
        style |= sensitivity_style.get(name, {}) if sensitivity_style else {}
        styles[name] = style
    return styles


def get_latex_limit_label(paper: DataSet, bold: bool = False) -> str:
    """Get a LaTeX label for a limit paper."""
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


def get_latex_theory_label(paper: DataSet, bold: bool = False) -> str:
    """Get a LaTeX label for a theory paper."""
    label_start = " $\\bf{Theory:} \\bf{" if bold else " $\\bf{Theory:} \\rm{"
    label_end = "}$"
    return (
        label_start
        + r"\ ".join(paper.telescope.split(" "))
        + r"\ ("
        + r"\ ".join(paper.author.split(" "))
        + r",\ "
        + str(paper.year)
        + ")"
        + label_end
    )


def plot_limits(
    limits: list[DataSet],
    limit_styles: dict[str, dict[str, Any]],
    limit_labels: list[str],
    shade_limits: float | None,
    delta_squared_range: tuple[float, float],
    scalar_map: cmx.ScalarMappable,
):
    """Plot limit papers on the current plot."""
    lines = []

    for limit, label in zip(limits, limit_labels, strict=True):
        logging.getLogger("eor_limits").info(f"Plotting {limit.author} {limit.year}")

        limit_style = limit_styles[limit.key]
        as_line = limit_style.pop(
            "as_line"
        )  # we pop this since it's not a valid argument
        # for plt.plot or plt.scatter

        # If we are plotting as points, we plot each redshift with specific colors
        # and making sure to meet towards the right edges to avoid overlaps.
        if not as_line:
            k = np.concatenate(limit.data.k)
            dsq = np.concatenate(limit.data.delta_squared)
            z = list(
                chain.from_iterable(
                    (
                        [z] * len(kk)
                        for z, kk in zip(limit.data.z, limit.data.k, strict=True)
                    ),
                )
            )

            line = plt.scatter(
                k,
                dsq,
                color=scalar_map.to_rgba(z),
                label=label,
                zorder=10,
                **limit_style,
            )

            if (
                shade_limits
                and limit.data.k_lower is not None
                and limit.data.k_upper is not None
            ):
                color_use = "grey"
                zorder = 0
                alpha = shade_limits

                for klow, khi, dsq in zip(
                    limit.data.k_lower,
                    limit.data.k_upper,
                    limit.data.delta_squared,
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

        # If we are plotting as lines, we need to flatten the data
        # since the k and delta_squared values are given as lists of arrays for each z.
        else:
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
                    color="black",
                    linewidth=limit_style["linewidth"] + 2,
                    zorder=2,
                )

                (this_line,) = plt.plot(
                    k_edges,
                    delta_edges,
                    color=color_val,
                    linewidth=limit_style["linewidth"],
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

        lines.append(line)

    return lines


def plot_theories(
    theories: list[DataSet],
    theory_styles: dict[str, dict[str, Any]],
    theory_labels: list[str],
    shade_theories: float | None,
    delta_squared_range: tuple[float, float],
):
    """Plot theory lines on the current plot."""
    lines = []

    if shade_theories is None:
        shade_theories = 1.0 / len(theories)

    for theory, label in zip(theories, theory_labels, strict=True):
        logging.getLogger("eor_limits").info(f"Plotting {theory.author} {theory.year}")

        theory_style = theory_styles[theory.key]

        (line,) = plt.plot(
            theory.data.k[0],
            theory.data.delta_squared[0],
            label=label,
            zorder=2,
            **theory_style,
        )

        if shade_theories:
            color_use = theory_style["color"]
            zorder = 0
            alpha = shade_theories
            plt.fill_between(
                theory.data.k[0],
                theory.data.delta_squared[0],
                delta_squared_range[0],
                color=color_use,
                alpha=alpha,
                zorder=zorder,
            )

        lines.append(line)

    return lines


def plot_sensitivities(sensitivities, sensitivity_style, fontsize):
    """Plot the sensitivity curves."""
    sensitivity_style = sensitivity_style or {}
    sensitivities = sensitivities or {}
    for indx, (name, fname) in enumerate(sensitivities.items()):
        # Set the style.
        if name in sensitivity_style:
            style = sensitivity_style[name]
        else:
            style = sensitivity_style

        sense_kind = style["sensitivity_kind"]

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
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )

        # Put the instrument name right on the plot.
        # We know the sensitivity will go up to the right, so we put it at about 2/3
        # of the way, and align it to top.
        k_ind = int(len(ks) * (0.8 - 0.1 * indx))
        plt.text(
            ks[k_ind], sense[k_ind], name, fontsize=fontsize, verticalalignment="top"
        )
