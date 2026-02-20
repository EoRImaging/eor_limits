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
    delta_squared_range: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
    k_range: tuple[float, float] | None = None,
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
        limits = [DataSet.load(l).drop_nan() for l in limits]
        #limits = limits.sort(key=lambda limit: limit.year)
    else:
        limits = [DataSet.load(l).drop_nan() for l in limits]
    
    # Select the specified k and z ranges from the limits
    if z_range is None:
        z_min = min(min(limit.data.z) for limit in limits)
        z_max = max(max(limit.data.z) for limit in limits)
        z_range = (z_min, z_max)
    
    if k_range is None:
        k_min = min(min(k) for limit in limits for k in limit.data.k)
        k_max = max(max(k) for limit in limits for k in limit.data.k)
        min_factor = 10 ** np.ceil(np.log10(k_min) * -1)
        max_factor = 10 ** np.ceil(np.log10(k_max) * -1)
        k_range = (
            np.floor(k_min * min_factor) / min_factor,
            np.ceil(k_max * max_factor) / max_factor,
        )
    
    if delta_squared_range is None:
        if theories is not None:
            delta_squared_range = (1e0, 1e6)
        else:
            delta_squared_range = (1e3, 1e6)
    
    limits = select_k_and_z_ranges(limits, z_range, k_range, delta_squared_range)
    
    # Set up colormap for redshift.
    if z_range[0] == z_range[1]:
        z_range_use = [z_range[0] - 1, z_range[0] + 1]
    else:
        z_range_use = z_range
    norm = colors.Normalize(vmin=z_range_use[0], vmax=z_range_use[1])
    scalar_map = cmx.ScalarMappable(norm=norm, cmap=colormap)

    # Building plotting styles for each limit.
    limit_styles = build_limit_styles(
        limits, aspoints, aslines, nk_for_lines,
        base_limit_style, limit_styles
    )
    
    # Whether to bold each limit in the legend
    bold_limits = bold_limits or [] # equivalent to: if bold_limits is None: bold_limits = []
    limit_labels = [get_latex_limit_label(limit, bold=(limit.key in bold_limits)) for limit in limits]

    # Plotting the limits as points or lines, depending on the number of k values and user specifications.
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
    theories = theories or [] # equivalent to: if theories is None: theories = []
    theories = [load_theory_model(theory) for theory in theories]
    
    # Downselecting to specified redshifts for theories,
    # or closest redshift to centre of redshift range if no redshifts specified.
    theory_redshifts = theory_redshifts or {} # equivalent to: if theory_redshifts is None: theory_redshifts = {}
    new_theory_data = []
    for theory in theories:
        if theory.key not in theory_redshifts:
            theory_redshifts[theory.key] = [0.5 * (z_range[0] + z_range[1])]
        for z in theory_redshifts[theory.key]:
            new_theory_data.append(theory.select_closest_z(z))
    theory_data = new_theory_data
    
    # Build styles for theory lines, applying any overrides specified by the user.
    theory_styles = build_theory_styles(theory_data, base_theory_style, theory_styles)
    
    # Whether to bold each theory in the legend
    bold_theories = bold_theories or [] # equivalent to: if bold_theories is None: bold_theories = []
    theory_labels = [get_latex_theory_label(theory) for theory in theories]
    
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
            try:
                limit = limit.select_z_range(*z_range)
            except ValueError:
                logger.info(
                    f"{limit.key} skipped since its outside redshift range "
                    f"[{z_range[0]} < z < {z_range[1]}]"
                )
                continue

        if k_range is not None:
            if len(k_range) != 2:
                raise ValueError(
                    "k range must have 2 elements with the second element greater than "
                    "the first element."
                )
            if k_range[0] > k_range[1]:
                raise ValueError(
                    "k range must have 2 elements with the second element greater than "
                    "the first element."
                )
            try:
                limit = limit.select_k_range(*k_range)
            except ValueError:
                logger.info(
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
                logger.info(
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
    aspoints = aspoints or [] # equivalent to: if aspoints is None: aspoints = []
    aslines = aslines or [] # equivalent to: if aslines is None: aslines = []
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
        if style["as_line"] == "points":
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
            "c": "lightsteelblue",
        }
        style |= base_override if base_override else {}
        style |= overrides.get(f"{theory.key}", {}) if overrides else {}
        styles[f"{theory.key}"] = style
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
    
    for limit, label in zip(limits, limit_labels):
        
        logger.info(f"Plotting {limit.author} {limit.year}")
        
        limit_style = limit_styles[limit.key]
        as_line = limit_style.pop("as_line") # we pop this since it's not a valid argument 
                                             # for plt.plot or plt.scatter
                                             
        # If we are plotting as lines, we need to flatten the data 
        # since the k and delta_squared values are given as lists of arrays for each redshift.
        if as_line:
            
            k = np.concatenate(limit.data.k)
            dsq = np.concatenate(limit.data.delta_squared)
            z = list(
                chain.from_iterable(
                    ([z] * len(kk) for z, kk in zip(limit.data.z, limit.data.k, strict=True)),
                )
            )

            line = plt.scatter(
                k,
                dsq,
                c=scalar_map.to_rgba(z),
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
                alpha = 0.5

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
        
        # If we are plotting as points, we plot each redshift with specific colors
        # and making sure to meet towards the right edges to avoid overlaps.
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
                    c="black",
                    linewidth=limit_style["linewidth"] + 2,
                    zorder=2,
                )

                (this_line,) = plt.plot(
                    k_edges,
                    delta_edges,
                    c=color_val,
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
    
    for theory, label in zip(theories, theory_labels):
        
        logger.info(f"Plotting {theory.author} {theory.year}")
        
        theory_style = theory_styles[theory.key]
        
        (line,) = plt.plot(
            theory.data.k[0],
            theory.data.delta_squared[0],
            zorder=2,
            **theory_style
        )
        
        if shade_theories is None:
            shade_theory_alpha = 1.0 / len(theories)
        if shade_theories:
            color_use = theory_style["c"]
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