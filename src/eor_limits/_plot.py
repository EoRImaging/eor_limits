"""Module defining a plotting function for EoR limits vs k and redshift."""

import json
import logging
from collections.abc import Sequence
from itertools import chain
from pathlib import Path
from typing import Annotated, Any, Literal

import h5py
import matplotlib.cm as cmx
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from cyclopts import Parameter, Token
from matplotlib import colors
from scipy import interpolate

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


def _json_str_to_dict(type_, tokens: Sequence[Token]) -> dict:
    """Convert a JSON string to a dictionary."""
    try:
        json_str = tokens[0].value
        return json.loads(json_str) if json_str else {}
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {json_str}") from e

StrList = Annotated[list[str] | None, Parameter(consume_multiple=True)]
JsonDict = Annotated[dict[str, Any] | None, Parameter(converter=_json_str_to_dict)]
JsonNestedDict = Annotated[
    dict[str, dict[str, Any]] | None, Parameter(converter=_json_str_to_dict)
]


def plot_vs_z(
    # Limit plotting options
    limits: StrList = None,
    *,
    base_limit_style: JsonDict = None,
    limit_styles: JsonNestedDict = None,
    bold_limits: StrList = None,
    shade_limits: bool = False,
    aspoints: StrList = None,
    aslines: StrList = None,
    nz_for_lines: int = 10,
    # Limits selection options
    z_range: tuple[float, float] | None = None,
    delta_squared_range: tuple[float, float] | None = None,
    # Theory plotting options
    theories: StrList = None,
    theory_k: float = 0.25,
    base_theory_style: JsonDict = None,
    theory_styles: JsonNestedDict = None,
    bold_theories: StrList = None,
    shade_theories: bool = True,
    # Sensitivity plotting options
    sensitivities: dict | None = None,
    sensitivity_style: dict | None = None,
    # General plotting options
    color_by: Literal["year", "k"] = "k",
    show_colorbar: bool = True,
    colormap: str = "viridis",
    legend_labeler: JsonDict = None,
    k_labels: Literal["legend", "title"] | None = "None",
    legend_ncols: int | None = None,
    fontsize: float | None = None,
    fig_width: float | None = None,
    fig_ratio: float | None = None,
    publication: bool = False,
    publication_width: Literal["single", "double"] | None = None,
    publication_font: Literal["times", "dejavu"] | None = None,
    # Output options
    fig: Annotated[plt.Figure | None, Parameter(show=False)] = None,
    ax: Annotated[plt.Axes | None, Parameter(show=False)] = None,
    out: str | Path | None = None,
) -> plt.Figure:
    """
    Plot 21-cm power spectrum limits as a function of redshift |z|.

    For each experiment, this plots the lowest (most constraining) limit
    at each redshift as a point or line. The color of the points/lines
    indicates the plotted |k| value by default (see ``color_by``,
    ``show_colorbar`` and ``colormap`` options).

    Parameters
    ----------
    limits : list[str] | None (default: ``None``)
        List of limits to include in the plot
        (see ``KNOWN_LIMITS`` for available limits).
        If not specified, **all** limits are plotted.
    base_limit_style : dict[str, Any] | None (default: ``None``)
        Base style parameters for plotting limits, applied to all limits before any
        individual overrides. For example, ``{'alpha': 0.7}`` to make all limits
        slightly transparent.
    limit_styles : dict[str, dict[str, Any]] | None (default: ``None``)
        Dictionary of style parameters for plotting limits. The keys are the limit
        keys (e.g. ``'Paciga2013'``), and the values are dictionaries with style
        parameters for plotting, e.g. ``{'color': 'C0', 's': 100}`` for points or
        ``{'color': 'C0', 'linewidth': 3}`` for lines. Note that this setting enables
        color customization that may be confusing if a colorbar is also present.
    bold_limits : list[str] | None (default: ``None``)
        List of limits to bold in the legend. If not specified, no limits are bolded.
    shade_limits : bool (default: ``False``)
        Whether to shade the area above each limit line (or points, if plotted as
        points). If ``True``, the area above each limit will be shaded with the color
        specified in the limit styles as ``shade_color`` (default grey) and an alpha
        value specified in the limit styles as ``shade_alpha`` (default 0.5).
    aspoints : list[str] | None (default: ``None``)
        List of limits to plot as points instead of lines.
        If not specified, the function automatically determines whether to plot as
        points or lines based on the number of |z| bins (see ``nz_for_lines``).
    aslines : list[str] | None (default: ``None``)
        List of limits to plot as lines instead of points.
        If not specified, the function automatically determines whether to plot as
        points or lines based on the number of |z| bins (see ``nz_for_lines``).
    nz_for_lines : int (default: ``10``)
        Threshold :math:`n_z` (number of |z| bins) to determine whether to plot a limit
        as points or lines if not specified in ``aspoints`` or ``aslines``.
        If a limit has :math:`len(z) > n_z`, it will be plotted as a line;
        otherwise, it will be plotted as points.
    z_range : tuple[float, float] | None (default: ``None``)
        Tuple specifying the redshift range to include in the plot, in the form
        ``(z_min, z_max)``. If not specified, all redshifts will be included.
    delta_squared_range : tuple[float, float] | None (default: ``None``)
        Tuple specifying the delta squared range to include in the plot, in the form
        ``(delta_squared_min, delta_squared_max)``. If not specified, set to
        ``[1e0, 1e6]`` if theories are plotted, otherwise
        ``[min(limit delta_squared), 1e6]``.
    theories : list[str] | None (default: ``None``)
        List of theories to include in the plot (see ``KNOWN_THEORIES`` for available
        models). If not specified, **no** theories are plotted.
    theory_k : float (default: ``0.25``)
        The |k| value at which to evaluate theories when plotting vs redshift.
        All theories will be evaluated at this k value using spline interpolation
        across their k-space data. Only used if theories are specified.
    base_theory_style : dict[str, Any] | None (default: ``None``)
        Base style parameters for plotting theories, applied to all theories before any
        individual overrides. For example, ``{'alpha': 0.7}`` to make all theories
        slightly transparent.
    theory_styles : dict[str, dict[str, Any]] | None (default: ``None``)
        Dictionary of style parameters for plotting theories. The keys are the theory
        keys (e.g. ``'Mesinger2016Faint'``), and the values are dictionaries with
        style parameters for plotting, e.g. ``{'color': 'C1', 'linestyle': '--'}``.
        Note that this setting enables color customization that may be confusing if
        a colorbar is also present.
    bold_theories : list[str] | None (default: ``None``)
        List of theories to bold in the legend.
        If not specified, no theories are bolded.
    shade_theories : bool (default: ``True``)
        Whether to shade the area below each theory line. If ``True``, the area below
        each theory will be shaded with the color specified in the theory styles as
        ``shade_color`` (default lightsteelblue) and an alpha value specified in the
        theory styles as ``shade_alpha`` (default ``1/len(theories)``).
    sensitivities : dict | None (default: ``None``)
        Dictionary of sensitivities to plot on the figure. The keys are labels for each
        sensitivity estimate, and the values are the file names of the
        sensitivities to plot, which must be outputs from 21cmSense v2+.
        If not specified, no sensitivities are plotted.
    sensitivity_style : dict | None (default: ``None``)
        Dictionary of style parameters for plotting sensitivities. The keys are
        labels for each sensitivity estimate, and the values are dictionaries with
        style parameters for plotting,
        e.g. ``{'color': 'k', 'linestyle': '--', 'linewidth': 3}``.
        An additional key 'sensitivity_kind' can be used to specify which kind of
        sensitivity to plot, e.g. ``'sample+thermal'``, ``'sample'`` or ``'thermal'``.
    color_by : {"year", "k"} (default: ``"k"``)
        Quantity to use for coloring limits. If ``"year"``, each paper is colored by
        experiment year. If ``"k"``, plotted points are colored by their |k| values.
    show_colorbar : bool (default: ``True``)
        Whether to display a colorbar showing the selected ``color_by`` values.
    colormap : str (default: ``'viridis'``)
        Matplotlib colormap to use for coloring limits.
    legend_labeler : dict[str, str] | None
        Optional mapping from limit or theory keys to custom legend labels.
        Keys not present in the mapping will be excluded from the legend.
    k_labels : {"legend", "title"} | None (default: ``None``)
        Where to show the plotted |k| values. If ``"legend"``, add each paper's
        plotted |k| range to its legend label. If ``"title"``, add the plotted |k|
        range across all papers to the title. If ``None``, do not show |k| labels.
    legend_ncols : int | None (default: ``None``)
        Number of columns to use in the legend.
    fontsize : float | None (default: ``None``)
        Font size to use in the legend and axis labels. If not specified, will default
        to 15 point unless ``publication = True``.
    fig_width : float | None (default: ``None``)
        Width of the figure in inches. If not specified, will default to 25 inches
        unless ``publication = True``.
    fig_ratio : float | None (default: ``None``)
        Height to width ratio of the figure. If not specified, ``height = 0.8 * width``
        if theories are plotted, and ``height = 0.6 * width`` if no theories are
        plotted.
    publication : bool | None (default: ``False``)
        Adjust size parameters to make a publication-style plot instead of one
        designed for the screen.
    publication_width : {"single", "double"}  | None (default: ``None``)
        Select a single or double column width to create a publication-style plot
    publication_font : {"times", "dejavu"} | None (default: ``None``)
        Select a publication font style
    fig : matplotlib.figure.Figure | None
        If specified, the figure to plot on. If not specified, a new figure
        will be created.
    ax : matplotlib.axes.Axes| None
        If specified, the axis to plot on. If not specified, a new axis
        will be created.
    out : str | Path | None
        If specified, the file name to save the figure to.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    """
    ###################################################################################
    # Set up the sizing defaults, the figure environment, and the axis environment
    size_defaults = _get_size_defaults(
        publication=publication,
        publication_width=publication_width,
        x="z",
    )

    fig_width = (
        fig_width if fig_width is not None
        else size_defaults["fig_width"]
    )
    fontsize = (
        fontsize if fontsize is not None
        else size_defaults["fontsize"]
    )
    if theories is not None:
        fig_height = fig_width * (fig_ratio or (size_defaults["fig_ratio"] + 0.2))
    else:
        fig_height = fig_width * (fig_ratio or size_defaults["fig_ratio"])

    font, math_fontset = _get_font_properties(
        publication=publication,
        publication_font=publication_font,
        fontsize=fontsize,
    )

    if math_fontset is not None:
        plt.rcParams["mathtext.fontset"] = math_fontset

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    elif ax is not None:
        fig = ax.get_figure()

    ###################################################################################
    # OBSERVATIONAL LIMITS
    # Load data for limits and sort by year.
    if limits is None:
        limits = [load_limit_data(limit).drop_nan() for limit in KNOWN_LIMITS]
        limits.sort(key=lambda limit: limit.year)
    else:
        # DataSet.load() instead of load_limit_data() to allow loading from a YAML file.
        limits = [DataSet.load(limit).drop_nan() for limit in limits]

    # Get the z and delta_squared ranges across all limits.
    z_range = z_range or _get_z_range_from_limits(limits)
    if delta_squared_range is None:
        if theories is not None:
            delta_squared_range = [1e0, 1e6]
        else:
            delta_squared_range = [_get_delta_squared_range_from_limits(limits)[0], 1e6]

    # Select the lowest delta_squared for each z in each limit
    limits = [
        limit.select_lowest_delta_squared(per_z=True, per_tag=False) for limit in limits
    ]

    # Filter by z and delta_squared ranges without applying any k-range selection.
    # This is because we want to plot the lowest delta_squared for each z.
    limits = select_k_and_z_ranges(
        limits,
        z_range=z_range,
        k_range=None,
        delta_squared_range=delta_squared_range,
    )

    # Update z_range after filtering
    z_range = _get_z_range_from_limits(limits)
    k_range = _get_k_range_from_limits(limits)

    # Get k_range's for each limit (exact for labels/coloring)
    delta_k_threshold = 0.05  # Threshold for considering k values as "similar"
    k_ranges_for_lbl = {}
    for limit in limits:
        k_range_for_lbl = np.array([k for k_values in limit.data.k for k in k_values])
        k_ranges_for_lbl[limit.key] = (
            np.min(k_range_for_lbl),
            np.max(k_range_for_lbl),
            np.mean(k_range_for_lbl),
        )

    # Set up colormap for the selected observational quantity.
    if color_by == "year":
        color_values = [limit.year for limit in limits]
        colorbar_label = "Year"
    else:
        color_values = k_range  # Rounded k_range
        colorbar_label = r"k ($h Mpc^{-1}$)"
    norm = colors.Normalize(vmin=min(color_values), vmax=max(color_values))
    scalar_map = cmx.ScalarMappable(norm=norm, cmap=colormap)

    # Building plotting styles for each limit.
    limit_styles = _build_limit_styles(
        limits=limits,
        aspoints=aspoints,
        aslines=aslines,
        nbins_for_lines=nz_for_lines,
        bin_type="z",
        shade_limits=shade_limits,
        base_override=base_limit_style,
        overrides=limit_styles,
        size_defaults=size_defaults,
    )

    # Whether to bold each limit in the legend
    bold_limits = bold_limits or []

    # If z_labels is "legend", we add the z range to the legend labels.
    # The rest of the label is either default or custom from legend_labeler.
    limit_labels = []
    for limit in limits:
        k_label_suffix = ""
        if k_labels == "legend":
            k_min, k_max, k_mean = k_ranges_for_lbl[limit.key]
            if np.abs(k_min - k_max) < delta_k_threshold:
                k_label_suffix = rf"\ (k \approx {k_mean:.2f}\ h/Mpc)"
            else:
                k_label_suffix = rf"\ (k \sim {k_min:.2f}-{k_max:.2f}\ h/Mpc)"
        if legend_labeler is None:
            limit_label = get_latex_label(
                limit,
                bold=(limit.key in bold_limits),
                label_suffix=k_label_suffix,
            )
        else:
            limit_label = legend_labeler.get(limit.key)
            if limit_label is not None and k_label_suffix:
                limit_label = limit_label + rf"${k_label_suffix}$"
        limit_labels.append(limit_label)

    # Plotting the limits as points or lines
    limit_lines = plot_limits_vs_z(
        ax=ax,
        limits=limits,
        limit_styles=limit_styles,
        limit_labels=limit_labels,
        shade_limits=shade_limits,
        delta_squared_range=delta_squared_range,
        scalar_map=scalar_map,
        color_by=color_by,
    )

    ###################################################################################
    # THEORY MODELS

    # Loading data for theories
    theories = theories or []
    theories = [load_theory_model(theory) for theory in theories]

    # For plot_vs_z, we use all available theory redshifts and interpolate at theory_k
    # No need to select specific redshifts since spline will handle interpolation

    # Build styles for theory lines, applying any overrides specified by the user.
    theory_styles = _build_theory_styles(
        theories=theories,
        shade_theories=shade_theories,
        base_override=base_theory_style,
        overrides=theory_styles,
        size_defaults=size_defaults,
    )

    # Whether to bold each theory in the legend
    bold_theories = bold_theories or []
    if legend_labeler is None:
        theory_labels = [
            get_latex_label(theory, bold=(theory.key in bold_theories), theory=True)
            for theory in theories
        ]
    else:
        theory_labels = [legend_labeler.get(theory.key) for theory in theories]

    # Plotting the theory curves vs z
    theory_lines = plot_theories_vs_z(
        ax=ax,
        theories=theories,
        theory_k=theory_k,
        theory_styles=theory_styles,
        theory_labels=theory_labels,
        shade_theories=shade_theories,
        delta_squared_range=delta_squared_range,
    )

    ###################################################################################
    # SENSITIVITIES

    # If sensitivities are specified, build styles and plot them.
    sensitivities = sensitivities or {}

    # Build styles for sensitivity lines, applying any overrides specified by the user.
    sensitivity_style = _build_sensitivity_styles(sensitivities, sensitivity_style)

    # Plot the sensitivity curves.
    plot_sensitivities_vs_k(
        ax=ax,
        sensitivities=sensitivities,
        sensitivity_style=sensitivity_style,
        font=font,
    )

    ###################################################################################
    # PLOT ADJUSTMENTS

    ax.set_xlabel(r"Redshift $z$", fontproperties=font)
    ax.set_ylabel(r"$\Delta^2$ ($mK^2$)", fontproperties=font)
    ax.set_yscale("log")
    ax.set_ylim(*delta_squared_range)
    ax.set_xlim(*z_range)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font)
    if k_labels == "title":
        all_k_values_for_lbl = np.array(list(k_ranges_for_lbl.values())).flatten()
        k_min = min(all_k_values_for_lbl)
        k_max = max(all_k_values_for_lbl)
        k_mean = np.mean(all_k_values_for_lbl)
        if np.abs(k_min - k_max) < delta_k_threshold:
            ax.set_title(rf"$k \approx {k_mean:.2f}\ h/Mpc$", fontproperties=font)
        else:
            ax.set_title(rf"$k \sim {k_min:.2f}-{k_max:.2f}\ h/Mpc$", fontproperties=font)

    # Create colorbar for selected color quantity (if requested)
    if show_colorbar:
        cb = fig.colorbar(
            scalar_map, ax=ax, fraction=0.1, pad=0.08, label=colorbar_label
        )
        cb.ax.yaxis.set_label_position("left")
        cb.ax.yaxis.set_ticks_position("left")
        for label in cb.ax.get_yticklabels():
            label.set_fontproperties(font)
        cb.set_label(label=colorbar_label, fontproperties=font)
    ax.grid(axis="y")

    limit_lines, limit_labels = _filter_legend_entries(limit_lines, limit_labels)
    theory_lines, theory_labels = _filter_legend_entries(theory_lines, theory_labels)
    n_entries = len(limit_labels) + len(theory_labels)

    point_size = 1 / 72.0  # typography standard (points/inch)
    font_inch = fontsize * point_size
    row_height = 2 * font_inch

    # Automatically choose enough columns to keep the legend below ~30%
    # of the figure height, unless the user explicitly specifies ncols.
    if legend_ncols is None:
        max_rows = max(1, int(0.3 * fig_height / row_height))
        legend_ncols = int(np.ceil(n_entries / max_rows))

    leg_rows = int(np.ceil(n_entries / legend_ncols))
    legend_height_norm = leg_rows * row_height / fig_height

    axis_height = 3 * fontsize * point_size
    axis_height_norm = axis_height / fig_height
    plot_bottom = legend_height_norm + axis_height_norm

    ax.legend(
        limit_lines + theory_lines,
        limit_labels + theory_labels,
        bbox_to_anchor=(0.48, legend_height_norm / 2.0),
        loc="center",
        bbox_transform=fig.transFigure,
        ncol=legend_ncols,
        frameon=False,
        prop=font,
    )

    fig.subplots_adjust(bottom=plot_bottom)
    fig.tight_layout()

    if out is not None:
        fig.savefig(out)

    return fig


def plot_vs_k(
    # Limit plotting options
    limits: StrList = None,
    *,
    base_limit_style: JsonDict = None,
    limit_styles: JsonNestedDict = None,
    bold_limits: StrList = None,
    shade_limits: bool = True,
    aspoints: StrList = None,
    aslines: StrList = None,
    nk_for_lines: int = 10,
    # Limits selection options
    z_range: tuple[float, float] | None = None,
    k_range: tuple[float, float] | None = None,
    delta_squared_range: tuple[float, float] | None = None,
    # Theory plotting options
    theories: StrList = None,
    theory_redshifts: dict[str, list[float]] | None = None,
    base_theory_style: JsonDict = None,
    theory_styles: JsonNestedDict = None,
    bold_theories: StrList = None,
    shade_theories: bool = True,
    # Sensitivity plotting options
    sensitivities: dict | None = None,
    sensitivity_style: dict | None = None,
    # General plotting options
    color_by: Literal["z", "year"] = "z",
    show_colorbar: bool = True,
    colormap: str = "Spectral_r",
    legend_labeler: JsonDict = None,
    z_labels: Literal["legend", "title"] | None = "None",
    legend_ncols: int = 3,
    fontsize: float | None = None,
    fig_width: float | None = None,
    fig_ratio: float | None = None,
    publication: bool = False,
    publication_width: Literal["single", "double"] | None = None,
    publication_font: Literal["times", "dejavu"] | None = None,
    # Output options
    fig: Annotated[plt.Figure | None, Parameter(show=False)] = None,
    ax: Annotated[plt.Axes | None, Parameter(show=False)] = None,
    out: str | Path | None = None,
) -> plt.Figure:
    """
    Plot 21-cm power spectrum limits as a function of scale |k|.

    The color of the points/lines indicates redshift by default
    (see ``color_by``, ``show_colorbar`` and ``colormap`` options).

    Parameters
    ----------
    limits : list[str] | None (default: ``None``)
        List of limits to include in the plot
        (see ``KNOWN_LIMITS`` for available limits).
        If not specified, **all** limits are plotted.
    base_limit_style : dict[str, Any] | None (default: ``None``)
        Base style parameters for plotting limits, applied to all limits before any
        individual overrides. For example, ``{'alpha': 0.7}`` to make all limits
        slightly transparent.
    limit_styles : dict[str, dict[str, Any]] | None (default: ``None``)
        Dictionary of style parameters for plotting limits. The keys are the limit
        keys (e.g. ``'Paciga2013'``), and the values are dictionaries with style
        parameters for plotting, e.g. ``{'color': 'C0', 's': 100}`` for points or
        ``{'color': 'C0', 'linewidth': 3}`` for lines. Note that this setting enables
        color customization that may be confusing if a colorbar is also present.
    bold_limits : list[str] | None (default: ``None``)
        List of limits to bold in the legend. If not specified, no limits are bolded.
    shade_limits : bool (default: ``True``)
        Whether to shade the area above each limit line (or points, if plotted as
        points). If ``True``, the area above each limit will be shaded with the color
        specified in the limit styles as ``shade_color`` (default grey) and an alpha
        value specified in the limit styles as ``shade_alpha`` (default 0.5).
    aspoints : list[str] | None (default: ``None``)
        List of limits to plot as points instead of lines.
        If not specified, the function automatically determines whether to plot as
        points or lines based on the number of |k| bins (see ``nk_for_lines``).
    aslines : list[str] | None (default: ``None``)
        List of limits to plot as lines instead of points.
        If not specified, the function automatically determines whether to plot as
        points or lines based on the number of |k| bins (see ``nk_for_lines``).
    nk_for_lines : int (default: ``10``)
        Threshold :math:`n_k` (number of |k| bins) to determine whether to plot a limit
        as points or lines if not specified in ``aspoints`` or ``aslines``.
        If a limit has :math:`len(k) > n_k`, it will be plotted as a line;
        otherwise, it will be plotted as points.
    z_range : tuple[float, float] | None (default: ``None``)
        Tuple specifying the redshift range to include in the plot, in the form
        ``(z_min, z_max)``. If not specified, all redshifts will be included.
    k_range : tuple[float, float] | None (default: ``None``)
        Tuple specifying the |k| range to include in the plot, in the form
        ``(k_min, k_max)``. If not specified, all |k| values will be included.
    delta_squared_range : tuple[float, float] | None (default: ``None``)
        Tuple specifying the delta squared range to include in the plot, in the form
        ``(delta_squared_min, delta_squared_max)``. If not specified, set to
        ``[1e0, 1e6]`` if theories are plotted, otherwise
        ``[min(limit delta_squared), 1e6]``.
    theories : list[str] | None (default: ``None``)
        List of theories to include in the plot (see ``KNOWN_THEORIES`` for available
        models). If not specified, **no** theories are plotted.
    theory_redshifts : dict[str, list[float]] | None (default: ``None``)
        Dictionary specifying which redshifts to plot for each theory. The keys are the
        theory keys (e.g. ``'Mesinger2016Faint'``), and the values are lists of
        redshifts to plot for that theory.
        If not specified, plots the line closest to the center of the ``z_range``.
    base_theory_style : dict[str, Any] | None (default: ``None``)
        Base style parameters for plotting theories, applied to all theories before any
        individual overrides. For example, ``{'alpha': 0.7}`` to make all theories
        slightly transparent.
    theory_styles : dict[str, dict[str, Any]] | None (default: ``None``)
        Dictionary of style parameters for plotting theories. The keys are the theory
        keys (e.g. ``'Mesinger2016Faint'``), and the values are dictionaries with
        style parameters for plotting, e.g. ``{'color': 'C1', 'linestyle': '--'}``.
        Note that this setting enables color customization that may be confusing if
        a colorbar is also present.
    bold_theories : list[str] | None (default: ``None``)
        List of theories to bold in the legend.
        If not specified, no theories are bolded.
    shade_theories : bool (default: ``True``)
        Whether to shade the area below each theory line. If ``True``, the area below
        each theory will be shaded with the color specified in the theory styles as
        ``shade_color`` (default lightsteelblue) and an alpha value specified in the
        theory styles as ``shade_alpha`` (default ``1/len(theories)``).
    sensitivities : dict | None (default: ``None``)
        Dictionary of sensitivities to plot on the figure. The keys are labels for each
        sensitivity estimate, and the values are the file names of the
        sensitivities to plot, which must be outputs from 21cmSense v2+.
        If not specified, no sensitivities are plotted.
    sensitivity_style : dict | None (default: ``None``)
        Dictionary of style parameters for plotting sensitivities. The keys are
        labels for each sensitivity estimate, and the values are dictionaries with
        style parameters for plotting,
        e.g. ``{'color': 'k', 'linestyle': '--', 'linewidth': 3}``.
        An additional key 'sensitivity_kind' can be used to specify which kind of
        sensitivity to plot, e.g. ``'sample+thermal'``, ``'sample'`` or ``'thermal'``.
    color_by : {"z", "year"} (default: ``"z"``)
        Quantity to use for coloring limits. If ``"z"``, limits are colored by
        redshift. If ``"year"``, each paper is colored by experiment year.
    show_colorbar : bool (default: ``True``)
        Whether to display a colorbar showing the selected ``color_by`` values.
    colormap : str (default: ``'Spectral_r'``)
        Matplotlib colormap to use for coloring limits.
    legend_labeler : dict[str, str] | None
        Optional mapping from limit or theory keys to custom legend labels.
        Keys not present in the mapping will be excluded from the legend.
    z_labels : {"legend", "title"} | None (default: ``None``)
        Where to show the plotted redshift values. If ``"legend"``, add each paper's
        plotted redshift range to its legend label. If ``"title"``, add the plotted
        redshift range across all papers to the title. If ``None``, do not show
        redshift labels.
    legend_ncols : int (default: ``3``)
        Number of columns to use in the legend.
    fontsize : float | None (default: ``None``)
        Font size to use in the legend and axis labels. If not specified, will default
        to 15 point unless ``publication = True``.
    fig_width : float | None (default: ``None``)
        Width of the figure in inches. If not specified, will default to 25 inches
        unless ``publication = True``.
    fig_ratio : float | None (default: ``None``)
        Height to width ratio of the figure. If not specified, ``height = 1 * width``
        if theories are plotted, and ``height = 0.5 * width`` if no theories are
        plotted.
    publication : bool | None (default: ``None``)
        Adjust size parameters to make a publication-style plot instead of one
        designed for the screen.
    publication_width : {"single", "double"} | None (default: ``None``)
        Select a single or double column width to create a publication-style plot
    publication_font : {"times", "dejavu"} | None (default: ``None``)
        Select a publication font style
    fig : matplotlib.figure.Figure | None
        If specified, the figure to plot on. If not specified, a new figure
        will be created.
    ax : matplotlib.axes.Axes| None
        If specified, the axis to plot on. If not specified, a new axis
        will be created.
    out : str | Path | None
        If specified, the file name to save the figure to.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    """
    ###################################################################################
    # Set up the sizing defaults, the figure environment, and the axis environment
    size_defaults = _get_size_defaults(
        publication=publication,
        publication_width=publication_width,
        x="k",
    )

    fig_width = (
        fig_width if fig_width is not None
        else size_defaults["fig_width"]
    )
    fontsize = (
        fontsize if fontsize is not None
        else size_defaults["fontsize"]
    )
    if theories is not None:
        fig_height = fig_width * (fig_ratio or (size_defaults["fig_ratio"] + 0.5))
    else:
        fig_height = fig_width * (fig_ratio or size_defaults["fig_ratio"])

    font, math_fontset = _get_font_properties(
        publication=publication,
        publication_font=publication_font,
        fontsize=fontsize,
    )

    if math_fontset is not None:
        plt.rcParams["mathtext.fontset"] = math_fontset

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    elif ax is not None:
        fig = ax.get_figure()


    ###################################################################################
    # OBSERVATIONAL LIMITS

    # Load data for limits and sort by year.
    if limits is None:
        limits = list(KNOWN_LIMITS.keys())
        limits = [load_limit_data(limit).drop_nan() for limit in limits]
        limits.sort(key=lambda limit: limit.year)
    else:
        # DataSet.load() instead of load_limit_data() to allow loading from a YAML file.
        limits = [DataSet.load(limit).drop_nan() for limit in limits]

    # Get the z and k ranges across all limits.
    z_range = z_range or _get_z_range_from_limits(limits)
    k_range = k_range or _get_k_range_from_limits(limits)
    if delta_squared_range is None:
        if theories is not None:
            delta_squared_range = (1e0, 1e6)
        else:
            delta_squared_range = (_get_delta_squared_range_from_limits(limits)[0], 1e6)

    # Filter by z, k, and delta_squared ranges. This will remove any limits that do not
    # have any data points within the specified ranges.
    limits = select_k_and_z_ranges(
        limits,
        z_range=z_range,
        k_range=k_range,
        delta_squared_range=delta_squared_range,
    )

    # Update z_range and k_range after filtering
    z_range = _get_z_range_from_limits(limits)
    k_range = _get_k_range_from_limits(limits)

    # Get z_range's for each limit (exact for labels/coloring)
    delta_z_threshold = 0.5  # Threshold for considering z values as "similar"
    z_ranges_for_lbl = {}
    for limit in limits:
        z_range_for_lbl = limit.data.z
        z_ranges_for_lbl[limit.key] = (
            np.min(z_range_for_lbl),
            np.max(z_range_for_lbl),
            np.mean(z_range_for_lbl),
        )

    # Set up colormap for the selected observational quantity.
    if color_by == "year":
        color_values = [limit.year for limit in limits]
        colorbar_label = "Year"
    else:
        color_values = z_range  # Rounded z_range
        colorbar_label = "Redshift"
    norm = colors.Normalize(vmin=min(color_values), vmax=max(color_values))
    scalar_map = cmx.ScalarMappable(norm=norm, cmap=colormap)

    # Building plotting styles for each limit.
    limit_styles = _build_limit_styles(
        limits=limits,
        aspoints=aspoints,
        aslines=aslines,
        nbins_for_lines=nk_for_lines,
        bin_type="k",
        shade_limits=shade_limits,
        base_override=base_limit_style,
        overrides=limit_styles,
        size_defaults=size_defaults,
    )

    # Whether to bold each limit in the legend
    bold_limits = bold_limits or []

    # If z_labels is "legend", we add the z range to the legend labels.
    # The rest of the label is either default or custom from legend_labeler.
    limit_labels = []
    for limit in limits:
        z_label_suffix = ""
        if z_labels == "legend":
            z_min, z_max, z_mean = z_ranges_for_lbl[limit.key]
            if np.abs(z_min - z_max) < delta_z_threshold:
                z_label_suffix = rf"\ (z \approx {z_mean:.1f})"
            else:
                z_label_suffix = rf"\ (z \sim {z_min:.1f}-{z_max:.1f})"
        if legend_labeler is None:
            limit_label = get_latex_label(
                limit,
                bold=(limit.key in bold_limits),
                label_suffix=z_label_suffix,
            )
        else:
            limit_label = legend_labeler.get(limit.key)
            if limit_label is not None and z_label_suffix:
                limit_label = limit_label + rf"${z_label_suffix}$"
        limit_labels.append(limit_label)

    # Plotting the limits as points or lines, depending on the number of k values
    # or user specifications.
    limit_lines = plot_limits_vs_k(
        ax=ax,
        limits=limits,
        limit_styles=limit_styles,
        limit_labels=limit_labels,
        shade_limits=shade_limits,
        delta_squared_range=delta_squared_range,
        scalar_map=scalar_map,
        color_by=color_by,
    )

    ###################################################################################
    # THEORY MODELS

    # Loading data for theories
    theories = theories or []
    theories = [load_theory_model(theory) for theory in theories]

    # Downselecting to specified redshifts for theories,
    # or closest redshift to centre of redshift range if no redshifts specified.
    theory_redshifts = theory_redshifts or {}
    new_theories = []
    for theory in theories:
        if theory.key not in theory_redshifts:
            theory_redshifts[theory.key] = [0.5 * (z_range[0] + z_range[1])]
        new_theories.extend([
            theory.select_closest_z(z) for z in theory_redshifts[theory.key]
        ])
    theories = new_theories

    # Build styles for theory lines, applying any overrides specified by the user.
    theory_styles = _build_theory_styles(
        theories=theories,
        shade_theories=shade_theories,
        base_override=base_theory_style,
        overrides=theory_styles,
        size_defaults=size_defaults,
    )

    # Whether to bold each theory in the legend
    bold_theories = bold_theories or []
    if legend_labeler is None:
        theory_labels = [
            get_latex_label(theory, bold=(theory.key in bold_theories), theory=True)
            for theory in theories
        ]
    else:
        theory_labels = [legend_labeler.get(theory.key) for theory in theories]

    # Plotting the theory lines.
    theory_lines = plot_theories_vs_k(
        ax=ax,
        theories=theories,
        theory_styles=theory_styles,
        theory_labels=theory_labels,
        shade_theories=shade_theories,
        delta_squared_range=delta_squared_range,
    )

    ###################################################################################
    # SENSITIVITIES

    # If sensitivities are specified, build styles and plot them.
    sensitivities = sensitivities or {}

    # Build styles for sensitivity lines, applying any overrides specified by the user.
    sensitivity_style = _build_sensitivity_styles(sensitivities, sensitivity_style)

    # Plot the sensitivity curves.
    plot_sensitivities_vs_k(
        ax=ax,
        sensitivities=sensitivities,
        sensitivity_style=sensitivity_style,
        font=font,
    )

    ###################################################################################
    # PLOT ADJUSTMENTS

    ax.set_xlabel(r"k ($h Mpc^{-1}$)", fontproperties=font)
    ax.set_ylabel(r"$\Delta^2$ ($mK^2$)", fontproperties=font)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim(*delta_squared_range)
    ax.set_xlim(*k_range)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font)
    if z_labels == "title":
        all_z_values_for_lbl = np.array(list(z_ranges_for_lbl.values())).flatten()
        z_min = min(all_z_values_for_lbl)
        z_max = max(all_z_values_for_lbl)
        z_mean = np.mean(all_z_values_for_lbl)
        if np.abs(z_min - z_max) < delta_z_threshold:
            ax.set_title(rf"$z \approx {z_mean:.2f}$", fontproperties=font)
        else:
            ax.set_title(rf"$z \sim {z_min:.2f}-{z_max:.2f}$", fontproperties=font)

    # Create colorbar for selected color quantity (if requested)
    if show_colorbar:
        cb = fig.colorbar(
            scalar_map, ax=ax, fraction=0.1, pad=0.08, label=colorbar_label
        )
        cb.ax.yaxis.set_label_position("left")
        cb.ax.yaxis.set_ticks_position("left")
        for label in cb.ax.get_yticklabels():
            label.set_fontproperties(font)
        cb.set_label(label=colorbar_label, fontproperties=font)
    ax.grid(axis="y")

    limit_lines, limit_labels = _filter_legend_entries(limit_lines, limit_labels)
    theory_lines, theory_labels = _filter_legend_entries(theory_lines, theory_labels)
    leg_rows = int(np.ceil((len(limit_labels) + len(theory_labels)) / legend_ncols))

    point_size = 1 / 72.0  # typography standard (points/inch)
    font_inch = fontsize * point_size
    legend_height = (2 * leg_rows) * font_inch

    legend_height_norm = legend_height / fig_height  # 0.25

    axis_height = 3 * fontsize * point_size
    axis_height_norm = axis_height / fig_height
    plot_bottom = legend_height_norm + axis_height_norm

    ax.legend(
        limit_lines + theory_lines,
        limit_labels + theory_labels,
        bbox_to_anchor=(0.48, legend_height_norm / 2.0),
        loc="center",
        bbox_transform=fig.transFigure,
        ncol=legend_ncols,
        frameon=False,
        prop=font,
    )

    fig.subplots_adjust(bottom=plot_bottom)
    fig.tight_layout()

    if out is not None:
        fig.savefig(out)

    return fig


def select_k_and_z_ranges(
    limits: list[DataSet],
    *,
    z_range: tuple[float, float] | None,
    k_range: tuple[float, float] | None,
    delta_squared_range: tuple[float, float] | None,
) -> list[DataSet]:
    """Select the specified k and redshift ranges from the limits.

    Parameters
    ----------
    limits : list[DataSet]
        The list of power spectrum upper limit datasets to filter.
    z_range : tuple[float, float] | None
        The redshift range to select (min, max). If None, do not down-select on
        redshift.
    k_range : tuple[float, float] | None
        The k range to select in 1/Mpc units (min, max). If None, do not down-select
        on wavenumber.
    delta_squared_range : tuple[float, float] | None
        The delta squared range to select. If None, do not down-select on delta squared.

    Returns
    -------
    list[DataSet]
        The filtered list of limit datasets.
    """
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


def _get_z_range_from_limits(limits, round_nums: bool = True) -> tuple[float, float]:
    """Get min/max z range across a list of datasets."""
    z_min = min(min(limit.data.z) for limit in limits)
    z_max = max(max(limit.data.z) for limit in limits)
    if round_nums:
        z_min = np.floor(z_min)
        z_max = np.ceil(z_max)
    return (z_min, z_max)


def _get_k_range_from_limits(limits, round_nums: bool = True) -> tuple[float, float]:
    """Get min/max k range across a list of datasets."""
    k_min = min(min(k) for limit in limits for k in limit.data.k)
    k_max = max(max(k) for limit in limits for k in limit.data.k)
    if round_nums:
        min_factor = 10 ** np.ceil(np.log10(k_min) * -1)
        max_factor = 10 ** np.ceil(np.log10(k_max) * -1)
        k_min = np.floor(k_min * min_factor) / min_factor
        k_max = np.ceil(k_max * max_factor) / max_factor
    return (k_min, k_max)


def _get_delta_squared_range_from_limits(
    limits: list[DataSet], round_nums: bool = True
) -> tuple[float, float]:
    """Get rounded min/max delta-squared range across a list of datasets."""
    delta_squared_min = min(
        min(dsq) for limit in limits for dsq in limit.data.delta_squared
    )
    delta_squared_max = max(
        max(dsq) for limit in limits for dsq in limit.data.delta_squared
    )
    if round_nums:
        min_factor = 10 ** np.ceil(np.log10(delta_squared_min) * -1)
        max_factor = 10 ** np.ceil(np.log10(delta_squared_max) * -1)
        delta_squared_min = np.floor(delta_squared_min * min_factor) / min_factor
        delta_squared_max = np.ceil(delta_squared_max * max_factor) / max_factor
    return (delta_squared_min, delta_squared_max)


def _build_limit_styles(
    *,
    limits: list[DataSet],
    aspoints: list[str] | None,
    aslines: list[str] | None,
    nbins_for_lines: int,
    bin_type: Literal["k", "z"],
    shade_limits: bool,
    base_override: dict[str, Any] | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
    size_defaults: dict[str, float],
) -> dict[str, dict[str, Any]]:
    """Build a dictionary of styles to use for each limit paper.

    Note that this is an internal function not meant to be used by users.

    Parameters
    ----------
    limits : list[DataSet]
        The list of limit datasets for which to build styles.
    aspoints : list[str] | None
        The list of limit keys to plot as points.
    aslines : list[str] | None
        The list of limit keys to plot as lines.
    nbins_for_lines : int
        The number of bins above which to automatically plot as lines.
    bin_type : {"k", "z"}
        The bin axis to use for automatic line selection.
    shade_limits : bool
        Whether to shade the limits.
    base_override : dict[str, Any] | None
        A dictionary of base style overrides.
    overrides : dict[str, dict[str, Any]] | None
        A dictionary of style overrides for each limit.
    size_defaults: dict[str, float]
        A dictionary of size defaults

    Returns
    -------
    dict[str, dict[str, Any]]
        A dictionary mapping limit keys to their styles.
    """
    aspoints = aspoints or []
    aslines = aslines or []
    styles = {}

    for limit in limits:
        # Empty
        style = {}
        nbins = (
            max(len(k) for k in limit.data.k) if bin_type == "k" else len(limit.data.z)
        )
        # Determine whether to plot as points or lines
        if limit.key in aspoints and limit.key in aslines:
            raise ValueError(
                f"Limit {limit.key} specified in both aspoints and aslines."
            )
        style["as_line"] = (
            False
            if limit.key in aspoints
            else True
            if limit.key in aslines
            else nbins > nbins_for_lines
        )
        # Set defaults for points
        if not style["as_line"]:
            style.setdefault("s", size_defaults["marker_size"])
            style.setdefault(
                "marker", DEFAULT_TELESCOPE_MARKERS.get(limit.telescope, "o")
            )
        # Set defaults for lines
        else:
            style.setdefault("linewidth", size_defaults["linewidth"])
            style.setdefault("linestyle", "-")
        # If we are shading the limits
        if shade_limits:
            style.setdefault("shade_alpha", 0.5)
            style.setdefault("shade_color", "grey")
        # Apply user overrides
        style |= base_override if base_override else {}
        style |= overrides.get(f"{limit.key}", {}) if overrides else {}
        # Add to styles dictionary
        styles[f"{limit.key}"] = style

    return styles


def _build_theory_styles(
    *,
    theories: list[DataSet],
    shade_theories: bool,
    base_override: dict[str, Any] | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
    size_defaults: dict[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build a dictionary of styles to use for each theory paper.

    Note that this is an internal function not meant to be used by users.

    Parameters
    ----------
    theories : list[DataSet]
        The list of theory datasets for which to build styles.
    shade_theories : bool
        Whether to shade the theories.
    base_override : dict[str, Any] | None
        A dictionary of base style overrides.
    overrides : dict[str, dict[str, Any]] | None
        A dictionary of style overrides for each theory.
    size_defaults: dict[str, float]
        A dictionary of size defaults

    Returns
    -------
    dict[str, dict[str, Any]]
        A dictionary mapping theory keys to their styles.
    """
    styles = {}
    for theory in theories:
        # Empty
        style = {}
        # Set default style parameters for theory lines.
        style.setdefault("linestyle", "--")
        style.setdefault("linewidth", size_defaults["linewidth"])
        style.setdefault("color", "lightsteelblue")
        # If we are shading the limits
        if shade_theories:
            style.setdefault("shade_alpha", 1 / len(theories))
            style.setdefault("shade_color", "lightsteelblue")
        # Set base and overrides
        style |= base_override if base_override else {}
        style |= overrides.get(f"{theory.key}", {}) if overrides else {}
        # Add to styles dictionary
        styles[f"{theory.key}"] = style
    return styles


def _build_sensitivity_styles(
    sensitivities: dict[str, str],
    sensitivity_style: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Build a dictionary of styles to use for each sensitivity curve.

    Note that this is an internal function not meant to be used by users.

    Parameters
    ----------
    sensitivities : dict[str, str]
        A dictionary mapping sensitivity names to their descriptions.
    sensitivity_style : dict[str, dict[str, Any]] | None
        A dictionary of style overrides for each sensitivity curve.

    Returns
    -------
    dict[str, dict[str, Any]]
        A dictionary mapping sensitivity names to their styles.
    """
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


def _filter_legend_entries(
    handles: Sequence[Any], labels: Sequence[str | None]
) -> tuple[list[Any], list[str]]:
    """Remove legend entries with no label while keeping handles aligned.

    Parameters
    ----------
    handles
        Matplotlib artists to include in the legend.
    labels
        Legend labels corresponding to ``handles``. Entries with ``None`` labels are
        omitted from the returned handles and labels.

    Returns
    -------
    list
        The legend handles whose corresponding label is not ``None``.
    list
        The non-``None`` legend labels.
    """
    entries = [
        (handle, label)
        for handle, label in zip(handles, labels, strict=True)
        if label is not None
    ]
    if not entries:
        return [], []

    filtered_handles, filtered_labels = zip(*entries, strict=True)
    return list(filtered_handles), list(filtered_labels)


def _get_size_defaults(
    publication: bool,
    publication_width: Literal["single", "double"] | None,
    x: Literal["z","k"] | None,
) -> dict[str, float]:
    """Return sizing defaults appropriate for screen or publication.
    
    Parameters
    ----------
    publication : bool
        Whether to default sizes to journal publication styles
    publication_width : Literal[str, str] | None
        Specify single or double column environment in publication style
    x : Literal[str, str] | None
        Specify the x-axis to adjust sizing

    Returns
    -------
    dict[str, float]
        A dictionary of sizing defaults for screen or publication
    """
    if x is None or x.lower() == "k":
        fig_ratio = 0.5
    elif x.lower() == "z":
        fig_ratio = 0.6
    else:
        raise ValueError("x must be 'k', 'z', or None.")

    if publication:
        # Single column is ~3.4 inches, double column is ~7 inches
        if publication_width is None or publication_width.lower() == "single":
            return {
                "fig_width": 3.4,
                "fig_ratio": fig_ratio,
                "fontsize": 8.0,
                "marker_size": 25.0,
                "linewidth": 1.0,
            }
        elif publication_width.lower() == "double":
            return {
                "fig_width": 7.0,
                "fig_ratio": fig_ratio,
                "fontsize": 9.0,
                "marker_size": 35.0,
                "linewidth": 1.25,
            }
        else:
            raise ValueError(
                "publication_width must be either 'single', 'double', or None."
            )

    return {
        "fig_width": 25.0,
        "fig_ratio": fig_ratio,
        "fontsize": 15.0,
        "marker_size": 150.0,
        "linewidth": 2.0,
    }

def _get_font_properties(
    publication: bool,
    publication_font: Literal["times", "dejavu"] | None,
    fontsize: float,
):
    """Returns font properties  and math fontset for a specified font.
        
    Parameters
    ----------
    publication : bool
        Whether to default font style to Times
    publication_font : Literal[str, str] | None
        Specify a type of font
    fontsize : float
        Specify a size of font

    Returns
    -------
    matplotlib.font_manager.FontProperties
        Font properties for plot text.
    str | None
        Matplotlib math fontset to use for mathematical text.
    """

    # Use Times by default for publication plots.
    if publication_font is None and publication:
        publication_font = "times"

    if publication_font is None:
        font = fm.FontProperties(size=fontsize)
        math_fontset = None
    elif publication_font.lower() == "times":
        font = fm.FontProperties(
            family="Times",
            style="normal",
            size=fontsize,
            weight="normal",
        )
        math_fontset = "stix"
    elif publication_font.lower() == "dejavu":
        font = fm.FontProperties(
            family="DejaVu Sans",
            style="normal",
            size=fontsize,
            weight="normal",
        )
        math_fontset = "dejavusans"
    else:
        raise ValueError(
            "publication_font must be 'times', 'dejavu', or None."
        )

    return font, math_fontset


def get_latex_label(
    paper: DataSet,
    bold: bool = False,
    theory: bool = False,
    label_suffix: str = "",
) -> str:
    """Get a LaTeX label for a limit or theory paper.

    Parameters
    ----------
    paper : DataSet
        The limit paper for which to generate a label.
    bold : bool, optional
        Whether to make the label bold, by default False.
    theory : bool, optional
        Whether this is a theory (True) or limit (False) paper.
    label_suffix : str, optional
        Optional LaTeX suffix.

    Returns
    -------
    str
        The LaTeX label for the limit paper.
    """
    if theory:
        label_start = " $\\bf{Theory:} \\bf{" if bold else " $\\bf{Theory:} \\rm{"
    else:
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
        + label_suffix
        + label_end
    )


def plot_limits_vs_k(
    *,
    ax: plt.Axes,
    limits: list[DataSet],
    limit_styles: dict[str, dict[str, Any]],
    limit_labels: list[str],
    shade_limits: bool,
    delta_squared_range: tuple[float, float],
    scalar_map: cmx.ScalarMappable,
    color_by: Literal["z", "year"],
):
    """Plot limit papers on k vs delta_squared axes.

    Parameters
    ----------
    ax : plt.Axes
        The axes on which to plot the limits.
    limits : list[DataSet]
        A list of limit papers to plot.
    limit_styles : dict[str, dict[str, Any]]
        A dictionary mapping limit paper keys to their styles.
    limit_labels : list[str]
        A list of labels for the limit papers.
    shade_limits : bool
        Whether to shade the limit regions.
    delta_squared_range : tuple[float, float]
        The range of delta squared values to display.
    scalar_map : cmx.ScalarMappable
        A scalar mappable for coloring the points.
    color_by : {"z", "year"}
        Quantity to use for coloring limits.
    """
    lines = []

    for limit, label in zip(limits, limit_labels, strict=True):
        logging.getLogger("eor_limits").info(f"Plotting {limit.author} {limit.year}")

        limit_style = limit_styles[limit.key].copy()

        # Pop invalid args for plt.plot or plt.scatter
        as_line = limit_style.pop("as_line")
        if shade_limits:
            shade_alpha = limit_style.pop("shade_alpha")
            shade_color = limit_style.pop("shade_color")
        # (especially to avoid errors when you have base styling)
        if as_line:
            for key in ["s"]:
                limit_style.pop(key, None)
        else:
            for key in ["linewidth", "lw", "linestyle", "ls"]:
                limit_style.pop(key, None)

        # Use user-provided color if available, otherwise use scalar_map.
        has_color_override = "color" in limit_style
        if has_color_override:
            color_val = limit_style.pop("color")
        elif color_by == "year":
            color_val = scalar_map.to_rgba(limit.year)

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

            line = ax.scatter(
                k,
                dsq,
                color=color_val if color_by == "year" else scalar_map.to_rgba(z),
                label=label,
                zorder=2,
                **limit_style,
            )

            if (
                shade_limits
                and limit.data.k_lower is not None
                and limit.data.k_upper is not None
            ):
                for klow, khi, dsq in zip(
                    limit.data.k_lower,
                    limit.data.k_upper,
                    limit.data.delta_squared,
                    strict=True,
                ):
                    k_edges = np.concatenate((klow, [khi[-1]]))
                    delta_edges = np.concatenate((dsq, dsq[-1:]))
                    ax.fill_between(
                        k_edges,
                        delta_edges,
                        delta_squared_range[1],
                        color=shade_color,
                        alpha=shade_alpha,
                        zorder=0,
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

                # make black outline by plotting thicker black line first
                ax.plot(
                    k_edges,
                    delta_edges,
                    color="black",
                    linewidth=limit_style["linewidth"] + 2,
                    zorder=1,
                )

                (this_line,) = ax.plot(
                    k_edges,
                    delta_edges,
                    label=label,
                    zorder=1,
                    color=(
                        color_val
                        if color_by == "year"
                        else scalar_map.to_rgba(redshift)
                    ),
                    **limit_style,
                )
                if shade_limits:
                    ax.fill_between(
                        k_edges,
                        delta_edges,
                        delta_squared_range[1],
                        color=shade_color,
                        alpha=shade_alpha,
                        zorder=0,
                    )

                if ind == 0:
                    line = this_line

        lines.append(line)

    return lines


def plot_theories_vs_k(
    *,
    ax: plt.Axes,
    theories: list[DataSet],
    theory_styles: dict[str, dict[str, Any]],
    theory_labels: list[str],
    shade_theories: bool,
    delta_squared_range: tuple[float, float],
):
    """Plot theory lines on k vs delta_squared axes.

    Parameters
    ----------
    ax : plt.Axes
        The axes on which to plot the theories.
    theories : list[DataSet]
        A list of theory papers to plot.
    theory_styles : dict[str, dict[str, Any]]
        A dictionary mapping theory paper keys to their styles.
    theory_labels : list[str]
        A list of labels for the theory papers.
    shade_theories : bool
        Whether to shade the theory regions.
    delta_squared_range : tuple[float, float]
        The range of delta squared values to display.
    """
    lines = []

    for theory, label in zip(theories, theory_labels, strict=True):
        logging.getLogger("eor_limits").info(f"Plotting {theory.author} {theory.year}")

        theory_style = theory_styles[theory.key].copy()

        # Shade first and pop the specific args
        if shade_theories:
            shade_alpha = theory_style.pop("shade_alpha")
            shade_color = theory_style.pop("shade_color")
            ax.fill_between(
                theory.data.k[0],
                theory.data.delta_squared[0],
                delta_squared_range[0],
                color=shade_color,
                alpha=shade_alpha,
                zorder=0,
            )

        # Plot the theory line on top
        (line,) = ax.plot(
            theory.data.k[0],
            theory.data.delta_squared[0],
            label=label,
            zorder=1,
            **theory_style,
        )

        lines.append(line)

    return lines


def plot_sensitivities_vs_k(
    *,
    ax: plt.Axes,
    sensitivities: dict[str, str] | None,
    sensitivity_style: dict[str, dict[str, Any]] | None,
    font: fm.FontProperties,
):
    """Plot the sensitivity curves on the given axes.

    Parameters
    ----------
    ax : plt.Axes
        The axes on which to plot the sensitivities.
    sensitivities : dict[str, str] | None
        A dictionary mapping sensitivity names to their file paths.
    sensitivity_style : dict[str, dict[str, Any]] | None
        A dictionary mapping sensitivity names to their styles.
    font : matplotlib.font_manager.FontProperties
        Font properties for the instrument names.
    """
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

        ax.plot(
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
        ax.text(
            ks[k_ind], sense[k_ind], name, fontproperties=font, verticalalignment="top"
        )


def plot_limits_vs_z(
    *,
    ax: plt.Axes,
    limits: list[DataSet],
    limit_styles: dict[str, dict[str, Any]],
    limit_labels: list[str | None],
    shade_limits: bool,
    delta_squared_range: tuple[float, float],
    scalar_map: cmx.ScalarMappable,
    color_by: Literal["year", "k"],
):
    """Plot limit papers on z vs delta_squared axes.

    Parameters
    ----------
    ax : plt.Axes
        The axes on which to plot the limits.
    limits : list[DataSet]
        A list of limit papers to plot, each reduced to one point per redshift.
    limit_styles : dict[str, dict[str, Any]]
        A dictionary mapping limit paper keys to their styles.
    limit_labels : list[str | None]
        A list of labels for the limit papers.
    shade_limits : bool
        Whether to shade the limit regions.
    delta_squared_range : tuple[float, float]
        The range of delta squared values to display.
    scalar_map : cmx.ScalarMappable
        A scalar mappable for coloring the limits.
    color_by : {"year", "k"}
        Quantity to use for coloring limits.
    """
    lines = []

    for _idx, (limit, label) in enumerate(zip(limits, limit_labels, strict=True)):
        logging.getLogger("eor_limits").info(f"Plotting {limit.author} {limit.year}")

        limit_style = limit_styles[limit.key].copy()

        # Pop invalid args for plt.plot or plt.scatter
        as_line = limit_style.pop("as_line")
        if shade_limits:
            shade_alpha = limit_style.pop("shade_alpha")
            shade_color = limit_style.pop("shade_color")
        # (especially to avoid errors when you have base styling)
        if as_line:
            for key in ["s"]:
                limit_style.pop(key, None)
        else:
            for key in ["linewidth", "lw", "linestyle", "ls"]:
                limit_style.pop(key, None)

        # Each limit now has one data point per redshift
        z_vals = limit.data.z
        dsq_vals = np.array([dsq[0] for dsq in limit.data.delta_squared])
        k_vals = np.array([k[0] for k in limit.data.k])

        # Use user-provided color if available, otherwise use scalar_map.
        has_color_override = "color" in limit_style
        if has_color_override:
            color_val = limit_style.pop("color")
        elif color_by == "k":
            color_val = scalar_map.to_rgba(k_vals)
        else:
            color_val = scalar_map.to_rgba(limit.year)

        if not as_line:
            # Plot as scatter points
            line = ax.scatter(
                z_vals,
                dsq_vals,
                color=color_val,
                label=label,
                zorder=2,
                **limit_style,
            )

            if shade_limits:
                # Shade the region above the limits
                ax.fill_between(
                    z_vals,
                    dsq_vals,
                    delta_squared_range[1],
                    color=shade_color,
                    alpha=shade_alpha,
                    zorder=0,
                )
        else:
            # Plot as connected line
            # Make black outline by plotting thicker black line first
            ax.plot(
                z_vals,
                dsq_vals,
                color="black",
                linewidth=limit_style.get("linewidth", 2) + 2,
                zorder=1,
            )

            (line,) = ax.plot(
                z_vals,
                dsq_vals,
                color=(
                    np.mean(color_val, axis=0)
                    if color_by == "k" and not has_color_override
                    else color_val
                ),
                label=label,
                zorder=1,
                **limit_style,
            )

            if shade_limits:
                ax.fill_between(
                    z_vals,
                    dsq_vals,
                    delta_squared_range[1],
                    color=shade_color,
                    alpha=shade_alpha,
                    zorder=0,
                )

        lines.append(line)

    return lines


def plot_theories_vs_z(
    *,
    ax: plt.Axes,
    theories: list[DataSet],
    theory_k: float,
    theory_styles: dict[str, dict[str, Any]],
    theory_labels: list[str],
    shade_theories: bool,
    delta_squared_range: tuple[float, float],
):
    """Plot theory lines on z vs delta_squared axes.

    Parameters
    ----------
    ax : plt.Axes
        The axes on which to plot the theories.
    theories : list[DataSet]
        A list of theory papers to plot.
    theory_k : float
        The |k| value at which to evaluate theories.
    theory_styles : dict[str, dict[str, Any]]
        A dictionary mapping theory paper keys to their styles.
    theory_labels : list[str]
        A list of labels for the theory papers.
    shade_theories : bool
        Whether to shade the theory regions.
    delta_squared_range : tuple[float, float]
        The range of delta squared values to display.
    """
    lines = []

    # If no theories provided, return empty list
    if not theories:
        return lines

    for theory, label in zip(theories, theory_labels, strict=True):
        logging.getLogger("eor_limits").info(f"Plotting {theory.author} {theory.year}")

        theory_style = theory_styles[theory.key].copy()

        # For theories on z plot, interpolate delta_squared at the specified k value
        # for each redshift
        z_vals = theory.data.z
        dsq_vals = []

        for z_idx, k_vals in enumerate(theory.data.k):
            dsq_arr = theory.data.delta_squared[z_idx]

            # Create a spline for this redshift's k-space data
            # Sort by k to ensure monotonic x values for interpolation
            sort_idx = np.argsort(k_vals)
            k_sorted = k_vals[sort_idx]
            dsq_sorted = dsq_arr[sort_idx]

            # Check if theory_k is within the available k range
            k_min, k_max = k_sorted[0], k_sorted[-1]
            if not (k_min <= theory_k <= k_max):
                logging.getLogger("eor_limits").warning(
                    f"theory_k={theory_k} outside k range [{k_min:.4f}, "
                    f"{k_max:.4f}] for {theory.key} at z={z_vals[z_idx]}. "
                    f"Using nearest available k value."
                )
                # Use nearest k value
                nearest_idx = np.argmin(np.abs(k_sorted - theory_k))
                dsq_vals.append(dsq_sorted[nearest_idx])
            else:
                # Check for non-finite values (NaN or inf)
                mask = np.isfinite(dsq_sorted)
                if not np.any(mask):
                    raise ValueError(
                        f"All delta_squared values are non-finite for {theory.key} "
                        f"at z={z_vals[z_idx]}."
                    )

                try:
                    spl = interpolate.CubicSpline(k_sorted[mask], dsq_sorted[mask])
                    dsq_at_k = spl(theory_k)
                    dsq_vals.append(dsq_at_k)
                except ValueError as e:
                    msg = f"CubicSpline failed for {theory.key}: {e}"
                    logging.getLogger("eor_limits").warning(
                        f"{msg}. Using nearest neighbor interpolation."
                    )
                    nearest_idx = np.argmin(np.abs(k_sorted - theory_k))
                    dsq_vals.append(dsq_sorted[nearest_idx])

        dsq_vals = np.array(dsq_vals)

        # Shade first and pop the specific args
        if shade_theories:
            shade_alpha = theory_style.pop("shade_alpha")
            shade_color = theory_style.pop("shade_color")
            ax.fill_between(
                z_vals,
                dsq_vals,
                delta_squared_range[0],
                color=shade_color,
                alpha=shade_alpha,
                zorder=0,
            )

        # Plot the theory line on top
        (line,) = ax.plot(
            z_vals,
            dsq_vals,
            label=label,
            zorder=1,
            **theory_style,
        )

        if label is not None:
            lines.append(line)

    return lines
