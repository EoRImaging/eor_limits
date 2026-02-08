#! /usr/bin/env python
# Copyright (c) 2019 Nichole Barry, Bryna Hazelton
# Licensed under the 2-clause BSD License
"""Code for plotting EoR Limits."""

from pathlib import Path

from cyclopts import App

from .plots.plot_vs_k_z import make_plot

app = App()


@app.command
def plot(
    papers=None,
    theory_legend: bool = True,
    theories: list[str] = None,
    theory_model: str = None,
    theory_nf: list[float] = None,
    theory_redshift: list[float] = None,
    theory_linewidth: list[float] = None,
    dsq_range: tuple[float, float] = None,
    z_range: tuple[float, float] = None,
    k_range: tuple[float, float] = None,
    shade_limits: float = 0.5,
    shade_theory: float | None = None,
    colormap: str = "Spectral_r",
    bold_papers: list[str] = None,
    fontsize: int = 15,
    sensitivities: list[tuple[str, str]] = None,
    sensitivity_style: list[tuple[str, str]] = None,
    height_ratio: float = None,
    nk_for_lines: int = 10,
    aspoints: list[str] = None,
    aslines: list[str] = None,
    out: Path = Path("eor_limits.pdf"),
):
    """
    Plot the current EoR Limits as a function of k or redshift.

    Parameters
    ----------
    papers
        Papers to include on plot (must be in data directory). Defaults to all papers
        in the data directory.
    theory_legend
        Whether to exclude theory lines from the legend. Used by some users who
        prefer to add the annotations on the lines by hand to improve readability.
    theories
        Theories to plot. Theory-specific options can be set to control which lines are
        drawn.
    theory_model
        Model type to select from theories (e.g. 'bright' or 'faint' for Mesinger et
        al. 2016).
    theory_nf
        Neutral fractions to select from theories.
    theory_redshift
        Redshifts to select from theories.
    theory_linewidth
        Linewidths for theory lines.
    dsq_range
        Range of Delta Squared to include on plot (yaxis range).
    z_range
        Range of redshifts to include on plot.
    k_range
        Range of k values to include on plot (xaxis range).
    shade_limits
        Type of shading above limits to apply.
    shade_theory
        An alpha value to use for shading below theory lines. By default, choose
        a shading such that when all theories overlap, it becomes opaque.
    colormap
        Matplotlib colormap to use.
    bold_papers
        List of papers to bold in caption.
    fontsize
        Font size to use in the legend.
    out
        Output file name.
    sensitivities
        List of tuples of (name, file) for which to include sensitivities.
        Files must be 21cmSense v2+ outputs, which can be generated with
        ``sense calc-sense --fname out.h5`` (see 21cmSense docs for more info).
    sensitivity_style
        List of style parameters for plotting sensitivities. Each entry
        should be a name and a dictionary. Each 'name' (if given) should correspond
        to a label in the sensitivities argument, and the values should be a JSON
        string with style parameters for plotting, e.g. {'color': 'k', 'ls': '--',
        'lw': 3}. An additional key 'sensitivity_kind' can be used to specify which
        kind of sensitivity to plot, e.g. 'sample+thermal', 'sample' or 'thermal'.
        If no name is given, the style will be applied to all sensitivities. If no
        style is given, the default style will be used.
    height_ratio
        Height to width ratio of the figure.
    nk_for_lines
        Number of k values required to plot a line instead of markers for a given
        limit.
    aspoints
        List of papers to plot as points instead of lines. By default, all limits with
        more than nk_for_lines k values will be plotted as lines, and the rest as
        points.
    aslines
        List of papers to plot as lines instead of points. By default, all limits with
        more than nk_for_lines k values will be plotted as lines, and the rest as
        points.

    """
    if theories is not None:
        theories, theory_model, theory_nf, theory_redshift = parse_theories(
            theories, theory_model, theory_nf, theory_redshift, theory_linewidth
        )

    # Process sensitivity arguments.
    if sensitivities:
        sensitivities = dict(sensitivities)

    if sensitivity_style:
        sensitivity_style = {k: json.loads(v) for k, v in sensitivity_style}

    fig = make_plot(
        papers=papers,
        theories=theories,
        theory_legend=theory_legend,
        delta_squared_range=dsq_range,
        redshift_range=z_range,
        k_range=k_range,
        shade_limits=shade_limits,
        shade_theory_alpha=shade_theory,
        colormap=colormap,
        bold_papers=bold_papers,
        fontsize=fontsize,
        sensitivities=sensitivities,
        sensitivity_style=sensitivity_style,
        fig_ratio=height_ratio,
        nk_for_lines=nk_for_lines,
        aspoints=aspoints,
        aslines=aslines,
    )
    fig.savefig(out)
