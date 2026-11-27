"""Test the plotting functions."""

import logging
from pathlib import Path

import numpy as np
import pytest

from ps21cm_limits import load_sensitivity_data, plot_vs_k, plot_vs_z

PLOT_FUNCS = [plot_vs_k, plot_vs_z]

# The limits and theories that every plot below starts from
PLOTTING_ARGS = {
    "limits": [
        # HERA
        "HERA2022",
        "HERA2023",
        "HERA2026",
        # MWA
        "Dillon2014",
        "Dillon2015",
        "Beardsley2016",
        "Ewall-Wice2016",
        "Barry2019",
        "Li2019",
        "Trott2020",
        "Rahimi2021",
        "Yoshiura2021",
        "Wilensky2023",
        "Nunhokee2025",
        "Trott2025",
    ],
    "theories": [
        "Mesinger2016Faint",
        "Mesinger2016Bright",
        "Munoz2018FDM3",
        "Munoz2022AllGalaxies",
        "Munoz2022Optimistic",
    ],
}

# Options accepted by both plotting functions, each drawn on top of PLOTTING_ARGS
OPTIONS = {
    "fig_styling": {
        "fontsize": 12,
        "colormap": "viridis",
        "fig_width": 20.0,
        "fig_ratio": 2.0,
        "legend_ncols": 2,
    },
    "publication": {"publication": True, "dpi": 300},
    "z_range": {"z_range": (6.0, 10.0)},
    "delta_squared_range": {"delta_squared_range": (1e3, 1e5)},
    "aspoints": {"aspoints": PLOTTING_ARGS["limits"]},
    "aslines": {"aslines": PLOTTING_ARGS["limits"]},
    "bold_limits": {"bold_limits": ["HERA2022", "HERA2023"]},
    "bold_theories": {"bold_theories": ["Mesinger2016Faint"]},
    "limit_and_theory_styling": {
        "base_limit_style": {"linewidth": 3, "alpha": 0.8},
        "limit_styles": {"HERA2023": {"color": "C3"}},
        "base_theory_style": {"linestyle": "-."},
        "theory_styles": {
            "Mesinger2016Faint": {"color": "C0", "shade_alpha": 0.5},
            "Mesinger2016Bright": {"color": "C1", "shade_alpha": 0.1},
        },
    },
}

# Options accepted only by plot_vs_k, drawn on top of PLOTTING_ARGS
K_OPTIONS = {
    "k_range": {"k_range": (0.1, 1.0)},
    "limit_shading": {
        "base_limit_style": {"shade_alpha": 0.1},
        "limit_styles": {"HERA2023": {"shade_alpha": 0.25, "shade_color": "C3"}},
    },
}

# SKA-Low sensitivities distributed with the package, keyed by plot label, and the
# redshift range they are plotted over (holding their z = 7.9, 9.1, 10.4 and 10.8)
SKA_SENSITIVITIES = {
    "SKA-AA4": "Breitman2026_AA4_deep1080h",
    "SKA-AA*": "Breitman2026_AAstar_deep1080h",
}
SENSITIVITY_Z_RANGE = (7.0, 11.0)


def _plot(plot_func, tmp_path: Path, **kwargs):
    """Draw ``plot_func`` with PLOTTING_ARGS and ``kwargs``, saved to ``tmp_path``."""
    return plot_func(**{**PLOTTING_ARGS, **kwargs}, out=tmp_path / "plot.png")


def _legend_texts(fig) -> list[str]:
    """Return the texts of the legend entries of ``fig``."""
    return [text.get_text() for text in fig.axes[0].get_legend().get_texts()]


def _find_lines(ax, x, y) -> list:
    """Return the lines on ``ax`` whose x and y data equal ``x`` and ``y``."""
    return [
        line
        for line in ax.get_lines()
        if len(line.get_xdata()) == len(x)
        and np.allclose(line.get_xdata(), x)
        and np.allclose(line.get_ydata(), y)
    ]


@pytest.mark.parametrize("plot_func", PLOT_FUNCS)
def test_lib_plot_defaults(plot_func, tmp_path):
    """Test plotting with no arguments, which draws every known limit."""
    plot_func(out=tmp_path / "plot.png")


@pytest.mark.parametrize("plot_func", PLOT_FUNCS)
@pytest.mark.parametrize("option", OPTIONS)
def test_lib_plot_with_option(plot_func, option, tmp_path):
    """Test that each plotting option draws a plot without errors."""
    _plot(plot_func, tmp_path, **OPTIONS[option])


@pytest.mark.parametrize("option", K_OPTIONS)
def test_lib_plot_vs_k_with_option(option, tmp_path):
    """Test that each option specific to plot_vs_k draws a plot without errors."""
    _plot(plot_vs_k, tmp_path, **K_OPTIONS[option])


def test_lib_plot_vs_k_limits_outside_k_range_are_skipped(caplog, tmp_path):
    """Test that limits without data in the k range are skipped.

    Pal2021 only has data at k = 2.36 h/Mpc.
    """
    caplog.set_level(logging.INFO, logger="ps21cm_limits")
    fig = _plot(
        plot_vs_k,
        tmp_path,
        limits=[*PLOTTING_ARGS["limits"], "Pal2021"],
        k_range=(0.1, 1.0),
    )
    assert not any("Pal" in text for text in _legend_texts(fig))
    assert any(
        "Pal2021" in record.message and "k range" in record.message
        for record in caplog.records
    )


@pytest.mark.parametrize(
    ("plot_func", "color_by", "label"),
    [
        (plot_vs_k, "z", "Redshift"),
        (plot_vs_k, "year", "Year"),
        (plot_vs_z, "k", r"$k$ ($h\,\mathrm{Mpc}^{-1}$)"),
        (plot_vs_z, "year", "Year"),
    ],
)
def test_lib_plot_colorbar(plot_func, color_by, label, tmp_path):
    """Test that the colorbar label matches what the lines are coloured by."""
    fig = _plot(plot_func, tmp_path, color_by=color_by)
    assert len(fig.axes) == 2
    assert fig.axes[1].get_ylabel() == label


@pytest.mark.parametrize("plot_func", PLOT_FUNCS)
def test_lib_plot_without_colorbar(plot_func, tmp_path):
    """Test that the colorbar can be turned off."""
    fig = _plot(plot_func, tmp_path, show_colorbar=False)
    assert len(fig.axes) == 1


@pytest.mark.parametrize(
    ("plot_func", "option", "text"),
    [(plot_vs_k, "z_labels", "z"), (plot_vs_z, "k_labels", "h/Mpc")],
)
def test_lib_plot_ranges_in_legend(plot_func, option, text, tmp_path):
    """Test that each limit's z (or k) range can be shown in the legend."""
    fig = _plot(plot_func, tmp_path, color_by="year", **{option: "legend"})
    limit_texts = [t for t in _legend_texts(fig) if "Theory" not in t]
    assert limit_texts
    assert all(text in t for t in limit_texts)


@pytest.mark.parametrize("plot_func", PLOT_FUNCS)
def test_lib_plot_with_legend_labeler(plot_func, tmp_path):
    """Test that only the datasets given custom labels appear in the legend."""
    fig = _plot(plot_func, tmp_path, legend_labeler={"HERA2023": "HERA 2023"})
    assert _legend_texts(fig) == ["HERA 2023"]


# Tests for plotting sensitivities


def _plot_sensitivities(plot_func, tmp_path: Path, **kwargs):
    """Draw ``plot_func`` with the SKA sensitivities over SENSITIVITY_Z_RANGE."""
    defaults = {"sensitivities": SKA_SENSITIVITIES, "z_range": SENSITIVITY_Z_RANGE}
    return _plot(plot_func, tmp_path, **{**defaults, **kwargs})


def _expected_sensitivity_lines(plot_func, name: str) -> list:
    """Return the (x, y) data of the lines expected for the sensitivity ``name``.

    plot_vs_k draws one line per redshift, and plot_vs_z draws the lowest finite
    sensitivity at each redshift (issue #57).
    """
    data = (
        load_sensitivity_data(name).drop_nan().select_z_range(*SENSITIVITY_Z_RANGE).data
    )
    if plot_func is plot_vs_k:
        return list(zip(data.k, data.delta_squared, strict=True))
    return [(data.z, [np.min(dsq) for dsq in data.delta_squared])]


def _closest_finite(k: np.ndarray, dsq: np.ndarray, k_target: float) -> float:
    """Return the finite value of ``dsq`` at the k bin closest to ``k_target``."""
    finite = np.isfinite(dsq)
    return dsq[finite][np.argmin(np.abs(k[finite] - k_target))]


@pytest.mark.parametrize("plot_func", PLOT_FUNCS)
def test_lib_plot_with_sensitivities(plot_func, tmp_path):
    """Test that the sensitivities are drawn, labelled and kept within the axis.

    Without a delta_squared_range, the bottom of the axis is lowered to the lowest
    sensitivity inside the redshift range, but not to the much lower SKA values
    at z = 5.6.
    """
    ax = _plot_sensitivities(plot_func, tmp_path).axes[0]
    for label, name in SKA_SENSITIVITIES.items():
        for x, y in _expected_sensitivity_lines(plot_func, name):
            assert len(_find_lines(ax, x, y)) == 1
        assert any(label in text.get_text() for text in ax.texts)

    sensitivities = [
        load_sensitivity_data(name).drop_nan() for name in SKA_SENSITIVITIES.values()
    ]
    lowest_in_range = min(
        np.min(dsq)
        for sensitivity in sensitivities
        for dsq in sensitivity.select_z_range(*SENSITIVITY_Z_RANGE).data.delta_squared
    )
    lowest = min(
        np.min(dsq)
        for sensitivity in sensitivities
        for dsq in sensitivity.data.delta_squared
    )
    assert lowest < ax.get_ylim()[0] <= lowest_in_range


@pytest.mark.parametrize("plot_func", PLOT_FUNCS)
def test_lib_plot_with_sensitivity_style(plot_func, tmp_path):
    """Test that sensitivity style parameters are passed on to the lines."""
    style = {"SKA-AA4": {"color": "C3", "linestyle": ":", "alpha": 0.5}}
    ax = _plot_sensitivities(plot_func, tmp_path, sensitivity_style=style).axes[0]
    for x, y in _expected_sensitivity_lines(plot_func, SKA_SENSITIVITIES["SKA-AA4"]):
        (line,) = _find_lines(ax, x, y)
        assert line.get_color() == "C3"
        assert line.get_linestyle() == ":"
        assert line.get_alpha() == pytest.approx(0.5)


@pytest.mark.parametrize(
    "sensitivity_k",
    [
        0.2,
        # Below the first finite k bin: the SKA files hold no sensitivity at low k.
        0.01,
        # Above the last k bin (1.59 h/Mpc).
        10.0,
    ],
)
def test_lib_plot_vs_z_with_sensitivity_k(sensitivity_k, tmp_path):
    """Test that sensitivity_k selects the closest k bin with a finite value."""
    ax = _plot_sensitivities(plot_vs_z, tmp_path, sensitivity_k=sensitivity_k).axes[0]
    for name in SKA_SENSITIVITIES.values():
        data = load_sensitivity_data(name).select_z_range(*SENSITIVITY_Z_RANGE).data
        at_k = [
            _closest_finite(k, dsq, sensitivity_k)
            for k, dsq in zip(data.k, data.delta_squared, strict=True)
        ]
        assert len(_find_lines(ax, data.z, at_k)) == 1


@pytest.mark.parametrize("plot_func", PLOT_FUNCS)
def test_lib_plot_sensitivities_outside_z_range_are_skipped(
    plot_func, caplog, tmp_path
):
    """Test that sensitivities without a redshift in the plotted range are skipped.

    The SKA files have no redshift between z = 16.8 and z = 22.7. The HERA 2026 and
    Ewall-Wice 2016 limits in that range are above the default delta_squared range,
    which is therefore widened.
    """
    ax = _plot_sensitivities(
        plot_func,
        tmp_path,
        z_range=(17.0, 22.0),
        delta_squared_range=(1e0, 1e12),
    ).axes[0]
    for label in SKA_SENSITIVITIES:
        assert not any(label in text.get_text() for text in ax.texts)
        assert label in caplog.text


@pytest.mark.parametrize("plot_func", PLOT_FUNCS)
def test_lib_plot_sensitivity_kind_is_rejected(plot_func, tmp_path):
    """Test that the removed sensitivity_kind style option raises an error."""
    style = {"SKA-AA4": {"sensitivity_kind": "thermal"}}
    with pytest.raises(ValueError, match="sensitivity_kind"):
        _plot_sensitivities(plot_func, tmp_path, sensitivity_style=style)
