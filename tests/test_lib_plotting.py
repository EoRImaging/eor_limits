"""Test the CLI and plotting modules."""

from pathlib import Path

import numpy as np
import pytest

from eor_limits import (
    KNOWN_LIMITS,
    KNOWN_THEORIES,
    load_sensitivity_data,
    plot_vs_k,
    plot_vs_z,
)

# Output directory for test PDFs
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# SKA-Low sensitivities distributed with the package, keyed by plot label
SKA_SENSITIVITIES = {
    "SKA-AA4": "Breitman2026_AA4_deep1080h",
    "SKA-AA*": "Breitman2026_AAstar_deep1080h",
}
SENSITIVITY_LIMITS = ["HERA2022", "HERA2023"]
# Rounded redshift range of the HERA 2022 and 2023 limits (z = 7.9 and 10.4)
HERA_Z_RANGE = (7.0, 11.0)


def _find_lines(ax, x, y) -> list:
    """Return the lines on ``ax`` whose x and y data equal ``x`` and ``y``."""
    return [
        line
        for line in ax.get_lines()
        if len(line.get_xdata()) == len(x)
        and np.allclose(line.get_xdata(), x)
        and np.allclose(line.get_ydata(), y)
    ]


# Tests for plot_vs_k function


def test_lib_plot_vs_k_basic():
    """Test making a plot with default parameters."""
    fig = plot_vs_k(out=OUTPUT_DIR / "test_lib_plot_vs_k_basic.png")
    assert fig is not None


def test_lib_plot_vs_k_with_fig_styling():
    """Test making a plot with custom styling parameters."""
    fig = plot_vs_k(
        fontsize=12,
        colormap="viridis",
        fig_width=20.0,
        fig_ratio=2.0,
        legend_ncols=2,
        out=OUTPUT_DIR / "test_lib_plot_vs_k_with_fig_styling.png",
    )
    assert fig is not None


def test_lib_plot_vs_k_with_publication():
    """Test making a publication-style plot."""
    fig = plot_vs_k(
        publication=True,
        dpi=300,
        out=OUTPUT_DIR / "test_lib_plot_vs_k_with_publication.png",
    )
    assert fig is not None


def test_lib_plot_vs_k_with_z_range():
    """Test making a plot with redshift range filtering."""
    fig = plot_vs_k(
        z_range=(6.0, 10.0), out=OUTPUT_DIR / "test_lib_plot_vs_k_with_z_range.png"
    )
    assert fig is not None


def test_lib_plot_vs_k_with_k_range():
    """Test making a plot with k range filtering."""
    fig = plot_vs_k(
        k_range=(0.1, 1.0), out=OUTPUT_DIR / "test_lib_plot_vs_k_with_k_range.png"
    )
    assert fig is not None


def test_lib_plot_vs_k_with_delta_squared_range():
    """Test making a plot with delta_squared range filtering."""
    fig = plot_vs_k(
        delta_squared_range=(1e3, 1e5),
        out=OUTPUT_DIR / "test_lib_plot_vs_k_with_delta_squared_range.png",
    )
    assert fig is not None


def test_lib_plot_vs_k_with_aspoints():
    """Test making a plot with specific limits as points."""
    limits = list(KNOWN_LIMITS.keys())
    fig = plot_vs_k(
        limits=limits,
        aspoints=limits,
        out=OUTPUT_DIR / "test_lib_plot_vs_k_with_aspoints.png",
    )
    assert fig is not None


def test_lib_plot_vs_k_with_aslines():
    """Test making a plot with specific limits as lines."""
    limits = list(KNOWN_LIMITS.keys())
    fig = plot_vs_k(
        limits=limits,
        aslines=limits,
        out=OUTPUT_DIR / "test_lib_plot_vs_k_with_aslines.png",
    )
    assert fig is not None


def test_lib_plot_vs_k_with_bold_limits():
    """Test making a plot with bolded specific limits."""
    limits = list(KNOWN_LIMITS.keys())
    bold_limits = ["HERA2022", "HERA2023"]
    fig = plot_vs_k(
        limits=limits,
        bold_limits=bold_limits,
        out=OUTPUT_DIR / "test_lib_plot_vs_k_with_bold_limits.png",
    )
    assert fig is not None


def test_lib_plot_vs_k_with_bold_theories():
    """Test making a plot with bolded specific theories."""
    theories = list(KNOWN_THEORIES.keys())
    bold_theories = ["Mesinger2016Faint", "Mesinger2016Bright"]
    fig = plot_vs_k(
        theories=theories,
        bold_theories=bold_theories,
        out=OUTPUT_DIR / "test_lib_plot_vs_k_with_bold_theories.png",
    )
    assert fig is not None


def test_lib_plot_vs_k_with_limit_and_theory_styling():
    """Test making a plot with custom styling parameters for limits and theories."""
    limits = list(KNOWN_LIMITS.keys())
    theories = ["Mesinger2016Faint", "Mesinger2016Bright"]
    fig = plot_vs_k(
        limits=limits,
        theories=theories,
        base_limit_style={"linewidth": 5, "shade_alpha": 0.1},
        limit_styles={"HERA2023": {"shade_alpha": 0.25, "shade_color": "C3"}},
        base_theory_style={"linestyle": "-."},
        theory_styles={
            "Mesinger2016Faint": {
                "color": "C0",
                "shade_alpha": 0.5,
                "shade_color": "C2",
            },
            "Mesinger2016Bright": {
                "color": "C1",
                "shade_alpha": 0.1,
                "shade_color": "C2",
            },
        },
        out=OUTPUT_DIR / "test_lib_plot_vs_k_with_limit_and_theory_styling.png",
    )
    assert fig is not None


def test_lib_plot_vs_k_without_colorbar():
    """Test making a plot without a redshift colorbar."""
    fig = plot_vs_k(
        show_colorbar=False,
        out=OUTPUT_DIR / "test_lib_plot_vs_k_without_colorbar.png",
    )
    assert fig is not None
    assert len(fig.axes) == 1


def test_lib_plot_vs_k_with_color_by_year():
    """Test making a plot_vs_k colored by year."""
    fig = plot_vs_k(
        color_by="year",
        show_colorbar=True,
        out=OUTPUT_DIR / "test_lib_plot_vs_k_with_color_by_year.png",
    )

    assert fig is not None
    assert len(fig.axes) == 2
    assert fig.axes[1].get_ylabel() == "Year"


def test_lib_plot_vs_k_with_z_labels_in_legend():
    """Test making a plot_vs_k with the z range in the legend."""
    fig = plot_vs_k(
        color_by="year",
        z_labels="legend",
        out=OUTPUT_DIR / "test_lib_plot_vs_k_with_z_labels_in_legend.png",
    )

    legend = fig.axes[0].get_legend()
    assert fig is not None
    assert legend is not None
    assert all("z" in text.get_text() for text in legend.get_texts())


def test_lib_plot_vs_k_with_legend_labeler():
    """Test making a plot_vs_k with custom legend labels."""
    fig = plot_vs_k(
        limits=["HERA2022", "HERA2023", "Paciga2013"],
        legend_labeler={"HERA2023": "HERA 2023"},
        out=OUTPUT_DIR / "test_lib_plot_vs_k_with_legend_labeler.png",
    )

    legend = fig.axes[0].get_legend()
    assert fig is not None
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["HERA 2023"]


# Tests for plot_vs_z function


def test_lib_plot_vs_z_basic():
    """Test making a plot_vs_z with default parameters."""
    fig = plot_vs_z(out=OUTPUT_DIR / "test_lib_plot_vs_z_basic.png")
    assert fig is not None
    assert len(fig.axes) == 2
    assert fig.axes[1].get_ylabel() == r"$k$ ($h\,\mathrm{Mpc}^{-1}$)"


def test_lib_plot_vs_z_with_fig_styling():
    """Test making a plot_vs_z with custom styling parameters."""
    fig = plot_vs_z(
        fontsize=12,
        colormap="viridis",
        fig_ratio=1.0,
        out=OUTPUT_DIR / "test_lib_plot_vs_z_with_fig_styling.png",
    )
    assert fig is not None


def test_lib_plot_vs_z_with_publication():
    """Test making a publication-style plot."""
    fig = plot_vs_z(
        publication=True,
        dpi=300,
        out=OUTPUT_DIR / "test_lib_plot_vs_z_with_publication.png",
    )
    assert fig is not None


def test_lib_plot_vs_z_with_z_range():
    """Test making a plot_vs_z with redshift range filtering."""
    fig = plot_vs_z(
        z_range=(6.0, 10.0), out=OUTPUT_DIR / "test_lib_plot_vs_z_with_z_range.png"
    )
    assert fig is not None


def test_lib_plot_vs_z_with_delta_squared_range():
    """Test making a plot_vs_z with delta_squared range filtering."""
    fig = plot_vs_z(
        delta_squared_range=(1e3, 1e5),
        out=OUTPUT_DIR / "test_lib_plot_vs_z_with_delta_squared_range.png",
    )
    assert fig is not None


def test_lib_plot_vs_z_with_specific_limits():
    """Test making a plot_vs_z with specific limits."""
    limits = ["HERA2022", "HERA2023", "Paciga2013"]
    fig = plot_vs_z(
        limits=limits,
        out=OUTPUT_DIR / "test_lib_plot_vs_z_with_specific_limits.png",
    )
    assert fig is not None


def test_lib_plot_vs_z_with_aspoints():
    """Test making a plot_vs_z with specific limits as points."""
    limits = list(KNOWN_LIMITS.keys())[:5]
    fig = plot_vs_z(
        limits=limits,
        aspoints=limits,
        out=OUTPUT_DIR / "test_lib_plot_vs_z_with_aspoints.png",
    )
    assert fig is not None


def test_lib_plot_vs_z_with_aslines():
    """Test making a plot_vs_z with specific limits as lines."""
    limits = list(KNOWN_LIMITS.keys())[:5]
    fig = plot_vs_z(
        limits=limits,
        aslines=limits,
        out=OUTPUT_DIR / "test_lib_plot_vs_z_with_aslines.png",
    )
    assert fig is not None


def test_lib_plot_vs_z_with_bold_limits():
    """Test making a plot_vs_z with bolded specific limits."""
    limits = ["HERA2022", "HERA2023", "Paciga2013"]
    bold_limits = ["HERA2022", "HERA2023"]
    fig = plot_vs_z(
        limits=limits,
        bold_limits=bold_limits,
        out=OUTPUT_DIR / "test_lib_plot_vs_z_with_bold_limits.png",
    )
    assert fig is not None


def test_lib_plot_vs_z_with_bold_theories():
    """Test making a plot_vs_z with bolded specific theories."""
    theories = list(KNOWN_THEORIES.keys())
    bold_theories = ["Mesinger2016Faint", "Mesinger2016Bright"]
    fig = plot_vs_z(
        theories=theories,
        bold_theories=bold_theories,
        out=OUTPUT_DIR / "test_lib_plot_vs_z_with_bold_theories.png",
    )
    assert fig is not None


def test_lib_plot_vs_z_with_limit_and_theory_styling():
    """Test making a plot_vs_z with custom styling for limits and theories."""
    limits = ["HERA2022", "HERA2023", "Paciga2013"]
    theories = ["Mesinger2016Faint", "Mesinger2016Bright"]
    fig = plot_vs_z(
        limits=limits,
        theories=theories,
        base_limit_style={"linewidth": 3, "alpha": 0.8},
        limit_styles={"HERA2023": {"color": "C3"}},
        base_theory_style={"linestyle": "-."},
        theory_styles={
            "Mesinger2016Faint": {
                "color": "C0",
                "shade_alpha": 0.5,
                "shade_color": "C2",
            },
            "Mesinger2016Bright": {
                "color": "C1",
                "shade_alpha": 0.1,
                "shade_color": "C2",
            },
        },
        out=OUTPUT_DIR / "test_lib_plot_vs_z_with_limit_and_theory_styling.png",
    )
    assert fig is not None


def test_lib_plot_vs_z_without_colorbar():
    """Test making a plot_vs_z without a colorbar."""
    fig = plot_vs_z(
        show_colorbar=False,
        out=OUTPUT_DIR / "test_lib_plot_vs_z_without_colorbar.png",
    )
    assert fig is not None
    assert len(fig.axes) == 1


def test_lib_plot_vs_z_with_color_by_year():
    """Test making a plot_vs_z colored by year."""
    fig = plot_vs_z(
        color_by="year",
        show_colorbar=True,
        out=OUTPUT_DIR / "test_lib_plot_vs_z_with_color_by_year.png",
    )

    assert fig is not None
    assert len(fig.axes) == 2
    assert fig.axes[1].get_ylabel() == "Year"


def test_lib_plot_vs_z_with_k_labels_in_legend():
    """Test making a plot_vs_z with the k range in the legend."""
    fig = plot_vs_z(
        color_by="year",
        k_labels="legend",
        out=OUTPUT_DIR / "test_lib_plot_vs_z_with_k_labels_in_legend.png",
    )

    legend = fig.axes[0].get_legend()
    assert fig is not None
    assert legend is not None
    assert all("h/Mpc" in text.get_text() for text in legend.get_texts())


def test_lib_plot_vs_z_with_legend_labeler():
    """Test making a plot_vs_z with custom legend labels."""
    fig = plot_vs_z(
        limits=["HERA2022", "HERA2023", "Paciga2013"],
        legend_labeler={"HERA2023": "HERA 2023"},
        out=OUTPUT_DIR / "test_lib_plot_vs_z_with_legend_labeler.png",
    )

    legend = fig.axes[0].get_legend()
    assert fig is not None
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["HERA 2023"]


# Tests for plotting sensitivities


def _closest_finite(k: np.ndarray, dsq: np.ndarray, k_target: float) -> float:
    """Return the finite value of ``dsq`` at the k bin closest to ``k_target``."""
    finite = np.isfinite(dsq)
    return dsq[finite][np.argmin(np.abs(k[finite] - k_target))]


def test_lib_plot_vs_k_with_sensitivities():
    """Test that sensitivities are plotted vs k, one line per redshift in range."""
    fig = plot_vs_k(
        limits=SENSITIVITY_LIMITS,
        z_range=(7.5, 8.5),
        sensitivities=SKA_SENSITIVITIES,
        out=OUTPUT_DIR / "test_lib_plot_vs_k_with_sensitivities.png",
    )
    ax = fig.axes[0]
    for label, name in SKA_SENSITIVITIES.items():
        expected = load_sensitivity_data(name).drop_nan().select_z_range(7.5, 8.5)
        for k, dsq in zip(expected.data.k, expected.data.delta_squared, strict=True):
            assert len(_find_lines(ax, k, dsq)) == 1
        assert any(label in text.get_text() for text in ax.texts)


def test_lib_plot_vs_k_with_sensitivity_style():
    """Test that sensitivity style parameters are passed on to the lines."""
    name = SKA_SENSITIVITIES["SKA-AA4"]
    fig = plot_vs_k(
        limits=SENSITIVITY_LIMITS,
        sensitivities={"SKA-AA4": name},
        sensitivity_style={"SKA-AA4": {"color": "C3", "linestyle": ":", "alpha": 0.5}},
    )
    ax = fig.axes[0]
    expected = load_sensitivity_data(name).drop_nan().select_z_range(*HERA_Z_RANGE)
    for k, dsq in zip(expected.data.k, expected.data.delta_squared, strict=True):
        (line,) = _find_lines(ax, k, dsq)
        assert line.get_color() == "C3"
        assert line.get_linestyle() == ":"
        assert line.get_alpha() == pytest.approx(0.5)


def test_lib_plot_vs_z_with_sensitivities():
    """Test that sensitivities are plotted vs z (issue #57).

    By default the lowest finite sensitivity at each redshift is plotted.
    """
    fig = plot_vs_z(
        limits=SENSITIVITY_LIMITS,
        sensitivities=SKA_SENSITIVITIES,
        out=OUTPUT_DIR / "test_lib_plot_vs_z_with_sensitivities.png",
    )
    ax = fig.axes[0]
    for label, name in SKA_SENSITIVITIES.items():
        expected = load_sensitivity_data(name).drop_nan().select_z_range(*HERA_Z_RANGE)
        lowest = [np.min(dsq) for dsq in expected.data.delta_squared]
        assert len(_find_lines(ax, expected.data.z, lowest)) == 1
        assert any(label in text.get_text() for text in ax.texts)


def test_lib_plot_vs_z_with_sensitivity_k():
    """Test plotting sensitivities vs z at a chosen k."""
    fig = plot_vs_z(
        limits=SENSITIVITY_LIMITS,
        sensitivities=SKA_SENSITIVITIES,
        sensitivity_k=0.2,
        out=OUTPUT_DIR / "test_lib_plot_vs_z_with_sensitivity_k.png",
    )
    ax = fig.axes[0]
    for name in SKA_SENSITIVITIES.values():
        expected = load_sensitivity_data(name).select_z_range(*HERA_Z_RANGE)
        at_k = [
            _closest_finite(k, dsq, 0.2)
            for k, dsq in zip(expected.data.k, expected.data.delta_squared, strict=True)
        ]
        assert len(_find_lines(ax, expected.data.z, at_k)) == 1


@pytest.mark.parametrize(
    "sensitivity_k",
    [
        # Below the first finite k bin: the SKA files hold no sensitivity at low k.
        0.01,
        # Above the last k bin (1.59 h/Mpc).
        10.0,
    ],
)
def test_lib_plot_vs_z_sensitivity_k_selects_closest_finite_k(sensitivity_k):
    """Test that sensitivity_k selects the closest k bin with a finite value."""
    name = SKA_SENSITIVITIES["SKA-AA4"]
    fig = plot_vs_z(
        limits=SENSITIVITY_LIMITS,
        sensitivities={"SKA-AA4": name},
        sensitivity_k=sensitivity_k,
    )
    ax = fig.axes[0]
    expected = load_sensitivity_data(name).select_z_range(*HERA_Z_RANGE)
    at_k = [
        _closest_finite(k, dsq, sensitivity_k)
        for k, dsq in zip(expected.data.k, expected.data.delta_squared, strict=True)
    ]
    assert len(_find_lines(ax, expected.data.z, at_k)) == 1


@pytest.mark.parametrize("plot_func", [plot_vs_k, plot_vs_z])
def test_lib_plot_sensitivities_outside_z_range_are_skipped(plot_func, caplog):
    """Test that sensitivities without a redshift in the plotted range are skipped.

    The Eastwood 2019 limit is at z = 18.4, between the SKA bands at z = 16.8
    and z = 22.7. Its values are above the default delta_squared range, which is
    therefore widened.
    """
    fig = plot_func(
        limits=["Eastwood2019"],
        delta_squared_range=(1e0, 1e12),
        sensitivities=SKA_SENSITIVITIES,
    )
    ax = fig.axes[0]
    for label in SKA_SENSITIVITIES:
        assert not any(label in text.get_text() for text in ax.texts)
        assert label in caplog.text


@pytest.mark.parametrize("plot_func", [plot_vs_k, plot_vs_z])
def test_lib_plot_sensitivities_extend_default_delta_squared_range(plot_func):
    """Test that the default delta_squared range includes the sensitivities.

    Only the redshifts inside the redshift range set the bottom of the range, so
    the much lower SKA values at z = 5.6 do not stretch the axis.
    """
    fig = plot_func(limits=SENSITIVITY_LIMITS, sensitivities=SKA_SENSITIVITIES)
    sensitivities = [
        load_sensitivity_data(name).drop_nan() for name in SKA_SENSITIVITIES.values()
    ]
    lowest_in_range = min(
        np.min(dsq)
        for sensitivity in sensitivities
        for dsq in sensitivity.select_z_range(*HERA_Z_RANGE).data.delta_squared
    )
    lowest = min(
        np.min(dsq)
        for sensitivity in sensitivities
        for dsq in sensitivity.data.delta_squared
    )
    ylim_low = fig.axes[0].get_ylim()[0]
    assert ylim_low <= lowest_in_range
    assert ylim_low > lowest


@pytest.mark.parametrize("plot_func", [plot_vs_k, plot_vs_z])
def test_lib_plot_sensitivity_kind_is_rejected(plot_func):
    """Test that the removed sensitivity_kind style option raises an error."""
    with pytest.raises(ValueError, match="sensitivity_kind"):
        plot_func(
            limits=SENSITIVITY_LIMITS,
            sensitivities=SKA_SENSITIVITIES,
            sensitivity_style={"SKA-AA4": {"sensitivity_kind": "thermal"}},
        )
