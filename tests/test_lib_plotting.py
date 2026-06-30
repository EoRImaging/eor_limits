"""Test the CLI and plotting modules."""

from pathlib import Path

from eor_limits import KNOWN_LIMITS, KNOWN_THEORIES, plot_vs_k, plot_vs_z

# Output directory for test PDFs
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)


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
        leg_cols=2,
        out=OUTPUT_DIR / "test_lib_plot_vs_k_with_fig_styling.png",
    )
    assert fig is not None


def test_lib_plot_vs_k_without_colorbar():
    """Test making a plot without a redshift colorbar."""
    fig = plot_vs_k(
        colorbar=False,
        out=OUTPUT_DIR / "test_lib_plot_vs_k_without_colorbar.png",
    )
    assert fig is not None
    assert len(fig.axes) == 1


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


def test_lib_plot_vs_z_with_fig_styling():
    """Test making a plot_vs_z with custom styling parameters."""
    fig = plot_vs_z(
        fontsize=12,
        colormap="viridis",
        fig_ratio=1.0,
        out=OUTPUT_DIR / "test_lib_plot_vs_z_with_fig_styling.png",
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


def test_lib_plot_vs_z_with_limit_styling():
    """Test making a plot_vs_z with custom styling for limits."""
    limits = ["HERA2022", "HERA2023", "Paciga2013"]
    fig = plot_vs_z(
        limits=limits,
        base_limit_style={"linewidth": 3, "alpha": 0.8},
        limit_styles={"HERA2023": {"color": "C3"}},
        out=OUTPUT_DIR / "test_lib_plot_vs_z_with_limit_styling.png",
    )
    assert fig is not None


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
