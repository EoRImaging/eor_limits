"""Test the CLI and plotting modules."""

from eor_limits._cli import app
from eor_limits import KNOWN_LIMITS, KNOWN_THEORIES, load_theory_model, make_plot


def test_cli_app_exists():
    """Test that the CLI app is properly instantiated."""
    assert app is not None


def test_make_plot_basic():
    """Test making a plot with default parameters."""
    fig = make_plot()
    assert fig is not None


def test_make_plot_with_specific_limits():
    """Test making a plot with specific limits."""
    limits = list(KNOWN_LIMITS.keys())[:2]  # Use first 2 limits
    fig = make_plot(limits=limits, 
                    out='test_make_plot_with_specific_limits.pdf')
    assert fig is not None


def test_make_plot_with_multiple_theories():
    """Test making a plot with multiple theories."""
    theories = list(KNOWN_THEORIES.keys())[:2]  # Use first 2 theories
    theory_redshifts = []
    for theory in theories:
        theory = load_theory_model(theory)
        theory_redshifts.append(theory.data.z[:3])
    fig = make_plot(theories=theories, 
                    out='test_make_plot_with_multiple_theories.pdf')
    assert fig is not None


def test_make_plot_with_limits_and_theories():
    """Test making a plot with both limits and theories."""
    limits = list(KNOWN_LIMITS.keys())[:2]  # Use first 2 limits
    theories = list(KNOWN_THEORIES.keys())[:2]  # Use first 2 theories
    fig = make_plot(limits=limits, theories=theories, 
                    out='test_make_plot_with_limits_and_theories.pdf')
    assert fig is not None

def test_make_plot_with_styling_options():
    """Test making a plot with custom styling parameters."""
    fig = make_plot(fontsize=12, colormap="viridis", shade_limits=0.3,
                    out='test_make_plot_with_styling_options.pdf')
    assert fig is not None


def test_make_plot_with_z_range():
    """Test making a plot with redshift range filtering."""
    fig = make_plot(z_range=(6.0, 10.0), 
                    out='test_make_plot_with_z_range.pdf')
    assert fig is not None


def test_make_plot_with_k_range():
    """Test making a plot with k range filtering."""
    fig = make_plot(k_range=(0.1, 1.0), 
                    out='test_make_plot_with_k_range.pdf')
    assert fig is not None


def test_make_plot_with_delta_squared_range():
    """Test making a plot with delta_squared range filtering."""
    fig = make_plot(delta_squared_range=(1e3, 1e5), 
                    out='test_make_plot_with_delta_squared_range.pdf')
    assert fig is not None


def test_make_plot_with_bold_limits():
    """Test making a plot with bolded specific limits."""
    limits = list(KNOWN_LIMITS.keys())[:3]
    bold_limits = limits[:2]
    fig = make_plot(limits=limits, bold_limits=bold_limits,
                    out='test_make_plot_with_bold_limits.pdf')
    assert fig is not None


def test_make_plot_no_shade_limits():
    """Test making a plot with shading disabled for limits."""
    fig = make_plot(shade_limits=None, 
                    out='test_make_plot_no_shade_limits.pdf')
    assert fig is not None


def test_make_plot_with_aspoints():
    """Test making a plot with specific limits as points."""
    limits = list(KNOWN_LIMITS.keys())[:2]
    fig = make_plot(limits=limits, aspoints=limits,
                    out='test_make_plot_with_aspoints.pdf')
    assert fig is not None


def test_make_plot_with_aslines():
    """Test making a plot with specific limits as lines."""
    limits = list(KNOWN_LIMITS.keys())[:2]
    fig = make_plot(limits=limits, aslines=limits,
                    out='test_make_plot_with_aslines.pdf')
    assert fig is not None


def test_make_plot_with_bold_theories():
    """Test making a plot with bolded specific theories."""
    theories = list(KNOWN_THEORIES.keys())[:2]
    bold_theories = theories[:2]
    fig = make_plot(theories=theories, bold_theories=bold_theories,
                    out='test_make_plot_with_bold_theories.pdf')
    assert fig is not None


def test_make_plot_with_dark_shade_theories():
    """Test making a plot with dark shading for theories."""
    theories = list(KNOWN_THEORIES.keys())[:2]
    fig = make_plot(theories=theories, shade_theories=0.75,
                    out='test_make_plot_with_dark_shade_theories.pdf')
    assert fig is not None


def test_make_plot_custom_fig_ratio():
    """Test making a plot with custom figure aspect ratio."""
    fig = make_plot(fig_ratio=2.0, 
                    out='test_make_plot_custom_fig_ratio.pdf')
    assert fig is not None
