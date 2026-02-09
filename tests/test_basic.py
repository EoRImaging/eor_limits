"""Really basic tests."""

from eor_limits import make_plot


def test_make_plot_basic():
    """Test making a basic plot with default parameters."""
    fig = make_plot()
    assert fig is not None
