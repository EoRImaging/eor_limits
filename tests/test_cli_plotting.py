"""Test CLI plotting commands."""

from pathlib import Path

import pytest
from matplotlib.testing.compare import compare_images

import eor_limits
from eor_limits._cli import app

OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Tolerance for pixel-level image comparison (in RMS pixel units).
IMAGE_TOL = 0.1


def _run_plot_vs_k(*args: str) -> None:
    """Invoke the CLI plot-vs-k command with provided args."""
    with pytest.raises(SystemExit) as exc_info:
        app(["plot-vs-k", *args], exit_on_error=False)
    assert exc_info.value.code == 0


def _assert_images_match(test_name: str, cli_path: Path) -> None:
    """Compare a CLI-generated PNG against the library-generated reference.

    This helper skips the test if the reference image is missing, so CLI tests
    can be run in isolation without depending on test_plotting side effects.
    """
    ref_path = OUTPUT_DIR / f"test_lib_plot_vs_k_{test_name}.png"
    if not ref_path.exists():
        pytest.skip(
            f"Reference image not found: {ref_path}. "
            "Generate reference images via the plotting API before "
            "running CLI image-comparison tests. "
        )

    result = compare_images(str(ref_path), str(cli_path), tol=IMAGE_TOL)
    assert result is None, result


def test_cli_app_exists():
    """Test that the CLI app is properly instantiated."""
    assert app is not None


def test_cli_plot_vs_k_basic():
    """Test making a plot with default parameters through the CLI."""
    out = OUTPUT_DIR / "test_cli_plot_vs_k_basic.png"
    _run_plot_vs_k("--out", str(out))
    assert out.exists()
    _assert_images_match("basic", out)


def test_cli_plot_vs_k_with_fig_styling():
    """Test making a plot with custom styling parameters through the CLI."""
    out = OUTPUT_DIR / "test_cli_plot_vs_k_with_fig_styling.png"
    _run_plot_vs_k(
        "--fontsize",
        "12",
        "--colormap",
        "viridis",
        "--fig-ratio",
        "2.0",
        "--out",
        str(out),
    )
    assert out.exists()
    _assert_images_match("with_fig_styling", out)


def test_cli_plot_vs_k_with_z_range():
    """Test making a plot with redshift range filtering through the CLI."""
    out = OUTPUT_DIR / "test_cli_plot_vs_k_with_z_range.png"
    _run_plot_vs_k("--z-range", "6.0", "10.0", "--out", str(out))
    assert out.exists()
    _assert_images_match("with_z_range", out)


def test_cli_plot_vs_k_with_k_range():
    """Test making a plot with k range filtering through the CLI."""
    out = OUTPUT_DIR / "test_cli_plot_vs_k_with_k_range.png"
    _run_plot_vs_k("--k-range", "0.1", "1.0", "--out", str(out))
    assert out.exists()
    _assert_images_match("with_k_range", out)


def test_cli_plot_vs_k_with_delta_squared_range():
    """Test making a plot with delta squared range filtering through the CLI."""
    out = OUTPUT_DIR / "test_cli_plot_vs_k_with_delta_squared_range.png"
    _run_plot_vs_k("--delta-squared-range", "1e3", "1e5", "--out", str(out))
    assert out.exists()
    _assert_images_match("with_delta_squared_range", out)


def test_cli_plot_vs_k_with_aspoints():
    """Test making a plot with specific limits as points through the CLI."""
    limits = list(eor_limits.KNOWN_LIMITS.keys())
    out = OUTPUT_DIR / "test_cli_plot_vs_k_with_aspoints.png"
    _run_plot_vs_k(
        "--limits",
        *limits,
        "--aspoints",
        *limits,
        "--out",
        str(out),
    )
    assert out.exists()
    _assert_images_match("with_aspoints", out)


def test_cli_plot_vs_k_with_aslines():
    """Test making a plot with specific limits as lines through the CLI."""
    limits = list(eor_limits.KNOWN_LIMITS.keys())
    out = OUTPUT_DIR / "test_cli_plot_vs_k_with_aslines.png"
    _run_plot_vs_k(
        "--limits",
        *limits,
        "--aslines",
        *limits,
        "--out",
        str(out),
    )
    assert out.exists()
    _assert_images_match("with_aslines", out)


def test_cli_plot_vs_k_with_bold_limits():
    """Test making a plot with bolded specific limits through the CLI."""
    limits = list(eor_limits.KNOWN_LIMITS.keys())
    out = OUTPUT_DIR / "test_cli_plot_vs_k_with_bold_limits.png"
    _run_plot_vs_k(
        "--limits",
        *limits,
        "--bold-limits",
        "HERA2022",
        "HERA2023",
        "--out",
        str(out),
    )
    assert out.exists()
    _assert_images_match("with_bold_limits", out)


def test_cli_plot_vs_k_with_bold_theories():
    """Test making a plot with bolded specific theories through the CLI."""
    theories = list(eor_limits.KNOWN_THEORIES.keys())
    out = OUTPUT_DIR / "test_cli_plot_vs_k_with_bold_theories.png"
    _run_plot_vs_k(
        "--theories",
        *theories,
        "--bold-theories",
        "Mesinger2016Faint",
        "Mesinger2016Bright",
        "--out",
        str(out),
    )
    assert out.exists()
    _assert_images_match("with_bold_theories", out)


def test_cli_plot_vs_k_with_limit_and_theory_styling():
    """Test making a plot with custom limit/theory styling through the CLI."""
    limits = list(eor_limits.KNOWN_LIMITS.keys())
    out = OUTPUT_DIR / "test_cli_plot_vs_k_with_limit_and_theory_styling.png"
    _run_plot_vs_k(
        "--limits",
        *limits,
        "--theories",
        "Mesinger2016Faint",
        "Mesinger2016Bright",
        "--base-limit-style",
        '{"linewidth": 5, "shade_alpha": 0.1}',
        "--limit-styles",
        '{"HERA2023": {"shade_alpha": 0.25, "shade_color": "C3"}}',
        "--base-theory-style",
        '{"linestyle": "-."}',
        "--theory-styles",
        '{"Mesinger2016Faint": {"color": "C0",\
            "shade_alpha": 0.5, "shade_color": "C2"},\
        "Mesinger2016Bright": {"color": "C1",\
            "shade_alpha": 0.1, "shade_color": "C2"}}',
        "--out",
        str(out),
    )
    assert out.exists()
    _assert_images_match("with_limit_and_theory_styling", out)
