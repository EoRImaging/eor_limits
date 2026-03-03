"""Test CLI plotting commands."""

from pathlib import Path

import pytest

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


def test_cli_app_exists():
    """Test that the CLI app is properly instantiated."""
    assert app is not None


def test_cli_plot_vs_k_basic():
    """Test making a plot with default parameters through the CLI."""
    out = OUTPUT_DIR / "test_cli_plot_vs_k_basic.pdf"
    _run_plot_vs_k("--out", str(out))
    assert out.exists()


def test_cli_plot_vs_k_with_fig_styling():
    """Test making a plot with custom styling parameters through the CLI."""
    out = OUTPUT_DIR / "test_cli_plot_vs_k_with_fig_styling.pdf"
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


def test_cli_plot_vs_k_with_z_range():
    """Test making a plot with redshift range filtering through the CLI."""
    out = OUTPUT_DIR / "test_cli_plot_vs_k_with_z_range.pdf"
    _run_plot_vs_k("--z-range", "6.0", "10.0", "--out", str(out))
    assert out.exists()


def test_cli_plot_vs_k_with_k_range():
    """Test making a plot with k range filtering through the CLI."""
    out = OUTPUT_DIR / "test_cli_plot_vs_k_with_k_range.pdf"
    _run_plot_vs_k("--k-range", "0.1", "1.0", "--out", str(out))
    assert out.exists()


def test_cli_plot_vs_k_with_delta_squared_range():
    """Test making a plot with delta squared range filtering through the CLI."""
    out = OUTPUT_DIR / "test_cli_plot_vs_k_with_delta_squared_range.pdf"
    _run_plot_vs_k("--delta-squared-range", "1e3", "1e5", "--out", str(out))
    assert out.exists()


def test_cli_plot_vs_k_with_aspoints():
    """Test making a plot with specific limits as points through the CLI."""
    limits = list(eor_limits.KNOWN_LIMITS.keys())
    out = OUTPUT_DIR / "test_cli_plot_vs_k_with_aspoints.pdf"
    _run_plot_vs_k(
        "--limits",
        *limits,
        "--aspoints",
        *limits,
        "--out",
        str(out),
    )
    assert out.exists()


def test_cli_plot_vs_k_with_aslines():
    """Test making a plot with specific limits as lines through the CLI."""
    limits = list(eor_limits.KNOWN_LIMITS.keys())
    out = OUTPUT_DIR / "test_cli_plot_vs_k_with_aslines.pdf"
    _run_plot_vs_k(
        "--limits",
        *limits,
        "--aslines",
        *limits,
        "--out",
        str(out),
    )
    assert out.exists()


def test_cli_plot_vs_k_with_bold_limits():
    """Test making a plot with bolded specific limits through the CLI."""
    limits = list(eor_limits.KNOWN_LIMITS.keys())
    out = OUTPUT_DIR / "test_cli_plot_vs_k_with_bold_limits.pdf"
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


def test_cli_plot_vs_k_with_bold_theories():
    """Test making a plot with bolded specific theories through the CLI."""
    theories = list(eor_limits.KNOWN_THEORIES.keys())
    out = OUTPUT_DIR / "test_cli_plot_vs_k_with_bold_theories.pdf"
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


def test_cli_plot_vs_k_with_limit_and_theory_styling():
    """Test making a plot with custom limit/theory styling through the CLI."""
    limits = list(eor_limits.KNOWN_LIMITS.keys())
    out = OUTPUT_DIR / "test_cli_plot_vs_k_with_limit_and_theory_styling.pdf"
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
