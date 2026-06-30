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


def _run_plot_command(command: str, *args: str, expect_success: bool = True) -> None:
    """Invoke a CLI plotting command with provided args."""
    with pytest.raises(SystemExit) as exc_info:
        app([command, *args], exit_on_error=True)
    if expect_success:
        assert exc_info.value.code == 0
        if "--out" in args:
            out_index = args.index("--out") + 1
            out_path = Path(args[out_index])
            assert out_path.exists(), f"Expected output file not found: {out_path}"
    else:
        assert exc_info.value.code != 0  # negative test case


def _run_plot_vs_k(*args: str, expect_success: bool = True) -> None:
    """Invoke the CLI plot-vs-k command with provided args."""
    _run_plot_command("plot-vs-k", *args, expect_success=expect_success)


def _run_plot_vs_z(*args: str, expect_success: bool = True) -> None:
    """Invoke the CLI plot-vs-z command with provided args."""
    _run_plot_command("plot-vs-z", *args, expect_success=expect_success)


def _assert_images_match(plot_name: str, test_name: str, cli_path: Path) -> None:
    """Compare a CLI-generated PNG against the library-generated reference.

    This helper skips the test if the reference image is missing, so CLI tests
    can be run in isolation without depending on test_plotting side effects.
    """
    ref_path = OUTPUT_DIR / f"test_lib_{plot_name}_{test_name}.png"
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
    _run_plot_vs_k("--out", str(out), expect_success=True)
    _assert_images_match("plot_vs_k", "basic", out)


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
        expect_success=True,
    )
    _assert_images_match("plot_vs_k", "with_fig_styling", out)


def test_cli_plot_vs_k_with_z_range():
    """Test making a plot with redshift range filtering through the CLI."""
    out = OUTPUT_DIR / "test_cli_plot_vs_k_with_z_range.png"
    _run_plot_vs_k("--z-range", "6.0", "10.0", "--out", str(out), expect_success=True)
    _assert_images_match("plot_vs_k", "with_z_range", out)


def test_cli_plot_vs_k_with_k_range():
    """Test making a plot with k range filtering through the CLI."""
    out = OUTPUT_DIR / "test_cli_plot_vs_k_with_k_range.png"
    _run_plot_vs_k("--k-range", "0.1", "1.0", "--out", str(out), expect_success=True)
    _assert_images_match("plot_vs_k", "with_k_range", out)


def test_cli_plot_vs_k_with_delta_squared_range():
    """Test making a plot with delta squared range filtering through the CLI."""
    out = OUTPUT_DIR / "test_cli_plot_vs_k_with_delta_squared_range.png"
    _run_plot_vs_k(
        "--delta-squared-range", "1e3", "1e5", "--out", str(out), expect_success=True
    )
    _assert_images_match("plot_vs_k", "with_delta_squared_range", out)


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
        expect_success=True,
    )
    _assert_images_match("plot_vs_k", "with_aspoints", out)


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
        expect_success=True,
    )
    _assert_images_match("plot_vs_k", "with_aslines", out)


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
        expect_success=True,
    )
    _assert_images_match("plot_vs_k", "with_bold_limits", out)


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
        expect_success=True,
    )
    _assert_images_match("plot_vs_k", "with_bold_theories", out)


def test_cli_plot_vs_k_with_limit_and_theory_styling():
    """Test making a plot with custom limit/theory styling through the CLI.

    Also exercises the error path for invalid JSON passed to style arguments,
    asserting that the CLI exits non-zero and emits an informative message.
    """
    limits = list(eor_limits.KNOWN_LIMITS.keys())

    # Positive case: valid JSON style arguments work and produce an output file.
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
    _assert_images_match("plot_vs_k", "with_limit_and_theory_styling", out)

    # Negative case: invalid JSON should trigger a non-zero exit and an error message.
    bad_out = OUTPUT_DIR / "test_cli_plot_vs_k_with_invalid_style_json.png"
    _run_plot_vs_k(
        "--limits",
        *limits,
        "--theories",
        "Mesinger2016Faint",
        "Mesinger2016Bright",
        "--base-limit-style",
        '{"this-is": "not valid json",',  # Deliberately malformed JSON.
        "--out",
        str(bad_out),
        expect_success=False,  # Negative test case
    )


def test_cli_plot_vs_k_with_legend_labeler():
    """Test making a plot_vs_k with custom legend labels through the CLI."""
    out = OUTPUT_DIR / "test_cli_plot_vs_k_with_legend_labeler.png"
    _run_plot_vs_k(
        "--limits",
        "HERA2022",
        "HERA2023",
        "Paciga2013",
        "--legend-labeler",
        '{"HERA2023": "HERA 2023"}',
        "--out",
        str(out),
        expect_success=True,
    )
    _assert_images_match("plot_vs_k", "with_legend_labeler", out)


def test_cli_plot_vs_z_basic():
    """Test making a plot_vs_z with default parameters through the CLI."""
    out = OUTPUT_DIR / "test_cli_plot_vs_z_basic.png"
    _run_plot_vs_z("--out", str(out), expect_success=True)
    _assert_images_match("plot_vs_z", "basic", out)


def test_cli_plot_vs_z_with_fig_styling():
    """Test making a plot_vs_z with custom styling parameters through the CLI."""
    out = OUTPUT_DIR / "test_cli_plot_vs_z_with_fig_styling.png"
    _run_plot_vs_z(
        "--fontsize",
        "12",
        "--colormap",
        "viridis",
        "--fig-ratio",
        "1.0",
        "--out",
        str(out),
        expect_success=True,
    )
    _assert_images_match("plot_vs_z", "with_fig_styling", out)


def test_cli_plot_vs_z_with_z_range():
    """Test making a plot_vs_z with redshift range filtering through the CLI."""
    out = OUTPUT_DIR / "test_cli_plot_vs_z_with_z_range.png"
    _run_plot_vs_z("--z-range", "6.0", "10.0", "--out", str(out), expect_success=True)
    _assert_images_match("plot_vs_z", "with_z_range", out)


def test_cli_plot_vs_z_with_delta_squared_range():
    """Test making a plot_vs_z with delta squared range filtering through the CLI."""
    out = OUTPUT_DIR / "test_cli_plot_vs_z_with_delta_squared_range.png"
    _run_plot_vs_z(
        "--delta-squared-range", "1e3", "1e5", "--out", str(out), expect_success=True
    )
    _assert_images_match("plot_vs_z", "with_delta_squared_range", out)


def test_cli_plot_vs_z_with_specific_limits():
    """Test making a plot_vs_z with specific limits through the CLI."""
    out = OUTPUT_DIR / "test_cli_plot_vs_z_with_specific_limits.png"
    _run_plot_vs_z(
        "--limits",
        "HERA2022",
        "HERA2023",
        "Paciga2013",
        "--out",
        str(out),
        expect_success=True,
    )
    _assert_images_match("plot_vs_z", "with_specific_limits", out)


def test_cli_plot_vs_z_with_aspoints():
    """Test making a plot_vs_z with specific limits as points through the CLI."""
    limits = list(eor_limits.KNOWN_LIMITS.keys())[:5]
    out = OUTPUT_DIR / "test_cli_plot_vs_z_with_aspoints.png"
    _run_plot_vs_z(
        "--limits",
        *limits,
        "--aspoints",
        *limits,
        "--out",
        str(out),
        expect_success=True,
    )
    _assert_images_match("plot_vs_z", "with_aspoints", out)


def test_cli_plot_vs_z_with_aslines():
    """Test making a plot_vs_z with specific limits as lines through the CLI."""
    limits = list(eor_limits.KNOWN_LIMITS.keys())[:5]
    out = OUTPUT_DIR / "test_cli_plot_vs_z_with_aslines.png"
    _run_plot_vs_z(
        "--limits",
        *limits,
        "--aslines",
        *limits,
        "--out",
        str(out),
        expect_success=True,
    )
    _assert_images_match("plot_vs_z", "with_aslines", out)


def test_cli_plot_vs_z_with_bold_limits():
    """Test making a plot_vs_z with bolded specific limits through the CLI."""
    out = OUTPUT_DIR / "test_cli_plot_vs_z_with_bold_limits.png"
    _run_plot_vs_z(
        "--limits",
        "HERA2022",
        "HERA2023",
        "Paciga2013",
        "--bold-limits",
        "HERA2022",
        "HERA2023",
        "--out",
        str(out),
        expect_success=True,
    )
    _assert_images_match("plot_vs_z", "with_bold_limits", out)


def test_cli_plot_vs_z_with_limit_styling():
    """Test making a plot_vs_z with custom limit styling through the CLI."""
    out = OUTPUT_DIR / "test_cli_plot_vs_z_with_limit_styling.png"
    _run_plot_vs_z(
        "--limits",
        "HERA2022",
        "HERA2023",
        "Paciga2013",
        "--base-limit-style",
        '{"linewidth": 3, "alpha": 0.8}',
        "--limit-styles",
        '{"HERA2023": {"color": "C3"}}',
        "--out",
        str(out),
        expect_success=True,
    )
    _assert_images_match("plot_vs_z", "with_limit_styling", out)


def test_cli_plot_vs_z_with_legend_labeler():
    """Test making a plot_vs_z with custom legend labels through the CLI."""
    out = OUTPUT_DIR / "test_cli_plot_vs_z_with_legend_labeler.png"
    _run_plot_vs_z(
        "--limits",
        "HERA2022",
        "HERA2023",
        "Paciga2013",
        "--legend-labeler",
        '{"HERA2023": "HERA 2023"}',
        "--out",
        str(out),
        expect_success=True,
    )
    _assert_images_match("plot_vs_z", "with_legend_labeler", out)
