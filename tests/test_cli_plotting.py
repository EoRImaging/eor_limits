"""Test the command-line interface."""

import inspect
from pathlib import Path

import pytest

from eor_limits import _cli

# Output directory for the test figures, kept for inspection
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Command lines covering each kind of option, and the arguments they should pass to
# the plotting function
CLI_CASES = [
    pytest.param(
        ["plot-vs-k", "--limits", "HERA2022", "HERA2023", "--aspoints", "HERA2022"],
        {"limits": ["HERA2022", "HERA2023"], "aspoints": ["HERA2022"]},
        id="lists",
    ),
    pytest.param(
        ["plot-vs-k", "--z-range", "6", "10", "--delta-squared-range", "1e3", "1e5"],
        {"z_range": (6.0, 10.0), "delta_squared_range": (1e3, 1e5)},
        id="ranges",
    ),
    pytest.param(
        ["plot-vs-k", "--publication", "--show-colorbar=False"],
        {"publication": True, "show_colorbar": False},
        id="flags",
    ),
    pytest.param(
        ["plot-vs-k", "--fontsize", "12", "--color-by", "year", "--legend-ncols", "2"],
        {"fontsize": 12.0, "color_by": "year", "legend_ncols": 2},
        id="values",
    ),
    pytest.param(
        [
            "plot-vs-k",
            "--limit-styles",
            '{"HERA2023": {"color": "C3"}}',
            "--legend-labeler",
            '{"HERA2023": "HERA 2023"}',
            "--theory-redshifts",
            '{"Mesinger2016Faint": [8, 10]}',
        ],
        {
            "limit_styles": {"HERA2023": {"color": "C3"}},
            "legend_labeler": {"HERA2023": "HERA 2023"},
            "theory_redshifts": {"Mesinger2016Faint": [8, 10]},
        },
        id="json",
    ),
    pytest.param(
        [
            "plot-vs-z",
            "--sensitivities.SKA-AA4",
            "Breitman2026_AA4_deep1080h",
            "--sensitivity-k",
            "0.2",
            "--sensitivity-style",
            '{"SKA-AA4": {"color": "C3"}}',
        ],
        {
            "sensitivities": {"SKA-AA4": "Breitman2026_AA4_deep1080h"},
            "sensitivity_k": 0.2,
            "sensitivity_style": {"SKA-AA4": {"color": "C3"}},
        },
        id="plot-vs-z",
    ),
]


def _run(*args: str) -> int:
    """Run the CLI with ``args`` and return its exit code."""
    with pytest.raises(SystemExit) as exc_info:
        _cli.app(list(args), exit_on_error=True)
    return exc_info.value.code


@pytest.fixture
def plot_calls(monkeypatch) -> list[tuple[str, dict]]:
    """List of the calls the CLI makes to the plotting functions.

    The plotting functions are replaced, so nothing is drawn. Each call is recorded
    as the function name and its arguments by parameter name (the CLI passes some
    of them by position).
    """
    calls = []

    def recorder(plot_func):
        signature = inspect.signature(plot_func)

        def record(*args, **kwargs):
            arguments = signature.bind(*args, **kwargs).arguments
            calls.append((plot_func.__name__, arguments))

        return record

    monkeypatch.setattr(_cli, "_plot_vs_k", recorder(_cli._plot_vs_k))
    monkeypatch.setattr(_cli, "_plot_vs_z", recorder(_cli._plot_vs_z))
    return calls


@pytest.mark.parametrize(("args", "expected"), CLI_CASES)
def test_cli_passes_options_to_plotting(plot_calls, args, expected):
    """Test that each kind of option is passed on as the right argument."""
    assert _run(*args) == 0
    assert plot_calls == [(args[0].replace("-", "_"), expected)]


def test_cli_rejects_invalid_json(plot_calls):
    """Test that an invalid JSON option stops the CLI before anything is plotted."""
    assert _run("plot-vs-k", "--base-limit-style", '{"not": "valid json",') != 0
    assert plot_calls == []


@pytest.mark.parametrize("command", ["plot-vs-k", "plot-vs-z"])
def test_cli_reports_plotting_errors(monkeypatch, capsys, command):
    """Test that an error while plotting is shown and gives a non-zero exit code."""

    def fail(*args, **kwargs):
        raise ValueError("Something went wrong")

    monkeypatch.setattr(_cli, f"_{command.replace('-', '_')}", fail)
    monkeypatch.delenv("EOR_LIMITS_DEBUG", raising=False)
    assert _run(command) == 1
    assert "Something went wrong" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["plot-vs-k", "plot-vs-z"])
def test_cli_draws_plot(command):
    """Test that each command draws a plot and saves it."""
    out = OUTPUT_DIR / f"test_cli_{command.replace('-', '_')}.png"
    out.unlink(missing_ok=True)
    assert _run(command, "--limits", "HERA2022", "HERA2023", "--out", str(out)) == 0
    assert out.exists()
