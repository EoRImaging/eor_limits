#! /usr/bin/env python
# Copyright (c) 2019 Nichole Barry, Bryna Hazelton
# Licensed under the 2-clause BSD License
"""CLI entry points for eor_limits."""

import functools
import logging
import os
import sys
from typing import Any

from cyclopts import App
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text
from rich.traceback import install

from ._plot import plot_vs_k as _plot_vs_k
from ._plot import plot_vs_z as _plot_vs_z

install(show_locals=True)
app = App(
    help="A package for plotting and comparing "
    "21-cm power spectrum limits and theories.",
    help_format="rich",
)
logger = logging.getLogger("eor_limits")
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
console = Console()


class CLIError(Exception):
    def __init__(self, message: str, *, title: str = "Error"):
        super().__init__(message)
        self.message = message
        self.title = title

    def render(self):
        return Panel(
            Text(self.message),
            title=self.title,
            title_align="left",
            border_style="red",
        )


def _coerce_types(val: Any) -> Any:
    """
    Recursively convert string numbers to floats.

    Cyclopts does not support nested dictionaries or Any types in its CLI arguments,
    and all values are passed as strings. This function attempts to convert any string
    that can be converted to a float, while leaving non-numeric strings unchanged.

    Additionally, it tries to parse strings that look like JSON objects or arrays
    to allow for more complex structures to be passed as command-line arguments.
    """
    if isinstance(val, str):
        try:
            return float(val)  # Try converting to float first
        except ValueError:
            return val  # Keep as string (colors, etc.)
    if isinstance(val, dict):
        return {k: _coerce_types(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_coerce_types(v) for v in val]
    return val


@functools.wraps(_plot_vs_k)
def plot_vs_k(*args, **kwargs):
    """CLI wrapper for plotting limits vs scale, k."""
    try:
        dict_keys = [
            "base_limit_style",
            "limit_styles",
            "base_theory_style",
            "theory_styles",
            "sensitivity_style",
        ]
        for key in dict_keys:
            if kwargs.get(key):
                kwargs[key] = _coerce_types(kwargs[key])
        _plot_vs_k(*args, **kwargs)
    except Exception as e:
        if os.getenv("EOR_LIMITS_DEBUG"):
            raise
        error = CLIError(str(e), title="Error")
        console.print(error.render())
        sys.exit(1)


@functools.wraps(_plot_vs_z)
def plot_vs_z(*args, **kwargs):
    """CLI wrapper for plotting limits vs scale, z."""
    try:
        dict_keys = [
            "base_limit_style",
            "limit_styles",
            "base_theory_style",
            "theory_styles",
            "sensitivity_style",
        ]
        for key in dict_keys:
            if kwargs.get(key):
                kwargs[key] = _coerce_types(kwargs[key])
        _plot_vs_z(*args, **kwargs)
    except Exception as e:
        if os.getenv("EOR_LIMITS_DEBUG"):
            raise
        error = CLIError(str(e), title="Error")
        console.print(error.render())
        sys.exit(1)


app.command(plot_vs_k)
app.command(plot_vs_z)
