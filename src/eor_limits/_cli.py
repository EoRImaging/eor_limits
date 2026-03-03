#! /usr/bin/env python
# Copyright (c) 2019 Nichole Barry, Bryna Hazelton
# Licensed under the 2-clause BSD License
"""CLI entry points for eor_limits."""

import functools
import logging
import os
import sys

from cyclopts import App, Parameter
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
    help_format="markdown",
    # show_default=True shows [default: None] for all None-typed params
    # negative="" remove the --empty options that cyclopts adds for None-typed params
    default_parameter=Parameter(show_default=True, negative=""),
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


@functools.wraps(_plot_vs_k)
def plot_vs_k(*args, **kwargs):
    """CLI wrapper for plotting limits vs scale, k."""
    try:
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
        _plot_vs_z(*args, **kwargs)
    except Exception as e:
        if os.getenv("EOR_LIMITS_DEBUG"):
            raise
        error = CLIError(str(e), title="Error")
        console.print(error.render())
        sys.exit(1)


app.command(plot_vs_k)
app.command(plot_vs_z)
