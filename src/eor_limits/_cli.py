#! /usr/bin/env python
# Copyright (c) 2019 Nichole Barry, Bryna Hazelton
# Licensed under the 2-clause BSD License
"""CLI entry points for eor_limits."""

import functools
import logging
from pathlib import Path

from cyclopts import App
from rich.logging import RichHandler

from ._plot import make_plot as _make_plot

app = App()
logger = logging.getLogger("eor_limits")

@functools.wraps(_make_plot)
def make_plot(**kwargs):
    """CLI wrapper for plotting EoR limits."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    _make_plot(**kwargs)
    
app.command(make_plot)

