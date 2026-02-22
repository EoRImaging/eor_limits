#! /usr/bin/env python
# Copyright (c) 2019 Nichole Barry, Bryna Hazelton
# Licensed under the 2-clause BSD License
"""CLI entry points for eor_limits."""

import functools
import logging

from cyclopts import App
from rich.logging import RichHandler

from ._plot import plot_vs_k as _plot_vs_k

app = App()
logger = logging.getLogger("eor_limits")


@functools.wraps(_plot_vs_k)
def plot_vs_k(**kwargs):
    """CLI wrapper for plotting EoR limits."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    _plot_vs_k(**kwargs)


app.command(plot_vs_k)
