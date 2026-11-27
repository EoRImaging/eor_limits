"""Pytest fixtures for ps21cm_limits tests."""

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def close_matplotlib_figures():
    """Close figures opened by each test."""
    yield
    plt.close("all")
