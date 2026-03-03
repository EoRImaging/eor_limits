"""Pytest configuration and fixtures for eor_limits tests.

This module provides custom pytest hooks to control test execution order,
ensuring that plotting tests run before CLI plotting tests for comparison
of generated plots.
"""


def pytest_collection_modifyitems(items):
    """Ensure test_plotting runs before test_cli_plotting."""
    lib_plot_tests = [i for i in items if "test_lib_plotting" in i.nodeid]
    cli_plot_tests = [i for i in items if "test_cli_plotting" in i.nodeid]
    other_tests = [
        i for i in items if i not in lib_plot_tests and i not in cli_plot_tests
    ]
    items[:] = other_tests + lib_plot_tests + cli_plot_tests
