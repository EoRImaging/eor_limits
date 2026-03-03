"""Pytest configuration and fixtures for eor_limits tests.

This module provides custom pytest hooks to control test execution order,
ensuring that plotting tests run before CLI plotting tests for comparison
of generated plots.
"""


def pytest_collection_modifyitems(items):
    """Ensure test_plotting runs before test_cli_plotting."""
    plotting_tests = [
        i for i in items if "test_plotting" in i.nodeid and "cli" not in i.nodeid
    ]
    cli_tests = [i for i in items if "test_cli_plotting" in i.nodeid]
    other_tests = [i for i in items if i not in plotting_tests and i not in cli_tests]
    items[:] = other_tests + plotting_tests + cli_tests
