"""Tests of the theory data processing."""

import pytest

from eor_limits.data import ALL_THEORIES, get_theory_data


@pytest.mark.parametrize("theory_name", ALL_THEORIES)
def test_load_theory(theory_name):
    """Test loading each theory dataset."""
    dataset = get_theory_data(theory_name)
    assert dataset.telescope is not None
    assert dataset.author is not None
    assert dataset.year is not None
    assert dataset.doi is not None
    assert dataset.data is not None
