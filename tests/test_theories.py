"""Tests of the theory data processing."""

import pytest

from eor_limits import KNOWN_THEORIES, load_theory_model


@pytest.mark.parametrize("theory_name", KNOWN_THEORIES)
def test_load_theory(theory_name):
    """Test loading each theory dataset."""
    dataset = load_theory_model(theory_name)
    assert dataset.telescope is not None
    assert dataset.author is not None
    assert dataset.year is not None
    assert dataset.doi is not None
    assert dataset.data is not None


@pytest.mark.parametrize("theory_name", KNOWN_THEORIES)
def test_theory_key_matches_filename(theory_name):
    """Each theory is a YAML file whose stored key matches the filename."""
    path = KNOWN_THEORIES[theory_name]
    assert path.suffix == ".yaml"
    assert path.stem == theory_name
    assert load_theory_model(theory_name).key == theory_name


def test_load_unknown_theory():
    """Requesting a theory that does not exist raises a helpful error."""
    with pytest.raises(ValueError, match="Theory 'NotATheory' not found"):
        load_theory_model("NotATheory")
