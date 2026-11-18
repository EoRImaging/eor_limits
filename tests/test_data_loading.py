"""Tests of the dataset lookup shared by all the loaders, and of the bundled files."""

from pathlib import Path

import pytest

from eor_limits import KNOWN_LIMITS, KNOWN_SENSITIVITIES, KNOWN_THEORIES, DataSet
from eor_limits._data_loading import _resolve_dataset_path

KNOWN = {"Known2026": Path("/registry/Known2026.yaml")}
BUNDLED = {**KNOWN_LIMITS, **KNOWN_THEORIES, **KNOWN_SENSITIVITIES}


def test_known_name_returns_registered_path():
    """Test that a known name is looked up in the registry."""
    assert _resolve_dataset_path("Known2026", KNOWN) == KNOWN["Known2026"]


@pytest.mark.parametrize("suffix", [".yaml", ".yml"])
@pytest.mark.parametrize("as_type", [str, Path])
def test_path_to_yaml_file(tmp_path, suffix, as_type):
    """Test that a path to a YAML file, as a str or a Path, is returned as a Path."""
    path = tmp_path / f"dataset{suffix}"
    path.touch()
    assert _resolve_dataset_path(as_type(path), KNOWN) == path


@pytest.mark.parametrize("name", ["Unknown2026", "no/such/dir/missing.yaml"])
def test_unknown_name_or_missing_file(name):
    """Test that a name that is neither known nor a file raises an error."""
    with pytest.raises(FileNotFoundError, match="neither a known name"):
        _resolve_dataset_path(name, KNOWN)


@pytest.mark.parametrize("suffix", [".h5", ".txt"])
def test_rejects_non_yaml_file(tmp_path, suffix):
    """Test that files that are not YAML are rejected."""
    path = tmp_path / f"dataset{suffix}"
    path.touch()
    with pytest.raises(ValueError, match="YAML"):
        _resolve_dataset_path(path, KNOWN)


@pytest.mark.parametrize("name", sorted(BUNDLED))
def test_bundled_key_matches_filename(name):
    """Test that each bundled file's key matches the name users refer to it by."""
    assert DataSet.load(BUNDLED[name]).key == name
