"""Tests of the data-loading functions."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from cattrs.errors import ClassValidationError

import eor_limits
from eor_limits import KNOWN_SENSITIVITIES, DataSet, load_sensitivity_data

# SKA-Low AA4 sensitivity distributed with the package
SENSITIVITY_NAME = "Breitman2026_AA4_deep1080h"
SENSITIVITY_FILE = KNOWN_SENSITIVITIES[SENSITIVITY_NAME]


@pytest.fixture
def sensitivity_yaml() -> dict:
    """Raw content of the SKA-Low AA4 sensitivity file."""
    with SENSITIVITY_FILE.open("r") as fl:
        return yaml.safe_load(fl)


def _write_yaml(path: Path, content: dict) -> Path:
    """Write ``content`` to ``path`` as YAML and return the path."""
    with path.open("w") as fl:
        yaml.safe_dump(content, fl)
    return path


def test_load_sensitivity_data_is_exported():
    """Test that the loader and the known sensitivities are part of the public API."""
    assert "load_sensitivity_data" in eor_limits.__all__
    assert "KNOWN_SENSITIVITIES" in eor_limits.__all__


@pytest.mark.parametrize("name", list(KNOWN_SENSITIVITIES))
def test_load_known_sensitivities(name):
    """Test loading each sensitivity estimate distributed with the package."""
    sensitivity = load_sensitivity_data(name)
    assert isinstance(sensitivity, DataSet)
    assert sensitivity.key == name
    for dsq in sensitivity.data.delta_squared:
        finite = dsq[np.isfinite(dsq)]
        assert len(finite) > 0
        assert np.all(finite > 0)


def test_load_sensitivity_data_name_and_path_agree():
    """Test that a known name and the path of its file give the same data."""
    by_name = load_sensitivity_data(SENSITIVITY_NAME)
    by_path = load_sensitivity_data(SENSITIVITY_FILE)
    np.testing.assert_array_equal(by_name.data.z, by_path.data.z)
    for dsq_a, dsq_b in zip(
        by_name.data.delta_squared, by_path.data.delta_squared, strict=True
    ):
        np.testing.assert_array_equal(dsq_a, dsq_b)


def test_ska_aa4_at_least_as_sensitive_as_aastar():
    """Test that AA4 (512 stations) is never less sensitive than AA* (307 stations).

    This guards against the two SKA files being swapped.
    """
    aa4 = load_sensitivity_data("Breitman2026_AA4_deep1080h")
    aastar = load_sensitivity_data("Breitman2026_AAstar_deep1080h")
    np.testing.assert_array_equal(aa4.data.z, aastar.data.z)
    for dsq_aa4, dsq_aastar in zip(
        aa4.data.delta_squared, aastar.data.delta_squared, strict=True
    ):
        finite = np.isfinite(dsq_aa4)
        np.testing.assert_array_equal(finite, np.isfinite(dsq_aastar))
        assert np.all(dsq_aa4[finite] <= dsq_aastar[finite])


@pytest.mark.parametrize("as_type", [str, Path])
def test_load_sensitivity_data_returns_dataset(as_type):
    """Test loading a sensitivity file given as a str or a Path."""
    sensitivity = load_sensitivity_data(as_type(SENSITIVITY_FILE))
    assert isinstance(sensitivity, DataSet)


def test_load_sensitivity_data_metadata(sensitivity_yaml):
    """Test that the metadata in the file is carried into the DataSet."""
    sensitivity = load_sensitivity_data(SENSITIVITY_FILE)
    assert sensitivity.telescope == sensitivity_yaml["telescope"]
    assert sensitivity.author == sensitivity_yaml["author"]
    assert sensitivity.year == sensitivity_yaml["year"]
    assert sensitivity.doi == sensitivity_yaml["doi"]
    assert sensitivity.notes == tuple(sensitivity_yaml["notes"])
    assert sensitivity.key == SENSITIVITY_NAME


def test_load_sensitivity_data_values(sensitivity_yaml):
    """Test that z, k and delta_squared match the file, including inf."""
    sensitivity = load_sensitivity_data(SENSITIVITY_FILE)
    raw = sensitivity_yaml["data"]

    np.testing.assert_array_equal(sensitivity.data.z, raw["z"])
    assert len(sensitivity.data.k) == len(raw["k"])
    for k_loaded, k_raw in zip(sensitivity.data.k, raw["k"], strict=True):
        np.testing.assert_array_equal(k_loaded, k_raw)
    assert len(sensitivity.data.delta_squared) == len(raw["delta_squared"])
    for dsq_loaded, dsq_raw in zip(
        sensitivity.data.delta_squared, raw["delta_squared"], strict=True
    ):
        np.testing.assert_array_equal(dsq_loaded, dsq_raw)


def test_load_sensitivity_data_delta(tmp_path, sensitivity_yaml):
    """Test that the sensitivity can be given as delta instead of delta_squared."""
    raw = sensitivity_yaml["data"]
    raw["delta"] = [np.sqrt(dsq).tolist() for dsq in raw.pop("delta_squared")]
    path = _write_yaml(tmp_path / "sensitivity.yaml", sensitivity_yaml)

    from_delta = load_sensitivity_data(path)
    from_delta_squared = load_sensitivity_data(SENSITIVITY_FILE)
    for dsq_a, dsq_b in zip(
        from_delta.data.delta_squared,
        from_delta_squared.data.delta_squared,
        strict=True,
    ):
        np.testing.assert_allclose(dsq_a, dsq_b)


def test_load_sensitivity_data_drop_nan(sensitivity_yaml):
    """Test that non-finite sensitivities are removed by drop_nan."""
    sensitivity = load_sensitivity_data(SENSITIVITY_FILE).drop_nan()
    for k, dsq, dsq_raw in zip(
        sensitivity.data.k,
        sensitivity.data.delta_squared,
        sensitivity_yaml["data"]["delta_squared"],
        strict=True,
    ):
        assert np.all(np.isfinite(dsq))
        assert len(k) == len(dsq) == np.isfinite(dsq_raw).sum()


@pytest.mark.parametrize("as_type", [str, Path])
def test_load_sensitivity_data_yml_suffix(tmp_path, as_type):
    """Test that files with a .yml suffix are accepted."""
    path = tmp_path / "sensitivity.yml"
    path.write_text(SENSITIVITY_FILE.read_text())
    sensitivity = load_sensitivity_data(as_type(path))
    assert isinstance(sensitivity, DataSet)


def test_load_sensitivity_data_missing_file(tmp_path):
    """Test that a missing file raises an error."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_sensitivity_data(tmp_path / "missing.yaml")


@pytest.mark.parametrize("name", ["HERA2023", "HERA2023.yaml"])
def test_load_sensitivity_data_does_not_use_known_limits(name):
    """Test that names of known limits are not looked up in the package data."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_sensitivity_data(name)


@pytest.mark.parametrize("suffix", [".h5", ".hdf5", ".txt"])
def test_load_sensitivity_data_rejects_non_yaml(tmp_path, suffix):
    """Test that non-YAML files (e.g. 21cmSense HDF5 outputs) are rejected."""
    path = tmp_path / f"sensitivity{suffix}"
    path.write_bytes(b"\x89HDF\r\n\x1a\n")
    with pytest.raises(ValueError, match="YAML"):
        load_sensitivity_data(path)


def test_load_sensitivity_data_requires_k_unit(tmp_path, sensitivity_yaml):
    """Test that a file without the k_unit entry is rejected."""
    del sensitivity_yaml["k_unit"]
    path = _write_yaml(tmp_path / "sensitivity.yaml", sensitivity_yaml)
    with pytest.raises(ValueError, match="k_unit: h/Mpc"):
        load_sensitivity_data(path)


@pytest.mark.parametrize("k_unit", ["1/Mpc", "Mpc^-1"])
def test_load_sensitivity_data_rejects_other_k_units(
    tmp_path, sensitivity_yaml, k_unit
):
    """Test that k units other than h/Mpc are rejected."""
    sensitivity_yaml["k_unit"] = k_unit
    path = _write_yaml(tmp_path / "sensitivity.yaml", sensitivity_yaml)
    with pytest.raises(ValueError, match="k_unit: h/Mpc"):
        load_sensitivity_data(path)


@pytest.mark.parametrize("field", ["telescope", "author", "year", "doi"])
def test_load_sensitivity_data_requires_metadata(tmp_path, sensitivity_yaml, field):
    """Test that the same metadata as for limits is required."""
    del sensitivity_yaml[field]
    path = _write_yaml(tmp_path / "sensitivity.yaml", sensitivity_yaml)
    with pytest.raises(ClassValidationError):
        load_sensitivity_data(path)


def test_load_sensitivity_data_validates_shapes(tmp_path, sensitivity_yaml):
    """Test that the data validation of limits also applies to sensitivities."""
    dsq = sensitivity_yaml["data"]["delta_squared"]
    dsq[0] = dsq[0][:-1]
    path = _write_yaml(tmp_path / "sensitivity.yaml", sensitivity_yaml)
    with pytest.raises(ClassValidationError):
        load_sensitivity_data(path)
