"""Test the functionality of the datatypes."""

import numpy as np
import pytest

from eor_limits import KNOWN_LIMITS, DataSet


@pytest.mark.parametrize("paper_name", list(KNOWN_LIMITS.keys()))
def test_load_all_datasets(paper_name):
    """Test loading all datasets."""
    dataset = DataSet.load(paper_name)
    assert isinstance(dataset, DataSet)


def test_drop_nan():
    """Test dropping NaN values from a dataset."""
    dataset = DataSet.load("Barry2019")  # known to have NaN values
    new = dataset.drop_nan()
    assert isinstance(new, DataSet)
    assert np.all(np.isfinite(new.data.delta_squared))


def test_select_z_range():
    """Test selecting a range of z values from a dataset."""
    dataset = DataSet.load("HERA2026")
    z_val = dataset.data.z[5]
    selected = dataset.select_z_range(z_val - 0.1, z_val + 0.1)
    assert isinstance(selected, DataSet)
    assert np.all(selected.data.z == z_val)


def test_select_k():
    """Test selecting a range of k values from a dataset."""
    dataset = DataSet.load("HERA2026")
    k_val = dataset.data.k[0][5]
    kmin = k_val - 0.25
    kmax = k_val + 0.25
    selected = dataset.select_k_range(kmin, kmax)
    assert isinstance(selected, DataSet)
    for k_arr in selected.data.k:
        assert np.all(kmin <= k_arr)
        assert np.all(k_arr <= kmax)


def test_select_delta_squared_range():
    """Test selecting a range of delta_squared values from a dataset."""
    dataset = DataSet.load("HERA2026")
    ds_val = dataset.data.delta_squared[0][5]
    dsmin = ds_val / 10
    dsmax = ds_val * 10
    selected = dataset.select_delta_squared_range(dsmin, dsmax)
    assert isinstance(selected, DataSet)

    for ds_arr in selected.data.delta_squared:
        assert np.all(dsmin <= ds_arr)
        assert np.all(ds_arr <= dsmax)


def test_select_closest_z():
    """Test selecting the closest z value from a dataset."""
    dataset = DataSet.load("HERA2026")
    z_val = dataset.data.z[5]
    selected = dataset.select_closest_z(z_val)
    assert isinstance(selected, DataSet)
    assert len(selected.data.z) == 1
    assert selected.data.z[0] == z_val


def test_select_closest_k():
    """Test selecting the closest k value from a dataset."""
    dataset = DataSet.load("HERA2026")
    k_val = dataset.data.k[0][5]
    selected = dataset.select_closest_k(k_val)
    assert isinstance(selected, DataSet)
    assert len(selected.data.k[0]) == 1
    assert selected.data.k[0][0] == k_val


def test_select_lowest_delta_squared():
    """Test selecting the lowest delta_squared value from a dataset."""
    dataset = DataSet.load("HERA2026")
    selected = dataset.select_lowest_delta_squared()
    assert isinstance(selected, DataSet)
    for ds_arr in selected.data.delta_squared:
        assert len(ds_arr) == 1
        assert ds_arr[0] == np.nanmin(ds_arr)
