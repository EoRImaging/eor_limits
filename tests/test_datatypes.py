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
    z_neighbour = min(
        dataset.data.z[4], dataset.data.z[6], key=lambda z: abs(z - z_val)
    )
    offset = (z_neighbour - z_val) / 4  # guaranteed closer to z_val than z_neighbour
    selected = dataset.select_closest_z(z_val + offset)
    assert isinstance(selected, DataSet)
    assert len(selected.data.z) == 1
    assert selected.data.z[0] == z_val


def test_select_closest_k():
    """Test selecting the closest k value from a dataset."""
    dataset = DataSet.load("HERA2026")
    k_val = dataset.data.k[0][5]
    k_neighbour = min(
        dataset.data.k[0][4], dataset.data.k[0][6], key=lambda k: abs(k - k_val)
    )
    offset = (k_neighbour - k_val) / 4  # guaranteed closer to k_val than k_neighbour
    selected = dataset.select_closest_k(k_val + offset)
    assert isinstance(selected, DataSet)
    assert len(selected.data.k[0]) == 1
    assert selected.data.k[0][0] == k_val


def test_select_lowest_delta_squared_per_z():
    """Test selecting the lowest delta_squared value per z (default behaviour)."""
    dataset = DataSet.load("HERA2023")
    selected = dataset.select_lowest_delta_squared()
    assert isinstance(selected, DataSet)
    assert len(selected.data.z) == len(set(dataset.data.z))
    for z_val in set(dataset.data.z):
        idx = [i for i, z in enumerate(dataset.data.z) if z == z_val]
        best_dsq = np.nanmin([np.nanmin(dataset.data.delta_squared[i]) for i in idx])
        selected_idx = list(selected.data.z).index(z_val)
        selected_dsq = selected.data.delta_squared[selected_idx][0]
        assert selected_dsq == best_dsq


def test_select_lowest_delta_squared_per_z_per_tag():
    """Test selecting the lowest delta_squared value per z and per tag."""
    dataset = DataSet.load("HERA2023")
    selected = dataset.select_lowest_delta_squared(per_z=True, per_tag=True)
    assert isinstance(selected, DataSet)
    assert len(selected.data.z) == len(dataset.data.z)
    for dataset_dsq, selected_dsq in zip(
        dataset.data.delta_squared, selected.data.delta_squared, strict=True
    ):
        assert len(selected_dsq) == 1
        assert selected_dsq[0] == np.nanmin(dataset_dsq)


def test_select_lowest_delta_squared_not_per_z_not_per_tag():
    """Test selecting the single globally lowest delta_squared value."""
    dataset = DataSet.load("HERA2023")
    selected = dataset.select_lowest_delta_squared(per_z=False, per_tag=False)
    assert isinstance(selected, DataSet)
    assert len(selected.data.z) == 1
    global_min = np.nanmin([np.nanmin(dsq) for dsq in dataset.data.delta_squared])
    selected_dsq = selected.data.delta_squared[0][0]
    assert selected_dsq == global_min


def test_select_lowest_delta_squared_not_per_z_per_tag():
    """Test selecting the lowest delta_squared value per tag across all z."""
    dataset = DataSet.load("HERA2023")
    selected = dataset.select_lowest_delta_squared(per_z=False, per_tag=True)
    assert isinstance(selected, DataSet)
    assert len(selected.data.z) == len(set(dataset.data.z_tags))
    for tag in set(dataset.data.z_tags):
        idx = [i for i, t in enumerate(dataset.data.z_tags) if t == tag]
        best_dsq = np.nanmin([np.nanmin(dataset.data.delta_squared[i]) for i in idx])
        selected_idx = list(selected.data.z_tags).index(tag)
        selected_dsq = selected.data.delta_squared[selected_idx][0]
        assert selected_dsq == best_dsq
