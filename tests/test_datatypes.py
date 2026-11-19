"""Test the functionality of the datatypes."""

import numpy as np
import pytest

from eor_limits import Data, DataSet


@pytest.fixture
def dataset() -> DataSet:
    """Small synthetic dataset covering what the tests below need.

    Seven redshifts, each with the same seven k bins. The first two entries share
    z = 7.0 with different tags, and delta_squared has one NaN and one inf.
    """
    shape = np.array([50.0, 40.0, 30.0, 20.0, 25.0, 35.0, 45.0])
    delta_squared = [(i + 1) * shape for i in range(7)]
    delta_squared[2][0] = np.nan  # z = 7.5, k = 0.1
    delta_squared[3][6] = np.inf  # z = 8.0, k = 0.8
    data = Data(
        z=[7.0, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5],
        z_tags=("A", "B", "A", "A", "A", "A", "A"),
        k=[[0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8]] * 7,
        delta_squared=tuple(delta_squared),
    )
    return DataSet(telescope="Fake", author="Doe", year=2026, doi="none", data=data)


def test_drop_nan(dataset):
    """Test that drop_nan removes NaN and inf values, along with the matching k."""
    new = dataset.drop_nan().data
    np.testing.assert_array_equal(new.k[2], [0.15, 0.2, 0.3, 0.4, 0.6, 0.8])
    np.testing.assert_array_equal(new.k[3], [0.1, 0.15, 0.2, 0.3, 0.4, 0.6])
    for k, dsq in zip(new.k, new.delta_squared, strict=True):
        assert np.all(np.isfinite(dsq))
        assert len(k) == len(dsq)


def test_delta_is_squared():
    """Test that delta values are converted to delta_squared."""
    data = Data(z=[8.0], k=[[0.1, 0.2]], delta=(np.array([3.0, 4.0]),))
    np.testing.assert_array_equal(data.delta_squared[0], [9.0, 16.0])


def test_delta_squared_must_match_k():
    """Test that delta_squared must have the same shape as k."""
    with pytest.raises(ValueError, match="same shape as k"):
        Data(z=[8.0], k=[[0.1, 0.2]], delta_squared=(np.array([10.0]),))


def test_select_z_range(dataset):
    """Test selecting a range of z values from a dataset."""
    z_val = dataset.data.z[5]
    selected = dataset.select_z_range(z_val - 0.1, z_val + 0.1)
    assert isinstance(selected, DataSet)
    assert np.all(selected.data.z == z_val)


def test_select_k(dataset):
    """Test selecting a range of k values from a dataset."""
    k_val = dataset.data.k[0][5]
    kmin = k_val - 0.25
    kmax = k_val + 0.25
    selected = dataset.select_k_range(kmin, kmax)
    assert isinstance(selected, DataSet)
    for k_arr in selected.data.k:
        assert np.all(kmin <= k_arr)
        assert np.all(k_arr <= kmax)


def test_select_delta_squared_range(dataset):
    """Test selecting a range of delta_squared values from a dataset."""
    ds_val = dataset.data.delta_squared[0][5]
    dsmin = ds_val / 10
    dsmax = ds_val * 10
    selected = dataset.select_delta_squared_range(dsmin, dsmax)
    assert isinstance(selected, DataSet)
    for ds_arr in selected.data.delta_squared:
        assert np.all(dsmin <= ds_arr)
        assert np.all(ds_arr <= dsmax)


def test_select_closest_z(dataset):
    """Test selecting the closest z value from a dataset."""
    z_val = dataset.data.z[5]
    z_neighbour = min(
        dataset.data.z[4], dataset.data.z[6], key=lambda z: abs(z - z_val)
    )
    offset = (z_neighbour - z_val) / 4  # guaranteed closer to z_val than z_neighbour
    selected = dataset.select_closest_z(z_val + offset)
    assert isinstance(selected, DataSet)
    assert len(selected.data.z) == 1
    assert selected.data.z[0] == z_val


def test_select_closest_k(dataset):
    """Test selecting the closest k value from a dataset."""
    k_val = dataset.data.k[0][5]
    k_neighbour = min(
        dataset.data.k[0][4], dataset.data.k[0][6], key=lambda k: abs(k - k_val)
    )
    offset = (k_neighbour - k_val) / 4  # guaranteed closer to k_val than k_neighbour
    selected = dataset.select_closest_k(k_val + offset)
    assert isinstance(selected, DataSet)
    assert len(selected.data.k[0]) == 1
    assert selected.data.k[0][0] == k_val


def test_select_lowest_delta_squared_per_z(dataset):
    """Test selecting the lowest delta_squared value per z (default behaviour)."""
    selected = dataset.select_lowest_delta_squared()
    assert isinstance(selected, DataSet)
    assert len(selected.data.z) == len(set(dataset.data.z))
    for z_val in set(dataset.data.z):
        idx = [i for i, z in enumerate(dataset.data.z) if z == z_val]
        best_dsq = np.nanmin([np.nanmin(dataset.data.delta_squared[i]) for i in idx])
        selected_idx = list(selected.data.z).index(z_val)
        selected_dsq = selected.data.delta_squared[selected_idx][0]
        assert selected_dsq == best_dsq


def test_select_lowest_delta_squared_per_z_per_tag(dataset):
    """Test selecting the lowest delta_squared value per z and per tag."""
    selected = dataset.select_lowest_delta_squared(per_z=True, per_tag=True)
    assert isinstance(selected, DataSet)
    assert len(selected.data.z) == len(dataset.data.z)
    for dataset_dsq, selected_dsq in zip(
        dataset.data.delta_squared, selected.data.delta_squared, strict=True
    ):
        assert len(selected_dsq) == 1
        assert selected_dsq[0] == np.nanmin(dataset_dsq)


def test_select_lowest_delta_squared_not_per_z_not_per_tag(dataset):
    """Test selecting the single globally lowest delta_squared value."""
    selected = dataset.select_lowest_delta_squared(per_z=False, per_tag=False)
    assert isinstance(selected, DataSet)
    assert len(selected.data.z) == 1
    global_min = np.nanmin([np.nanmin(dsq) for dsq in dataset.data.delta_squared])
    selected_dsq = selected.data.delta_squared[0][0]
    assert selected_dsq == global_min


def test_select_lowest_delta_squared_not_per_z_per_tag(dataset):
    """Test selecting the lowest delta_squared value per tag across all z."""
    selected = dataset.select_lowest_delta_squared(per_z=False, per_tag=True)
    assert isinstance(selected, DataSet)
    assert len(selected.data.z) == len(set(dataset.data.z_tags))
    for tag in set(dataset.data.z_tags):
        idx = [i for i, t in enumerate(dataset.data.z_tags) if t == tag]
        best_dsq = np.nanmin([np.nanmin(dataset.data.delta_squared[i]) for i in idx])
        selected_idx = list(selected.data.z_tags).index(tag)
        selected_dsq = selected.data.delta_squared[selected_idx][0]
        assert selected_dsq == best_dsq
