"""Tests of the sensitivity estimates and their loader."""

import numpy as np
import pytest

from eor_limits import KNOWN_SENSITIVITIES, load_sensitivity_data


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


@pytest.mark.parametrize("k_unit_line", ["", "k_unit: 1/Mpc", "k_unit: Mpc^-1"])
def test_load_sensitivity_data_requires_k_unit(tmp_path, k_unit_line):
    """Test that a file whose k_unit entry is missing or not h/Mpc is rejected."""
    path = tmp_path / "sensitivity.yaml"
    path.write_text(
        KNOWN_SENSITIVITIES["Breitman2026_AA4_deep1080h"]
        .read_text()
        .replace("k_unit: h/Mpc", k_unit_line)
    )
    with pytest.raises(ValueError, match="k_unit: h/Mpc"):
        load_sensitivity_data(path)
