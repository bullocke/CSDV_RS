"""Unit tests for csdv_core.preprocess.chm."""

from __future__ import annotations

import numpy as np
import pytest

from csdv_core.preprocess.chm import (
    convert_uint16_to_meters,
    mask_below,
    smooth_chm,
)


def test_smooth_chm_constant_input_preserved() -> None:
    """A constant array smooths to itself."""
    arr = np.full((10, 10), 5.0, dtype="float32")
    out = smooth_chm(arr, kernel=3)
    assert out.shape == arr.shape
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, 5.0, atol=1e-6)


def test_smooth_chm_ignores_nan() -> None:
    """NaN neighbours are excluded from the local mean."""
    arr = np.full((5, 5), 4.0, dtype="float32")
    arr[2, 2] = np.nan
    out = smooth_chm(arr, kernel=3)
    # Centre value should be 4.0 because the 8 non-NaN neighbours all equal 4.
    assert np.isclose(out[2, 2], 4.0)
    # No NaN should propagate from a single missing pixel.
    assert not np.isnan(out).any()


def test_smooth_chm_rejects_even_kernel() -> None:
    """Even kernels are rejected."""
    arr = np.zeros((4, 4), dtype="float32")
    with pytest.raises(ValueError):
        smooth_chm(arr, kernel=2)


def test_mask_below_threshold() -> None:
    """Pixels strictly below threshold become NaN."""
    arr = np.array([[0.5, 1.9], [2.0, 3.5]], dtype="float32")
    out = mask_below(arr, threshold_m=2.0)
    assert np.isnan(out[0, 0])
    assert np.isnan(out[0, 1])
    assert out[1, 0] == pytest.approx(2.0)
    assert out[1, 1] == pytest.approx(3.5)
    # Original array is not mutated.
    assert arr[0, 0] == pytest.approx(0.5)


def test_convert_uint16_to_meters_scale() -> None:
    """Scale factor 0.01 converts uint16 counts to metres."""
    u16 = np.array([[0, 100, 1500]], dtype="uint16")
    out = convert_uint16_to_meters(u16, scale=0.01)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [[0.0, 1.0, 15.0]], atol=1e-5)


def test_convert_uint16_to_meters_nodata() -> None:
    """src_nodata pixels are replaced with dst_nodata."""
    u16 = np.array([[0, 65535, 200]], dtype="uint16")
    out = convert_uint16_to_meters(
        u16, scale=0.01, src_nodata=65535, dst_nodata=-9999.0
    )
    assert out[0, 0] == pytest.approx(0.0)
    assert out[0, 1] == pytest.approx(-9999.0)
    assert out[0, 2] == pytest.approx(2.0)
