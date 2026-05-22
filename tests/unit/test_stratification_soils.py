"""Tests for :mod:`csdv_core.stratification.soils`."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from csdv_core.io.grids import GridSpec
from csdv_core.stratification import soils


def _grid(n: int = 4) -> GridSpec:
    return GridSpec(
        transform=Affine.translation(0, 0) * Affine.scale(1, -1),
        crs="EPSG:5070",
        pixel_size_m=1.0,
    )


def test_stack_soil_variables_fills_missing() -> None:
    arr = np.zeros((4, 4), dtype="float32")
    out = soils.stack_soil_variables({"hydric_pct": arr}, _grid())
    assert set(out.keys()) == set(soils.REQUIRED_SOIL_VARS)
    assert out["hydric_pct"].dtype == np.float32
    assert out["parmat_kind"].dtype == np.uint8
    # Missing continuous filled with NaN.
    assert np.all(np.isnan(out["awc_total_cm"]))
    # Missing categorical filled with 0 (unknown).
    assert (out["parmat_kind"] == 0).all()


def test_stack_soil_variables_dtype_normalized() -> None:
    arr_int = np.ones((3, 3), dtype="int32")
    arr_float = np.ones((3, 3), dtype="float64")
    out = soils.stack_soil_variables(
        {"parmat_kind": arr_int, "awc_total_cm": arr_float},
        _grid(3),
    )
    assert out["parmat_kind"].dtype == np.uint8
    assert out["awc_total_cm"].dtype == np.float32


def test_stack_soil_variables_shape_mismatch_raises() -> None:
    a = np.zeros((4, 4), dtype="float32")
    b = np.zeros((3, 3), dtype="float32")
    with pytest.raises(ValueError, match="shape"):
        soils.stack_soil_variables({"hydric_pct": a, "awc_total_cm": b}, _grid())


def test_encode_categorical_unknown_to_zero() -> None:
    labels = np.array([["sand", "loam"], ["bogus", "clay"]])
    out = soils.encode_categorical(labels, soils.TEXTURE_CLASS)
    assert out.shape == (2, 2)
    assert out[0, 0] == soils.TEXTURE_CLASS["sand"]
    assert out[0, 1] == soils.TEXTURE_CLASS["loam"]
    assert out[1, 0] == 0  # unknown
    assert out[1, 1] == soils.TEXTURE_CLASS["clay"]
    assert out.dtype == np.uint8
