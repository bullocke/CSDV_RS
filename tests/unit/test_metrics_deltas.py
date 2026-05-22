"""Tests for inter-NAIP delta metrics."""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import Affine, from_origin

from csdv_core.io.grids import GridSpec
from csdv_core.metrics._result import make_result
from csdv_core.metrics.deltas import (
    delta_crown_count,
    delta_crown_fraction,
    delta_gap_fraction,
    metric_delta,
)
from csdv_core.metrics.gap import gap_fraction


def _chm_and_grid(values: float) -> tuple[np.ndarray, GridSpec]:
    arr = np.full((50, 50), values, dtype=np.float32)
    transform = from_origin(0.0, 25.0, 0.5, 0.5)
    grid = GridSpec(transform=transform, crs="EPSG:5070", pixel_size_m=0.5)
    return arr, grid


def test_metric_delta_identical_grids_subtracts():
    a_chm, grid = _chm_and_grid(0.0)  # all gap
    b_chm, _ = _chm_and_grid(15.0)  # all canopy
    a = gap_fraction(a_chm, grid, window_m=12.5)
    b = gap_fraction(b_chm, grid, window_m=12.5)
    d = metric_delta(a, b)
    assert d.name == "delta_gap_fraction"
    np.testing.assert_allclose(d.array, np.ones_like(d.array))
    assert d.units == "fraction"
    assert d.transform == a.transform
    assert d.params["source"] == "gap_fraction"


def test_metric_delta_shape_mismatch_raises():
    arr = np.zeros((4, 4), dtype=np.float32)
    aff = from_origin(0.0, 4.0, 1.0, 1.0)
    a = make_result("m", arr, transform=aff, crs="EPSG:5070", window_m=1.0)
    b = make_result(
        "m",
        np.zeros((3, 3), dtype=np.float32),
        transform=aff,
        crs="EPSG:5070",
        window_m=1.0,
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        metric_delta(a, b)


def test_metric_delta_transform_mismatch_raises():
    arr = np.zeros((4, 4), dtype=np.float32)
    aff_a = from_origin(0.0, 4.0, 1.0, 1.0)
    aff_b = from_origin(10.0, 4.0, 1.0, 1.0)
    a = make_result("m", arr, transform=aff_a, crs="EPSG:5070", window_m=1.0)
    b = make_result("m", arr, transform=aff_b, crs="EPSG:5070", window_m=1.0)
    with pytest.raises(ValueError, match="transform"):
        metric_delta(a, b)


def test_metric_delta_window_mismatch_raises():
    arr = np.zeros((4, 4), dtype=np.float32)
    aff = Affine(1.0, 0, 0, 0, -1.0, 4.0)
    a = make_result("m", arr, transform=aff, crs="EPSG:5070", window_m=1.0)
    b = make_result("m", arr, transform=aff, crs="EPSG:5070", window_m=2.0)
    with pytest.raises(ValueError, match="window_m"):
        metric_delta(a, b)


def test_metric_delta_nan_propagates():
    aff = from_origin(0.0, 2.0, 1.0, 1.0)
    arr_a = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    arr_b = np.array([[1.0, 0.0], [np.nan, 1.0]], dtype=np.float32)
    a = make_result("x", arr_a, transform=aff, crs="EPSG:5070", window_m=1.0)
    b = make_result("x", arr_b, transform=aff, crs="EPSG:5070", window_m=1.0)
    d = metric_delta(a, b)
    assert np.isnan(d.array[0, 1])
    assert np.isnan(d.array[1, 0])
    assert d.array[0, 0] == pytest.approx(0.0)
    assert d.array[1, 1] == pytest.approx(3.0)


def test_named_delta_rejects_wrong_source():
    aff = from_origin(0.0, 2.0, 1.0, 1.0)
    arr = np.zeros((2, 2), dtype=np.float32)
    a = make_result("gap_fraction", arr, transform=aff, crs="EPSG:5070", window_m=1.0)
    b = make_result("crown_fraction", arr, transform=aff, crs="EPSG:5070", window_m=1.0)
    with pytest.raises(ValueError, match="expected both"):
        delta_gap_fraction(a, b)


def test_named_delta_accepts_correct_source():
    aff = from_origin(0.0, 2.0, 1.0, 1.0)
    arr = np.ones((2, 2), dtype=np.float32)
    a = make_result("crown_fraction", arr, transform=aff, crs="EPSG:5070", window_m=1.0)
    b = make_result(
        "crown_fraction", arr * 0.3, transform=aff, crs="EPSG:5070", window_m=1.0
    )
    d = delta_crown_fraction(a, b)
    assert d.name == "delta_crown_fraction"
    np.testing.assert_allclose(d.array, np.full((2, 2), 0.7), atol=1e-6)


def test_delta_crown_count_units_preserved():
    aff = from_origin(0.0, 2.0, 1.0, 1.0)
    arr = np.ones((2, 2), dtype=np.float32)
    a = make_result(
        "crown_count", arr, transform=aff, crs="EPSG:5070", window_m=1.0, units=""
    )
    b = make_result(
        "crown_count",
        arr * 0.0,
        transform=aff,
        crs="EPSG:5070",
        window_m=1.0,
        units="",
    )
    d = delta_crown_count(a, b)
    assert d.units == ""
