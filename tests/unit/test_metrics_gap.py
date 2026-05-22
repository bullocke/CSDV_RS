"""Tests for gap_fraction and crown_fraction."""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import from_origin

from csdv_core.io.grids import GridSpec
from csdv_core.metrics.gap import crown_fraction, gap_fraction, gap_persistence


@pytest.fixture()
def chm_gappy() -> tuple[np.ndarray, GridSpec]:
    """50x50 CHM at 0.5 m pixels (25 m side); top-left 25x25 is gap (0 m),
    rest is 15 m. Window 12.5 m -> 2x2 grid: cell (0,0) all gap, others all canopy."""
    arr = np.full((50, 50), 15.0, dtype=np.float32)
    arr[:25, :25] = 0.0
    transform = from_origin(0.0, 25.0, 0.5, 0.5)
    grid = GridSpec(transform=transform, crs="EPSG:5070", pixel_size_m=0.5)
    return arr, grid


def test_gap_fraction_known_values(chm_gappy):
    arr, grid = chm_gappy
    res = gap_fraction(arr, grid, window_m=12.5, height_threshold_m=2.0)
    assert res.name == "gap_fraction"
    assert res.array.shape == (2, 2)
    assert res.units == "fraction"
    # Top-left cell is all gap (height 0 < 2).
    assert res.array[0, 0] == pytest.approx(1.0)
    # Other cells are all canopy (15 >= 2).
    assert res.array[0, 1] == pytest.approx(0.0)
    assert res.array[1, 0] == pytest.approx(0.0)
    assert res.array[1, 1] == pytest.approx(0.0)


def test_crown_fraction_is_complement(chm_gappy):
    arr, grid = chm_gappy
    gf = gap_fraction(arr, grid, window_m=12.5)
    cf = crown_fraction(arr, grid, window_m=12.5)
    np.testing.assert_allclose(cf.array, 1.0 - gf.array, equal_nan=True)


def test_gap_fraction_nan_window(chm_gappy):
    arr, grid = chm_gappy
    arr = arr.copy()
    arr[:25, 25:50] = np.nan
    res = gap_fraction(arr, grid, window_m=25.0)  # 1x1 grid
    # window_m=25 over 50x50 at 0.5m pixel -> 50px windows -> 1x1 output.
    assert res.array.shape == (1, 1)
    # 1/4 of pixels are gap, 1/4 are NaN, 1/2 are canopy -> gap fraction over valid = 1/3.
    assert res.array[0, 0] == pytest.approx(1 / 3)


def test_output_array_is_readonly(chm_gappy):
    arr, grid = chm_gappy
    res = gap_fraction(arr, grid, window_m=12.5)
    with pytest.raises(ValueError):
        res.array[0, 0] = 0.5


def test_gap_persistence_known_values(chm_gappy):
    arr, grid = chm_gappy
    # t1: gappy fixture. t2: only top-left 12.5 m square is gap; rest canopy.
    t1 = arr
    t2 = np.full_like(arr, 15.0)
    t2[:25, :25] = 0.0  # same gap region
    # Both gaps overlap in upper-left 25x25 (cell (0,0) of 12.5 m window grid).
    res = gap_persistence(t1, t2, grid, window_m=12.5, height_threshold_m=2.0)
    assert res.name == "gap_persistence"
    assert res.units == "fraction"
    assert res.array.shape == (2, 2)
    assert res.array[0, 0] == pytest.approx(1.0)
    assert res.array[0, 1] == pytest.approx(0.0)
    assert res.array[1, 0] == pytest.approx(0.0)
    assert res.array[1, 1] == pytest.approx(0.0)


def test_gap_persistence_partial_overlap(chm_gappy):
    arr, grid = chm_gappy
    t1 = arr  # top-left 25x25 is gap
    t2 = np.full_like(arr, 15.0)
    t2[:25, :50] = 0.0  # top half is gap at t2
    # cell (0,0): all 25x25 pixels gap at both -> 1.0
    # cell (0,1): t1 canopy, t2 gap -> 0.0
    res = gap_persistence(t1, t2, grid, window_m=12.5)
    assert res.array[0, 0] == pytest.approx(1.0)
    assert res.array[0, 1] == pytest.approx(0.0)


def test_gap_persistence_shape_mismatch_raises(chm_gappy):
    arr, grid = chm_gappy
    other = np.zeros((40, 40), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        gap_persistence(arr, other, grid, window_m=12.5)


def test_gap_persistence_nan_window(chm_gappy):
    arr, grid = chm_gappy
    t1 = arr.copy()
    t2 = arr.copy()
    t1[:25, :25] = np.nan
    res = gap_persistence(t1, t2, grid, window_m=12.5)
    assert np.isnan(res.array[0, 0])
