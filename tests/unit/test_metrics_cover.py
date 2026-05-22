"""Tests for CHM-based cover height-band fractions."""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import from_origin

from csdv_core.io.grids import GridSpec
from csdv_core.metrics.cover import (
    mid_canopy_fraction,
    shrub_fraction,
    small_tree_fraction,
    tall_canopy_fraction,
)


@pytest.fixture()
def chm_banded() -> tuple[np.ndarray, GridSpec]:
    """80x80 CHM at 0.5 m pixels (40 m side); four 40x40 quadrants with one
    height per band: gap (0), shrub (1.0), small tree (5.0), tall (25.0)."""
    arr = np.zeros((80, 80), dtype=np.float32)
    arr[:40, :40] = 0.0  # gap
    arr[:40, 40:] = 1.0  # shrub band [0.5, 2.0)
    arr[40:, :40] = 5.0  # small tree [2, 10)
    arr[40:, 40:] = 25.0  # tall canopy >=20
    transform = from_origin(0.0, 40.0, 0.5, 0.5)
    grid = GridSpec(transform=transform, crs="EPSG:5070", pixel_size_m=0.5)
    return arr, grid


def test_shrub_fraction_per_quadrant(chm_banded):
    arr, grid = chm_banded
    # window_m=20 -> 40 px per side -> 2x2 grid aligned to quadrants.
    res = shrub_fraction(arr, grid, window_m=20.0)
    assert res.name == "shrub_fraction"
    assert res.units == "fraction"
    assert res.array.shape == (2, 2)
    expected = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(res.array, expected)


def test_small_tree_fraction_per_quadrant(chm_banded):
    arr, grid = chm_banded
    res = small_tree_fraction(arr, grid, window_m=20.0)
    expected = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(res.array, expected)


def test_mid_canopy_fraction_zero_everywhere(chm_banded):
    arr, grid = chm_banded
    res = mid_canopy_fraction(arr, grid, window_m=20.0)
    np.testing.assert_allclose(res.array, np.zeros((2, 2), dtype=np.float32))


def test_tall_canopy_fraction_per_quadrant(chm_banded):
    arr, grid = chm_banded
    res = tall_canopy_fraction(arr, grid, window_m=20.0)
    expected = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(res.array, expected)


def test_nan_window_returns_nan(chm_banded):
    arr, grid = chm_banded
    arr = arr.copy()
    arr[:40, :40] = np.nan  # nuke gap quadrant
    res = shrub_fraction(arr, grid, window_m=20.0)
    assert np.isnan(res.array[0, 0])
    assert res.array[0, 1] == pytest.approx(1.0)


def test_nodata_sentinel_masked(chm_banded):
    arr, grid = chm_banded
    arr = arr.copy()
    arr[:40, :40] = -9999.0
    res = shrub_fraction(arr, grid, window_m=20.0, nodata=-9999.0)
    assert np.isnan(res.array[0, 0])
