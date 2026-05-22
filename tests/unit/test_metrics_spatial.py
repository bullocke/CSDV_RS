"""Tests for spatial pattern metrics."""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import from_origin

from csdv_core.io.grids import GridSpec
from csdv_core.metrics.spatial import (
    edge_density,
    linearity_index,
    row_directionality,
)


def _grid(rows: int, cols: int, px: float = 0.5) -> GridSpec:
    transform = from_origin(0.0, rows * px, px, px)
    return GridSpec(transform=transform, crs="EPSG:5070", pixel_size_m=px)


def test_linearity_striped_vs_random():
    rng = np.random.default_rng(0)
    # 100x100 striped gap mask: every 5 rows is a stripe of gap.
    striped = np.zeros((100, 100), dtype=bool)
    striped[::5, :] = True
    random = rng.random((100, 100)) < 0.2
    grid = _grid(100, 100, px=0.5)
    res_s = linearity_index(striped, grid, window_m=50.0)
    res_r = linearity_index(random, grid, window_m=50.0)
    assert res_s.array.shape == (1, 1)
    assert res_s.array[0, 0] > res_r.array[0, 0]
    assert 0.0 <= res_r.array[0, 0] <= 1.0
    assert res_s.units == ""


def test_edge_density_solid_vs_checkerboard():
    solid = np.ones((100, 100), dtype=bool)
    checker = np.indices((100, 100)).sum(axis=0) % 2 == 0
    grid = _grid(100, 100, px=0.5)
    res_solid = edge_density(solid, grid, window_m=50.0)
    res_check = edge_density(checker, grid, window_m=50.0)
    assert res_solid.units == "1/m"
    assert res_solid.array.shape == (1, 1)
    assert res_solid.array[0, 0] == pytest.approx(0.0)
    assert res_check.array[0, 0] > 0.5  # dense edges per square meter


def test_row_directionality_striped_vs_random():
    rng = np.random.default_rng(1)
    striped = np.tile(np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32), (100, 20))
    random = rng.random((100, 100)).astype(np.float32)
    grid = _grid(100, 100, px=0.5)
    res_s = row_directionality(striped, grid, window_m=50.0)
    res_r = row_directionality(random, grid, window_m=50.0)
    assert res_s.array[0, 0] > res_r.array[0, 0]
    assert 0.0 <= res_r.array[0, 0] <= 1.0


def test_row_directionality_nan_propagation():
    arr = np.full((100, 100), np.nan, dtype=np.float32)
    grid = _grid(100, 100, px=0.5)
    res = row_directionality(arr, grid, window_m=50.0)
    assert np.isnan(res.array[0, 0])
