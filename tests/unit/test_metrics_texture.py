"""Tests for csdv_core.metrics.texture.glcm_texture."""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import from_origin

from csdv_core.io.grids import GridSpec
from csdv_core.metrics.texture import glcm_texture


def _grid(size: int) -> GridSpec:
    transform = from_origin(0.0, size * 0.5, 0.5, 0.5)
    return GridSpec(transform=transform, crs="EPSG:5070", pixel_size_m=0.5)


def test_flat_image_has_low_entropy():
    arr = np.full((80, 80), 5.0, dtype=np.float32)
    grid = _grid(80)
    res = glcm_texture(arr, grid, window_m=20.0, prop="entropy")
    # 80px * 0.5m = 40m extent / 20m window -> 2x2. Flat -> entropy = 0.
    assert res.array.shape == (2, 2)
    assert np.nanmax(res.array) == pytest.approx(0.0, abs=1e-6)


def test_checkerboard_has_higher_entropy():
    rng = np.random.default_rng(0)
    arr = rng.uniform(0.0, 30.0, size=(80, 80)).astype(np.float32)
    grid = _grid(80)
    flat = glcm_texture(np.full((80, 80), 5.0, dtype=np.float32), grid, window_m=20.0)
    noisy = glcm_texture(arr, grid, window_m=20.0)
    assert np.nanmean(noisy.array) > np.nanmean(flat.array)


def test_nan_dominant_window_is_nan():
    arr = np.full((80, 80), np.nan, dtype=np.float32)
    arr[0:2, 0:2] = 5.0  # tiny valid patch in cell (0,0); other cells fully NaN.
    grid = _grid(80)
    res = glcm_texture(arr, grid, window_m=20.0)
    assert np.all(np.isnan(res.array))
