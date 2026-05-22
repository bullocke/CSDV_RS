"""Tests for :mod:`csdv_core.stratification.topo`."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("richdem")

from csdv_core.stratification import topo  # noqa: E402


def _hill(n: int = 60, sigma: float = 12.0) -> np.ndarray:
    yy, xx = np.mgrid[:n, :n]
    cx = cy = (n - 1) / 2.0
    z = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    return (z * 100.0).astype("float32")


def test_compute_slope_aspect_shape_and_range() -> None:
    dem = _hill()
    slope, aspect, north, east = topo.compute_slope_aspect(dem, pixel_size_m=1.0)
    assert slope.shape == dem.shape
    assert np.all(slope >= 0)
    assert np.all((north >= -1) & (north <= 1))
    assert np.all((east >= -1) & (east <= 1))


def test_compute_slope_monotone_off_peak() -> None:
    dem = _hill(60, sigma=10.0)
    slope, _, _, _ = topo.compute_slope_aspect(dem, pixel_size_m=1.0)
    cx = cy = (dem.shape[0] - 1) // 2
    # Slope at peak should be near zero; flank should be larger.
    assert slope[cy, cx] < slope[cy, cx + 8]


def test_compute_topo_returns_all_keys() -> None:
    dem = _hill()
    out = topo.compute_topo(dem, pixel_size_m=1.0, accumulation_threshold=200.0)
    assert set(out.keys()) == {
        "slope_deg",
        "aspect_deg",
        "northness",
        "eastness",
        "twi",
        "hand",
        "rei",
        "profile_curvature",
    }
    for k, arr in out.items():
        assert arr.shape == dem.shape, k


def test_compute_topo_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        topo.compute_topo(np.zeros(5), pixel_size_m=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        topo.compute_topo(np.zeros((5, 5), dtype="float32"), pixel_size_m=0.0)
