"""Unit tests for csdv_core.segmentation.chm_watershed.segment_crowns."""

from __future__ import annotations

import numpy as np
from rasterio.transform import Affine

from csdv_core.segmentation.chm_watershed import segment_crowns


def _gaussian_chm(
    size: int = 100,
    peaks: tuple[tuple[int, int, float], ...] = (
        (20, 20, 15.0),
        (20, 80, 15.0),
        (80, 20, 15.0),
        (80, 80, 15.0),
    ),
    sigma_px: float = 6.0,
) -> np.ndarray:
    """Synthetic CHM with the given Gaussian crowns. Pixel size = 1 m."""
    arr = np.zeros((size, size), dtype="float32")
    y, x = np.indices(arr.shape)
    for r, c, h in peaks:
        arr += h * np.exp(-((y - r) ** 2 + (x - c) ** 2) / (2.0 * sigma_px**2))
    return arr.astype("float32")


def test_segment_crowns_recovers_four_synthetic_peaks() -> None:
    """Four well-separated Gaussian crowns should return ~4 polygons."""
    chm = _gaussian_chm()
    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)  # 1 m pixels
    gdf = segment_crowns(
        chm,
        transform=transform,
        crs="EPSG:5070",
        min_height_m=2.0,
        smooth_kernel=3,
        min_peak_distance_m=5.0,
        min_crown_area_m2=1.0,
    )
    # Allow watershed to merge or split a single crown by ±1.
    assert 3 <= len(gdf) <= 5
    # All crowns above the min area.
    assert (gdf["area_m2"] >= 1.0).all()
    # Reasonable crown diameter (a Gaussian with sigma=6, h=15 truncated at 2 m
    # gives a crown radius ~ sigma * sqrt(2 ln(15/2)) ~= 12 m, diam ~24 m).
    assert (gdf["crown_diam_m"] > 5.0).all()
    assert (gdf["crown_diam_m"] < 50.0).all()
    assert gdf.crs is not None


def test_segment_crowns_empty_when_below_threshold() -> None:
    """A CHM entirely below the height threshold returns an empty GeoDataFrame."""
    chm = np.full((50, 50), 0.5, dtype="float32")
    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    gdf = segment_crowns(
        chm,
        transform=transform,
        crs="EPSG:5070",
        min_height_m=2.0,
    )
    assert len(gdf) == 0
    assert {"segment_id", "area_m2", "crown_diam_m"}.issubset(gdf.columns)
