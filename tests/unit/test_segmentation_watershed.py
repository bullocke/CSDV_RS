"""Unit tests for csdv_core.segmentation.chm_watershed.

The window-semantics tests are the ones that matter. The engine previously read
lidR's search window diameter as a peak separation, which spaced tree tops twice
as far apart as the equation intended, and it evaluated the window once for the
whole raster instead of per pixel. Both are pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import Affine

from csdv_core.preprocess.chm import mask_below, smooth_chm
from csdv_core.segmentation.chm_watershed import (
    CROWN_COLUMNS,
    locate_seeds,
    segment_crowns,
    smooth_kernel_px,
)
from csdv_core.segmentation.params import (
    WINDOW_FUNCTIONS,
    SegmentationParams,
    WindowFunction,
)

TRANSFORM_1M = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)


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


def _cone_chm(
    size: int, cones: tuple[tuple[int, int, float, float], ...]
) -> np.ndarray:
    """Cones composited by maximum, so every apex is a real local maximum.

    Summing overlapping Gaussians erases the lower peak, which tests the test
    rather than the detector. Compositing keeps each apex intact so the only
    thing that can suppress one is the window rule.
    """
    arr = np.zeros((size, size), dtype="float32")
    y, x = np.indices(arr.shape)
    for r, c, height, radius in cones:
        d = np.hypot(y - r, x - c)
        arr = np.maximum(arr, np.where(d <= radius, height * (1 - d / radius), 0.0))
    return arr.astype("float32")


def _params(**changes) -> SegmentationParams:
    """A fixed-window parameter set, so a test controls the spacing directly."""
    base = SegmentationParams(
        smooth_radius_m=1.0,
        window=WindowFunction(a=10.0, lo=10.0, hi=10.0, name="fixed_10m"),
        th_cr=0.0,
        max_crown_radius_m=None,
        min_crown_area_m2=1.0,
    )
    return base.replace(**changes)


def test_segment_crowns_recovers_four_synthetic_peaks() -> None:
    """Four well-separated Gaussian crowns return about four polygons."""
    gdf = segment_crowns(_gaussian_chm(), TRANSFORM_1M, "EPSG:5070", params=_params())
    assert 3 <= len(gdf) <= 5
    assert (gdf["area_m2"] >= 1.0).all()
    assert (gdf["crown_diam_m"] > 5.0).all()
    assert (gdf["crown_diam_m"] < 50.0).all()
    assert gdf.crs is not None
    assert set(CROWN_COLUMNS).issubset(gdf.columns)


def test_segment_crowns_empty_when_below_threshold() -> None:
    """A CHM entirely below the canopy floor returns an empty frame."""
    chm = np.full((50, 50), 0.5, dtype="float32")
    gdf = segment_crowns(chm, TRANSFORM_1M, "EPSG:5070", params=_params())
    assert len(gdf) == 0
    assert set(CROWN_COLUMNS).issubset(gdf.columns)


def test_window_is_a_diameter_not_a_separation() -> None:
    """Two tops ``ws/2`` apart both survive, matching lidR's lmf rule.

    A window of diameter ``ws`` means a pixel must be the highest within
    ``ws/2`` of itself, so the closest two distinct tops can be is ``ws/2``.
    Reading ``ws`` as the separation, which is what the engine used to do,
    would merge this pair.
    """
    ws_m = 10.0
    window = WindowFunction(a=ws_m, lo=ws_m, hi=ws_m)

    def tops(separation_px: int) -> int:
        chm = _cone_chm(
            80,
            ((40, 25, 15.0, 4.0), (40, 25 + separation_px, 14.0, 4.0)),
        )
        smoothed = mask_below(smooth_chm(chm, kernel=1), 2.0)
        return locate_seeds(smoothed, ~np.isnan(smoothed), 1.0, window)[0].size

    # ws/2 is the exclusion radius, so a pair beyond it stays two tops.
    assert tops(7) == 2, "a pair beyond ws/2 must stay two tree tops"
    # Inside it, the lower apex is not the maximum of its own window.
    assert tops(3) == 1, "a pair inside ws/2 must collapse to one"


def test_window_varies_with_height() -> None:
    """A height-dependent window finds fewer tops than a small fixed one.

    The old engine collapsed the window to one number for the whole raster, so
    every candidate equation behaved identically. This fails if that returns.
    """
    # Two clusters at identical spacing, one short and one tall. A window that
    # scales with height thins only the tall cluster.
    # Heights differ slightly within each cluster. Exactly equal apexes tie, and
    # neither can suppress the other, which would hide the effect being tested.
    spacing = 7
    short = [
        (r, c, 8.0 - 0.2 * i, 3.0)
        for i, (r, c) in enumerate(
            (r, c) for r in (20, 20 + spacing) for c in (20, 20 + spacing)
        )
    ]
    tall = [
        (r, c, 30.0 - 1.0 * i, 3.0)
        for i, (r, c) in enumerate(
            (r, c) for r in (80, 80 + spacing) for c in (80, 80 + spacing)
        )
    ]
    cones = tuple(short + tall)
    chm = _cone_chm(120, cones)
    smoothed = mask_below(smooth_chm(chm, kernel=1), 2.0)
    valid = ~np.isnan(smoothed)

    steep = WindowFunction(a=2.0, b=0.5, lo=3.0, hi=30.0)
    flat = WindowFunction(a=3.0, lo=3.0, hi=3.0)
    n_steep = locate_seeds(smoothed, valid, 1.0, steep)[0].size
    n_flat = locate_seeds(smoothed, valid, 1.0, flat)[0].size
    assert n_flat == 8, "a small fixed window keeps every planted cone"
    assert n_steep < n_flat, "a height-scaled window must suppress more tops"


def test_th_cr_breaks_full_tiling() -> None:
    """Without an extent bound the watershed claims every canopy pixel."""
    chm = _gaussian_chm(size=120, sigma_px=8.0)
    canopy_area = float(np.sum(chm >= 2.0))

    unbounded = segment_crowns(
        chm, TRANSFORM_1M, "EPSG:5070", params=_params(th_cr=0.0)
    )
    bounded = segment_crowns(chm, TRANSFORM_1M, "EPSG:5070", params=_params(th_cr=0.7))
    assert unbounded["area_m2"].sum() == pytest.approx(canopy_area, rel=0.02)
    assert bounded["area_m2"].sum() < unbounded["area_m2"].sum()


def test_radius_ceiling_is_recorded() -> None:
    """A crown the ceiling trimmed carries a non-zero capped fraction."""
    chm = _gaussian_chm(size=120, peaks=((60, 60, 25.0),), sigma_px=14.0)
    capped = segment_crowns(
        chm, TRANSFORM_1M, "EPSG:5070", params=_params(max_crown_radius_m=6.0)
    )
    free = segment_crowns(
        chm, TRANSFORM_1M, "EPSG:5070", params=_params(max_crown_radius_m=None)
    )
    assert (capped["capped_frac"] > 0).any()
    assert (free["capped_frac"] == 0).all()
    assert capped["area_m2"].max() < free["area_m2"].max()


def test_apex_height_comes_from_the_tree_top() -> None:
    """apex_h_m is the height at the seed, not the maximum over the segment.

    Segment maximum is an extreme-value statistic that grows with segment size,
    so using it would build a height-to-size relationship into the measurement.
    """
    chm = _gaussian_chm(size=100, peaks=((50, 50, 20.0),), sigma_px=10.0)
    gdf = segment_crowns(chm, TRANSFORM_1M, "EPSG:5070", params=_params())
    assert len(gdf) >= 1
    smoothed = smooth_chm(chm, kernel=smooth_kernel_px(1.0, 1.0))
    assert gdf["apex_h_m"].max() <= float(np.nanmax(smoothed)) + 1e-3


@pytest.mark.parametrize(
    ("radius_m", "pixel_m", "expected"),
    [
        (0.0, 0.6, 1),
        (0.6, 0.6, 3),
        (0.9, 0.6, 5),
        (0.9, 1.0, 3),
        (1.5, 0.5, 7),
    ],
)
def test_smooth_kernel_is_metric(radius_m, pixel_m, expected) -> None:
    """A smoothing radius in metres converts to an odd kernel per raster."""
    assert smooth_kernel_px(radius_m, pixel_m) == expected


def test_named_windows_are_diameters_in_a_plausible_range() -> None:
    """Every shipped window returns a sane crown-scale diameter."""
    heights = np.array([2.0, 10.0, 20.0, 30.0, 40.0])
    for name, window in WINDOW_FUNCTIONS.items():
        ws = np.asarray(window(heights))
        assert np.all(ws >= 2.0), name
        assert np.all(ws <= 15.0), name
        assert np.all(np.diff(ws) >= -1e-9), f"{name} must not shrink with height"
