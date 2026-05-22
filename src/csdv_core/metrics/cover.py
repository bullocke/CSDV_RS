"""csdv_core.metrics.cover - CHM-based height-band cover fractions.

Phase 2b implements CHM-only height stratification: each metric is the
fraction of valid CHM pixels in a height band ``[lo_m, hi_m)`` per window.
Bands match the V5 mask/cover tiers reachable from CHM alone (shrub, small
tree, mid canopy, tall canopy). Bare-ground, herbaceous, and sawtimber
sub-tiers require NAIP NDVI and are deferred to Phase 2c.
"""

from __future__ import annotations

import logging

import numpy as np

from csdv_core.io.grids import GridSpec
from csdv_core.metrics._result import MetricResult, make_result
from csdv_core.metrics._window import (
    iter_tiles,
    tile_shape,
    window_pixels,
    window_transform,
)
from csdv_core.metrics.registry import register

logger = logging.getLogger(__name__)


def _height_band_fraction(
    chm: np.ndarray,
    grid: GridSpec,
    *,
    name: str,
    window_m: float,
    lo_m: float,
    hi_m: float,
    nodata: float | None,
) -> MetricResult:
    """Per-window fraction of valid CHM pixels in ``[lo_m, hi_m)``.

    Args:
        chm: 2-D CHM array in meters.
        grid: Source grid spec.
        name: Registered metric name (used in the result).
        window_m: Window side length in meters.
        lo_m: Inclusive lower height bound.
        hi_m: Exclusive upper height bound.
        nodata: Optional sentinel value to mask before computation.

    Returns:
        MetricResult with values in ``[0, 1]`` or NaN where no valid pixels.
    """
    arr = np.asarray(chm, dtype=np.float32)
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    window_px = window_pixels(window_m, grid.pixel_size_m)
    n_rows, n_cols = tile_shape(arr.shape[0], arr.shape[1], window_px)
    out = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    for r, c, tile in iter_tiles(arr, window_px):
        valid = tile[~np.isnan(tile)]
        if valid.size == 0:
            continue
        out[r, c] = float(np.mean((valid >= lo_m) & (valid < hi_m)))
    logger.info(
        "%s: %dx%d windows (%.0fm, px=%.2fm, band=[%.1f,%.1f) m)",
        name,
        n_rows,
        n_cols,
        window_m,
        grid.pixel_size_m,
        lo_m,
        hi_m,
    )
    return make_result(
        name,
        out,
        transform=window_transform(grid.transform, window_m),
        crs=grid.crs,
        window_m=window_m,
        params={"lo_m": lo_m, "hi_m": hi_m},
        units="fraction",
    )


@register("shrub_fraction")
def shrub_fraction(
    chm: np.ndarray,
    grid: GridSpec,
    *,
    window_m: float = 25.0,
    lo_m: float = 0.5,
    hi_m: float = 2.0,
    nodata: float | None = None,
) -> MetricResult:
    """Fraction of CHM pixels in the shrub/low-woody band (default 0.5-2 m)."""
    return _height_band_fraction(
        chm,
        grid,
        name="shrub_fraction",
        window_m=window_m,
        lo_m=lo_m,
        hi_m=hi_m,
        nodata=nodata,
    )


@register("small_tree_fraction")
def small_tree_fraction(
    chm: np.ndarray,
    grid: GridSpec,
    *,
    window_m: float = 25.0,
    lo_m: float = 2.0,
    hi_m: float = 10.0,
    nodata: float | None = None,
) -> MetricResult:
    """Fraction of CHM pixels in the small-tree band (default 2-10 m)."""
    return _height_band_fraction(
        chm,
        grid,
        name="small_tree_fraction",
        window_m=window_m,
        lo_m=lo_m,
        hi_m=hi_m,
        nodata=nodata,
    )


@register("mid_canopy_fraction")
def mid_canopy_fraction(
    chm: np.ndarray,
    grid: GridSpec,
    *,
    window_m: float = 25.0,
    lo_m: float = 10.0,
    hi_m: float = 20.0,
    nodata: float | None = None,
) -> MetricResult:
    """Fraction of CHM pixels in the mid-canopy band (default 10-20 m)."""
    return _height_band_fraction(
        chm,
        grid,
        name="mid_canopy_fraction",
        window_m=window_m,
        lo_m=lo_m,
        hi_m=hi_m,
        nodata=nodata,
    )


@register("tall_canopy_fraction")
def tall_canopy_fraction(
    chm: np.ndarray,
    grid: GridSpec,
    *,
    window_m: float = 25.0,
    lo_m: float = 20.0,
    hi_m: float = 100.0,
    nodata: float | None = None,
) -> MetricResult:
    """Fraction of CHM pixels in the tall-canopy band (default >=20 m)."""
    return _height_band_fraction(
        chm,
        grid,
        name="tall_canopy_fraction",
        window_m=window_m,
        lo_m=lo_m,
        hi_m=hi_m,
        nodata=nodata,
    )


__all__ = [
    "shrub_fraction",
    "small_tree_fraction",
    "mid_canopy_fraction",
    "tall_canopy_fraction",
]
