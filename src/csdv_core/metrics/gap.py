"""csdv_core.metrics.gap - Gap fraction and crown (canopy cover) fraction.

Pure-function port of ``poc_lib/metrics.py::gap_fraction`` and
``crown_fraction``. Inputs are NumPy arrays; output is a
:class:`csdv_core.metrics._result.MetricResult`.

Gap pixels are defined as CHM pixels with height below ``height_threshold_m``
(default 2.0 m per CSDV Classification V5). Crown fraction is the complement.
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


def _prepare_chm(chm: np.ndarray, nodata: float | None) -> np.ndarray:
    arr = np.asarray(chm, dtype=np.float32)
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    return arr


@register("gap_fraction")
def gap_fraction(
    chm: np.ndarray,
    grid: GridSpec,
    *,
    window_m: float = 25.0,
    height_threshold_m: float = 2.0,
    nodata: float | None = None,
) -> MetricResult:
    """Fraction of CHM pixels below ``height_threshold_m`` per window.

    Args:
        chm: 2-D canopy height array in meters.
        grid: :class:`GridSpec` describing the source raster.
        window_m: Window side length in meters. Default 25.
        height_threshold_m: Pixels strictly below this are gaps. Default 2.
        nodata: Optional sentinel value to mask before computation. If the
            array already uses NaN, leave as ``None``.

    Returns:
        :class:`MetricResult` with one cell per window. NaN where every
        window pixel is invalid.
    """
    data = _prepare_chm(chm, nodata)
    window_px = window_pixels(window_m, grid.pixel_size_m)
    n_rows, n_cols = tile_shape(data.shape[0], data.shape[1], window_px)
    out = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    for r, c, tile in iter_tiles(data, window_px):
        valid = tile[~np.isnan(tile)]
        if valid.size == 0:
            continue
        out[r, c] = float(np.mean(valid < height_threshold_m))
    logger.info(
        "gap_fraction: %dx%d windows (%.0fm, px=%.2fm, threshold=%.1fm)",
        n_rows,
        n_cols,
        window_m,
        grid.pixel_size_m,
        height_threshold_m,
    )
    return make_result(
        "gap_fraction",
        out,
        transform=window_transform(grid.transform, window_m),
        crs=grid.crs,
        window_m=window_m,
        params={"height_threshold_m": height_threshold_m},
        units="fraction",
    )


@register("crown_fraction")
def crown_fraction(
    chm: np.ndarray,
    grid: GridSpec,
    *,
    window_m: float = 25.0,
    height_threshold_m: float = 2.0,
    nodata: float | None = None,
) -> MetricResult:
    """Canopy cover fraction (1 - gap fraction) per window."""
    gf = gap_fraction(
        chm,
        grid,
        window_m=window_m,
        height_threshold_m=height_threshold_m,
        nodata=nodata,
    )
    arr = 1.0 - gf.array
    arr = np.where(np.isnan(gf.array), np.nan, arr).astype(np.float32)
    return make_result(
        "crown_fraction",
        arr,
        transform=gf.transform,
        crs=gf.crs,
        window_m=window_m,
        params={"height_threshold_m": height_threshold_m},
        units="fraction",
    )


__all__ = ["gap_fraction", "crown_fraction", "gap_persistence"]


@register("gap_persistence")
def gap_persistence(
    chm_t1: np.ndarray,
    chm_t2: np.ndarray,
    grid: GridSpec,
    *,
    window_m: float = 25.0,
    height_threshold_m: float = 2.0,
    nodata: float | None = None,
) -> MetricResult:
    """Fraction of pixels that are gap (CHM < threshold) at both dates.

    V5 "gap persistence" reports the share of a window where the canopy mask
    complement is True across two NAIP epochs. The two CHMs must be on the
    same grid; no resampling is performed.

    Args:
        chm_t1: 2-D CHM at date 1 (meters).
        chm_t2: 2-D CHM at date 2 (meters), same shape as ``chm_t1``.
        grid: Source grid spec (same for both dates).
        window_m: Window side length in meters. Default 25.
        height_threshold_m: Pixels strictly below this are gaps.
        nodata: Optional sentinel value applied to both arrays.

    Returns:
        MetricResult of joint-gap fraction in [0, 1]. NaN where every pixel
        in the window is invalid in either date.

    Raises:
        ValueError: If the two arrays have different shapes.
    """
    a = _prepare_chm(chm_t1, nodata)
    b = _prepare_chm(chm_t2, nodata)
    if a.shape != b.shape:
        raise ValueError(
            f"gap_persistence shape mismatch: {a.shape} vs {b.shape}",
        )
    window_px = window_pixels(window_m, grid.pixel_size_m)
    n_rows, n_cols = tile_shape(a.shape[0], a.shape[1], window_px)
    out = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    both_valid_all = ~(np.isnan(a) | np.isnan(b))
    joint_gap_all = (a < height_threshold_m) & (b < height_threshold_m) & both_valid_all
    for r in range(n_rows):
        for c in range(n_cols):
            sl = (
                slice(r * window_px, (r + 1) * window_px),
                slice(c * window_px, (c + 1) * window_px),
            )
            valid = both_valid_all[sl]
            n_valid = int(valid.sum())
            if n_valid == 0:
                continue
            out[r, c] = float(joint_gap_all[sl].sum()) / n_valid
    logger.info(
        "gap_persistence: %dx%d windows (%.0fm, px=%.2fm, threshold=%.1fm)",
        n_rows,
        n_cols,
        window_m,
        grid.pixel_size_m,
        height_threshold_m,
    )
    return make_result(
        "gap_persistence",
        out,
        transform=window_transform(grid.transform, window_m),
        crs=grid.crs,
        window_m=window_m,
        params={"height_threshold_m": height_threshold_m},
        units="fraction",
    )
