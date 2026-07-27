"""csdv_core.metrics.spatial - Spatial pattern metrics.

Phase 2b implements three V5 spatial metrics:

* :func:`linearity_index` - Hough-transform peak ratio over the edges of a
  binary gap mask, per window. Used to flag utility corridors and other
  linear openings. V5 specifies the index as "compute on gap mask or
  maintained opening mask"; the specific normalization is unstable.
* :func:`edge_density` - canopy/non-canopy boundary pixels per window area
  (units 1/m). Operates on a binary canopy mask.
* :func:`row_directionality` - FFT power-spectrum peakedness along the
  dominant orientation per window. Targets row-structured plantations. V5
  spec is unstable; the implementation here uses a generic radial-vs-angular
  peak ratio.

All inputs are 2-D NumPy arrays. Mask construction (e.g. ``chm < 2.0``) is
the caller's responsibility so this module remains array-pure.
"""

from __future__ import annotations

import logging

import numpy as np
from skimage.transform import hough_line

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


def edge_mask(mask: np.ndarray) -> np.ndarray:
    """Boolean edge map: True where a pixel differs from any 4-neighbor."""
    m = mask.astype(bool)
    edge = np.zeros_like(m)
    edge[:-1, :] |= m[:-1, :] != m[1:, :]
    edge[1:, :] |= m[:-1, :] != m[1:, :]
    edge[:, :-1] |= m[:, :-1] != m[:, 1:]
    edge[:, 1:] |= m[:, :-1] != m[:, 1:]
    return edge


def hough_peak_ratio(edges: np.ndarray, *, n_angles: int = 90) -> float:
    """Ratio of the strongest Hough accumulator peak to the mean.

    Returns 0.0 if no edge pixels are present. Higher values indicate a
    stronger single dominant line orientation. The output is clipped to
    [0, 1] using a soft normalization ``1 - mean/peak``.
    """
    if not edges.any():
        return 0.0
    thetas = np.linspace(-np.pi / 2, np.pi / 2, n_angles, endpoint=False)
    h, _, _ = hough_line(edges.astype(np.uint8), theta=thetas)
    peak = float(h.max())
    mean = float(h[h > 0].mean()) if (h > 0).any() else 0.0
    if peak <= 0.0:
        return 0.0
    return float(np.clip(1.0 - mean / peak, 0.0, 1.0))


@register("linearity_index")
def linearity_index(
    gap_mask: np.ndarray,
    grid: GridSpec,
    *,
    window_m: float = 50.0,
    n_angles: int = 90,
) -> MetricResult:
    """Per-window linearity index of a binary gap mask in ``[0, 1]``.

    Computed as ``1 - mean / peak`` of the Hough accumulator over the edge
    map of ``gap_mask``. Striped or single-orientation features approach 1;
    isotropic or empty windows approach 0. Spec is unstable; threshold
    calibration is deferred.

    Args:
        gap_mask: 2-D boolean (or 0/1) array marking gap pixels.
        grid: Source grid spec.
        window_m: Window side length in meters.
        n_angles: Number of Hough angles to sample over [-pi/2, pi/2).

    Returns:
        MetricResult with one value per window, NaN where the window is
        empty of edges and not even derivable.
    """
    mask = np.asarray(gap_mask).astype(bool)
    window_px = window_pixels(window_m, grid.pixel_size_m)
    n_rows, n_cols = tile_shape(mask.shape[0], mask.shape[1], window_px)
    out = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    for r, c, tile in iter_tiles(mask, window_px):
        edges = edge_mask(tile)
        out[r, c] = hough_peak_ratio(edges, n_angles=n_angles)
    logger.info(
        "linearity_index: %dx%d windows (%.0fm, px=%.2fm, n_angles=%d)",
        n_rows,
        n_cols,
        window_m,
        grid.pixel_size_m,
        n_angles,
    )
    return make_result(
        "linearity_index",
        out,
        transform=window_transform(grid.transform, window_m),
        crs=grid.crs,
        window_m=window_m,
        params={"n_angles": n_angles},
        units="",
    )


@register("edge_density")
def edge_density(
    canopy_mask: np.ndarray,
    grid: GridSpec,
    *,
    window_m: float = 50.0,
) -> MetricResult:
    """Per-window canopy edge density in 1/m.

    Edge pixels are pixels whose 4-neighborhood crosses a canopy/non-canopy
    boundary. Density is ``edge_pixel_count * pixel_size_m / window_area``,
    which has units 1/m.

    Args:
        canopy_mask: 2-D boolean (or 0/1) canopy mask.
        grid: Source grid spec.
        window_m: Window side length in meters.

    Returns:
        MetricResult with edge density per window. Units: 1/m.
    """
    mask = np.asarray(canopy_mask).astype(bool)
    window_px = window_pixels(window_m, grid.pixel_size_m)
    n_rows, n_cols = tile_shape(mask.shape[0], mask.shape[1], window_px)
    out = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    px = grid.pixel_size_m
    window_area_m2 = float(window_m * window_m)
    for r, c, tile in iter_tiles(mask, window_px):
        edges = edge_mask(tile)
        edge_len_m = float(edges.sum()) * px
        out[r, c] = edge_len_m / window_area_m2
    logger.info(
        "edge_density: %dx%d windows (%.0fm, px=%.2fm)",
        n_rows,
        n_cols,
        window_m,
        grid.pixel_size_m,
    )
    return make_result(
        "edge_density",
        out,
        transform=window_transform(grid.transform, window_m),
        crs=grid.crs,
        window_m=window_m,
        params={},
        units="1/m",
    )


def fft_directionality(tile: np.ndarray, *, n_bins: int = 36) -> float:
    """Ratio of peak angular power-spectrum bin to the mean over bins.

    Returns 0 for flat or single-value tiles. Higher values mean energy is
    concentrated in one direction (rows/columns of trees). Output clipped
    to [0, 1] via ``1 - mean/peak``.
    """
    valid = np.isfinite(tile)
    if valid.sum() < 16:
        return float("nan")
    img = np.where(valid, tile, np.nanmean(tile[valid])).astype(np.float64)
    img = img - img.mean()
    if not np.any(img):
        return 0.0
    rows, cols = img.shape
    win = np.outer(np.hanning(rows), np.hanning(cols))
    f = np.fft.fftshift(np.fft.fft2(img * win))
    power = np.abs(f) ** 2
    cy, cx = rows // 2, cols // 2
    yy, xx = np.indices(power.shape)
    dy = yy - cy
    dx = xx - cx
    r = np.hypot(dy, dx)
    inner = (r > 1.0) & (r < min(cy, cx))
    if not inner.any():
        return 0.0
    theta = np.arctan2(dy[inner], dx[inner]) % np.pi
    p = power[inner]
    bins = np.linspace(0.0, np.pi, n_bins + 1)
    idx = np.clip(np.digitize(theta, bins) - 1, 0, n_bins - 1)
    sums = np.bincount(idx, weights=p, minlength=n_bins)
    if sums.max() <= 0.0:
        return 0.0
    peak = float(sums.max())
    mean = float(sums.mean())
    return float(np.clip(1.0 - mean / peak, 0.0, 1.0))


@register("row_directionality")
def row_directionality(
    image: np.ndarray,
    grid: GridSpec,
    *,
    window_m: float = 50.0,
    n_bins: int = 36,
) -> MetricResult:
    """Per-window FFT-based row directionality in ``[0, 1]``.

    V5 specifies a "row directionality / periodicity index" for detecting
    planted rows. The exact normalization in V5 is unstable; this
    implementation reports ``1 - mean/peak`` of the angular FFT power
    spectrum, which is monotonic in directional energy concentration but
    not calibrated against V5 reference values. Treat magnitudes as
    relative, not absolute.

    Args:
        image: 2-D array (CHM or any single-band image). NaNs allowed.
        grid: Source grid spec.
        window_m: Window side length in meters.
        n_bins: Number of angular bins over [0, pi).

    Returns:
        MetricResult with values in [0, 1] or NaN where too few valid pixels.
    """
    arr = np.asarray(image, dtype=np.float32)
    window_px = window_pixels(window_m, grid.pixel_size_m)
    n_rows, n_cols = tile_shape(arr.shape[0], arr.shape[1], window_px)
    out = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    for r, c, tile in iter_tiles(arr, window_px):
        out[r, c] = fft_directionality(tile, n_bins=n_bins)
    logger.info(
        "row_directionality: %dx%d windows (%.0fm, px=%.2fm, bins=%d)",
        n_rows,
        n_cols,
        window_m,
        grid.pixel_size_m,
        n_bins,
    )
    return make_result(
        "row_directionality",
        out,
        transform=window_transform(grid.transform, window_m),
        crs=grid.crs,
        window_m=window_m,
        params={"n_bins": n_bins},
        units="",
    )


# The three kernels above are pure functions of an array and are reused by
# csdv_core.zonal.spatial, which applies them inside a stand polygon rather than
# over a grid of windows. They keep their old private names as aliases so that
# any existing import continues to resolve.
_edge_mask = edge_mask
_hough_peak_ratio = hough_peak_ratio
_fft_directionality = fft_directionality

__all__ = [
    "edge_density",
    "edge_mask",
    "fft_directionality",
    "hough_peak_ratio",
    "linearity_index",
    "row_directionality",
]
