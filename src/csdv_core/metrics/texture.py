"""csdv_core.metrics.texture - GLCM-based textural metrics.

Uses :func:`skimage.feature.graycomatrix` to compute a per-window
gray-level co-occurrence matrix at multiple angles, then averages a chosen
property (``entropy``, ``contrast``, ``homogeneity``, ...) across angles.

Inputs are 2-D arrays; floating-point arrays are quantized to ``levels``
bins after dropping NaN. Windows with too few valid pixels return NaN.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from skimage.feature import graycomatrix, graycoprops

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

GlcmProp = Literal[
    "contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"
]


def _quantize(tile: np.ndarray, levels: int) -> np.ndarray | None:
    """Quantize a window to ``[0, levels-1]``; return ``None`` if mostly invalid."""
    finite = np.isfinite(tile)
    if finite.sum() < max(8, int(0.5 * tile.size)):
        return None
    vmin = float(tile[finite].min())
    vmax = float(tile[finite].max())
    if vmax <= vmin:
        return np.zeros(tile.shape, dtype=np.uint8)
    scaled = (tile - vmin) / (vmax - vmin) * (levels - 1)
    quant = np.clip(scaled, 0, levels - 1).astype(np.uint8)
    quant[~finite] = 0
    return quant


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


@register("glcm_texture")
def glcm_texture(
    image: np.ndarray,
    grid: GridSpec,
    *,
    window_m: float = 50.0,
    levels: int = 16,
    distances: tuple[int, ...] = (1,),
    angles: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    prop: GlcmProp | Literal["entropy"] = "entropy",
) -> MetricResult:
    """Per-window GLCM property, averaged across ``angles`` and ``distances``.

    Args:
        image: 2-D array. NaN/inf pixels are treated as invalid.
        grid: :class:`GridSpec` for the source raster.
        window_m: Window side length in meters.
        levels: Number of gray levels to quantize to. Default 16.
        distances: GLCM pixel offsets.
        angles: GLCM offset angles in radians.
        prop: GLCM property. ``"entropy"`` is computed directly; other values
            are passed to :func:`skimage.feature.graycoprops`.

    Returns:
        :class:`MetricResult` with one cell per window.
    """
    arr = np.asarray(image, dtype=np.float32)
    window_px = window_pixels(window_m, grid.pixel_size_m)
    n_rows, n_cols = tile_shape(arr.shape[0], arr.shape[1], window_px)
    out = np.full((n_rows, n_cols), np.nan, dtype=np.float32)

    for r, c, tile in iter_tiles(arr, window_px):
        quant = _quantize(tile, levels)
        if quant is None:
            continue
        glcm = graycomatrix(
            quant,
            distances=list(distances),
            angles=list(angles),
            levels=levels,
            symmetric=True,
            normed=True,
        )
        if prop == "entropy":
            # graycomatrix axes: (level, level, distance, angle)
            vals = []
            for di in range(glcm.shape[2]):
                for ai in range(glcm.shape[3]):
                    vals.append(_entropy(glcm[:, :, di, ai].ravel()))
            out[r, c] = float(np.mean(vals))
        else:
            vals = graycoprops(glcm, prop=prop)
            out[r, c] = float(np.mean(vals))

    logger.info(
        "glcm_texture(%s): %dx%d windows (%.0fm), non-NaN = %d",
        prop,
        n_rows,
        n_cols,
        window_m,
        int(np.sum(~np.isnan(out))),
    )
    return make_result(
        "glcm_texture",
        out,
        transform=window_transform(grid.transform, window_m),
        crs=grid.crs,
        window_m=window_m,
        params={
            "levels": levels,
            "distances": list(distances),
            "angles": list(angles),
            "prop": prop,
        },
        units="bits" if prop == "entropy" else "",
    )


__all__ = ["glcm_texture"]
