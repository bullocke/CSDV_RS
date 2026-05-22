"""csdv_core.preprocess.naip — NAIP imagery preprocessing helpers.

Thin wrappers over :mod:`rasterio` for NAIP-specific normalization and
reprojection. File reading/writing lives in :mod:`csdv_core.io.raster`;
this module operates on arrays and rasterio metadata.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def normalize_to_uint8(
    arr: np.ndarray,
    *,
    src_min: float | None = None,
    src_max: float | None = None,
) -> np.ndarray:
    """Linearly scale ``arr`` to uint8 over ``[src_min, src_max]``.

    Args:
        arr: Input array, any numeric dtype. NaN values become 0.
        src_min: Lower bound of the source range. Defaults to ``np.nanmin``.
        src_max: Upper bound of the source range. Defaults to ``np.nanmax``.

    Returns:
        ``uint8`` array with the same shape as ``arr``.
    """
    src_min = float(np.nanmin(arr)) if src_min is None else float(src_min)
    src_max = float(np.nanmax(arr)) if src_max is None else float(src_max)
    if src_max <= src_min:
        return np.zeros_like(arr, dtype="uint8")
    scaled = (np.asarray(arr, dtype="float32") - src_min) / (src_max - src_min)
    scaled = np.clip(scaled, 0.0, 1.0)
    scaled = np.nan_to_num(scaled, nan=0.0)
    return (scaled * 255.0 + 0.5).astype("uint8")


def reproject_array(
    src_arr: np.ndarray,
    *,
    src_transform: Any,
    src_crs: Any,
    dst_crs: Any,
    dst_resolution: float | None = None,
    resampling: str = "bilinear",
) -> tuple[np.ndarray, Any]:
    """Reproject a 2-D array to a new CRS using ``rasterio.warp.reproject``.

    Args:
        src_arr: 2-D source array.
        src_transform: Source affine transform.
        src_crs: Source CRS.
        dst_crs: Destination CRS.
        dst_resolution: Output pixel size in destination CRS units. If
            None, the source resolution is preserved.
        resampling: Resampling method name (``"nearest"``, ``"bilinear"``,
            ``"cubic"``, ...).

    Returns:
        Tuple of ``(dst_arr, dst_transform)``.
    """
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject

    height, width = src_arr.shape
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs,
        dst_crs,
        width,
        height,
        *_bounds_from_transform(src_transform, width, height),
        resolution=dst_resolution,
    )
    dst_arr = np.zeros((dst_height, dst_width), dtype=src_arr.dtype)
    reproject(
        source=src_arr,
        destination=dst_arr,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling[resampling],
    )
    return dst_arr, dst_transform


def _bounds_from_transform(
    transform: Any, width: int, height: int
) -> tuple[float, float, float, float]:
    """Return ``(west, south, east, north)`` from an affine transform."""
    west, north = transform * (0, 0)
    east, south = transform * (width, height)
    return west, south, east, north


__all__ = ["normalize_to_uint8", "reproject_array"]
