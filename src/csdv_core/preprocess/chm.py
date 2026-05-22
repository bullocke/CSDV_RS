"""csdv_core.preprocess.chm — Pure CHM preprocessing helpers.

All functions operate on in-memory arrays. File I/O lives in
:mod:`csdv_core.io.raster`. The smoothing kernel mirrors the
``terra::focal(w=matrix(1,3,3), fun="mean")`` step from
``legacy/.../summary_document/02_crown_segmentation.R``.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import uniform_filter

logger = logging.getLogger(__name__)


def smooth_chm(chm: np.ndarray, kernel: int = 3) -> np.ndarray:
    """Apply a square mean filter to a CHM, ignoring NaN pixels.

    Args:
        chm: 2-D CHM in metres. NaN-valued pixels are excluded from the
            local mean, matching ``terra::focal(na.rm=TRUE)``.
        kernel: Side length in pixels. Must be a positive odd integer.

    Returns:
        Smoothed CHM as float32 with NaN where the entire neighbourhood was
        NaN.
    """
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError(f"kernel must be a positive odd integer, got {kernel}")
    arr = np.asarray(chm, dtype="float32")
    nan_mask = np.isnan(arr)
    filled = np.where(nan_mask, 0.0, arr)
    weights = (~nan_mask).astype("float32")
    summed = uniform_filter(filled, size=kernel, mode="reflect") * (kernel * kernel)
    counts = uniform_filter(weights, size=kernel, mode="reflect") * (kernel * kernel)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(counts > 0, summed / counts, np.nan).astype("float32")
    return out


def mask_below(chm: np.ndarray, threshold_m: float = 2.0) -> np.ndarray:
    """Return a copy of ``chm`` with values strictly below ``threshold_m`` as NaN.

    Args:
        chm: 2-D CHM in metres.
        threshold_m: Height threshold; pixels with values ``< threshold_m``
            become NaN. Default 2 m matches the gap definition used in
            ``metrics.gap`` and the legacy R script.

    Returns:
        Float32 array with sub-threshold pixels masked.
    """
    arr = np.asarray(chm, dtype="float32").copy()
    arr[arr < threshold_m] = np.nan
    return arr


def convert_uint16_to_meters(
    chm_u16: np.ndarray,
    *,
    scale: float = 0.01,
    src_nodata: float | int | None = None,
    dst_nodata: float = -9999.0,
) -> np.ndarray:
    """Convert a uint16 NAIP-CHM tile to float32 metres.

    The NAIP-CHM product stores heights as uint16 with a 0.01 m scale
    factor. This mirrors the in-memory portion of
    :func:`csdv_core.io.raster.clip_and_convert_naip_chm` for streamed
    workflows where the data are already in memory.

    Args:
        chm_u16: 2-D array of uint16 (or any integer) heights.
        scale: Multiplicative scale factor.
        src_nodata: Optional sentinel in ``chm_u16`` to replace with
            ``dst_nodata`` in the output.
        dst_nodata: Sentinel written for nodata pixels.

    Returns:
        Float32 array in metres.
    """
    out = np.asarray(chm_u16, dtype="float32") * float(scale)
    if src_nodata is not None:
        out = np.where(np.asarray(chm_u16) == src_nodata, dst_nodata, out)
    return out.astype("float32")


__all__ = ["smooth_chm", "mask_below", "convert_uint16_to_meters"]
