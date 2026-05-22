"""Topographic derivatives for site-type stratification.

Pure functions: take a DEM array plus its grid metadata and return a dict
of derivatives. No I/O. Reading and writing rasters belongs in ``cli.py``.

Derivatives produced (per V5 Section 2 environmental drivers):

- ``slope_deg``       slope in degrees
- ``aspect_deg``      aspect in degrees (0 = north, clockwise)
- ``northness``       cos(aspect)  (1 = north-facing, -1 = south-facing)
- ``eastness``        sin(aspect)
- ``twi``             topographic wetness index = ln(a / tan(slope))
- ``hand``            height above nearest drainage (m)
- ``rei``             relief-exposure index = elev - mean(elev within 1 km)
- ``profile_curvature`` profile curvature (positive = convex)

The implementation depends on ``richdem`` for slope, aspect, curvature, and
flow accumulation. ``richdem`` is imported lazily so unrelated unit tests
do not require it.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _import_richdem() -> Any:
    try:
        import richdem as rd  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "richdem is required for topographic stratification. "
            "Install with `mamba install -c conda-forge richdem`."
        ) from exc
    return rd


def _to_rd(dem: np.ndarray, no_data: float = -9999.0) -> Any:
    """Wrap a numpy DEM as a richdem array."""
    rd = _import_richdem()
    arr = np.where(np.isnan(dem), no_data, dem).astype("float32")
    rda = rd.rdarray(arr, no_data=no_data)
    return rda


def compute_slope_aspect(
    dem: np.ndarray,
    pixel_size_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slope (deg), aspect (deg), northness, eastness."""
    rd = _import_richdem()
    rda = _to_rd(dem)
    rda.geotransform = (0.0, pixel_size_m, 0.0, 0.0, 0.0, -pixel_size_m)
    slope = np.asarray(rd.TerrainAttribute(rda, attrib="slope_degrees"))
    aspect = np.asarray(rd.TerrainAttribute(rda, attrib="aspect"))
    aspect_rad = np.deg2rad(aspect)
    northness = np.cos(aspect_rad).astype("float32")
    eastness = np.sin(aspect_rad).astype("float32")
    return (
        slope.astype("float32"),
        aspect.astype("float32"),
        northness,
        eastness,
    )


def compute_profile_curvature(
    dem: np.ndarray,
    pixel_size_m: float,
) -> np.ndarray:
    """Profile curvature in 1/m. Positive values = convex slopes."""
    rd = _import_richdem()
    rda = _to_rd(dem)
    rda.geotransform = (0.0, pixel_size_m, 0.0, 0.0, 0.0, -pixel_size_m)
    curv = np.asarray(rd.TerrainAttribute(rda, attrib="profile_curvature"))
    return curv.astype("float32")


def compute_twi(
    dem: np.ndarray,
    pixel_size_m: float,
    *,
    min_slope_rad: float = 1e-3,
) -> np.ndarray:
    """Topographic wetness index ``ln(a / tan(beta))``.

    ``a`` is upslope contributing area per unit contour length, computed
    here as ``flow_accum * pixel_size_m`` (D-infinity). ``beta`` is local
    slope; flat pixels are clamped to ``min_slope_rad`` to avoid log(inf).
    """
    rd = _import_richdem()
    rda = _to_rd(dem)
    rda.geotransform = (0.0, pixel_size_m, 0.0, 0.0, 0.0, -pixel_size_m)
    filled = rd.FillDepressions(rda, in_place=False)
    accum = np.asarray(rd.FlowAccumulation(filled, method="Dinf")).astype("float32")
    slope_deg = np.asarray(rd.TerrainAttribute(filled, attrib="slope_degrees")).astype(
        "float32"
    )
    slope_rad = np.deg2rad(slope_deg)
    tan_beta = np.tan(np.maximum(slope_rad, min_slope_rad))
    a = (accum + 1.0) * pixel_size_m
    twi = np.log(a / tan_beta).astype("float32")
    return twi


def compute_hand(
    dem: np.ndarray,
    pixel_size_m: float,
    *,
    accumulation_threshold: float = 1000.0,
) -> np.ndarray:
    """Height Above Nearest Drainage in meters.

    Drainage cells are pixels with flow accumulation above
    ``accumulation_threshold`` (cells). HAND is the elevation difference
    to the nearest such cell, computed via a Euclidean nearest-neighbor
    lookup on the filled DEM. Approximate but adequate for stratification.
    """
    from scipy.ndimage import distance_transform_edt

    rd = _import_richdem()
    rda = _to_rd(dem)
    rda.geotransform = (0.0, pixel_size_m, 0.0, 0.0, 0.0, -pixel_size_m)
    filled = np.asarray(rd.FillDepressions(rda, in_place=False)).astype("float32")
    accum = np.asarray(rd.FlowAccumulation(rda, method="Dinf")).astype("float32")

    drainage = accum >= accumulation_threshold
    if not drainage.any():
        return np.full_like(filled, np.nan, dtype="float32")

    _, (idx_r, idx_c) = distance_transform_edt(~drainage, return_indices=True)
    nearest_elev = filled[idx_r, idx_c]
    hand = (filled - nearest_elev).astype("float32")
    hand = np.where(hand < 0, 0.0, hand).astype("float32")
    return hand


def compute_rei(
    dem: np.ndarray,
    pixel_size_m: float,
    *,
    window_m: float = 1000.0,
) -> np.ndarray:
    """Relief-exposure index: elevation minus moving-window mean elevation.

    Positive values = locally elevated (ridge); negative = locally depressed
    (valley). Window is a square of side ``window_m`` rounded to odd pixels.
    """
    from scipy.ndimage import uniform_filter

    arr = np.where(np.isnan(dem), 0.0, dem).astype("float32")
    valid = (~np.isnan(dem)).astype("float32")
    size = max(3, int(round(window_m / pixel_size_m)))
    if size % 2 == 0:
        size += 1
    sum_elev = uniform_filter(arr, size=size, mode="reflect") * (size * size)
    sum_valid = uniform_filter(valid, size=size, mode="reflect") * (size * size)
    mean_elev = np.where(sum_valid > 0, sum_elev / np.maximum(sum_valid, 1), np.nan)
    rei = (dem - mean_elev).astype("float32")
    return rei


def compute_topo(
    dem: np.ndarray,
    pixel_size_m: float,
    *,
    accumulation_threshold: float = 1000.0,
    rei_window_m: float = 1000.0,
) -> dict[str, np.ndarray]:
    """Compute the eight topographic derivatives.

    Args:
        dem: 2-D DEM in meters. ``np.nan`` for nodata.
        pixel_size_m: Pixel size in meters (square pixels assumed).
        accumulation_threshold: Cells of flow accumulation that define a
            drainage pixel for HAND.
        rei_window_m: Side length of the REI moving window in meters.

    Returns:
        Dict mapping derivative name to 2-D float32 array.
    """
    if dem.ndim != 2:
        raise ValueError(f"dem must be 2-D, got shape {dem.shape}")
    if pixel_size_m <= 0:
        raise ValueError(f"pixel_size_m must be positive, got {pixel_size_m}")

    slope, aspect, northness, eastness = compute_slope_aspect(dem, pixel_size_m)
    twi = compute_twi(dem, pixel_size_m)
    hand = compute_hand(
        dem, pixel_size_m, accumulation_threshold=accumulation_threshold
    )
    rei = compute_rei(dem, pixel_size_m, window_m=rei_window_m)
    profile_curvature = compute_profile_curvature(dem, pixel_size_m)
    return {
        "slope_deg": slope,
        "aspect_deg": aspect,
        "northness": northness,
        "eastness": eastness,
        "twi": twi,
        "hand": hand,
        "rei": rei,
        "profile_curvature": profile_curvature,
    }
