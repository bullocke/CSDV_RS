"""csdv_core.zonal.pixel — canopy height metrics inside a stand polygon.

These mirror the arithmetic of :mod:`csdv_core.metrics.gap` and
:mod:`csdv_core.metrics.cover` but report a single value per stand instead of a
grid of windows. The windowed versions cannot be reused because they floor-
divide the array into tiles before doing anything else.

Inputs are a canopy height array in metres and a boolean in-stand mask of the
same shape, as produced by :mod:`csdv_core.zonal.mask`. NaN heights are invalid
and drop out of both numerator and denominator, which is the convention in
Appendix D of the classification document.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

CANOPY_HEIGHT_THRESHOLD_M = 2.0
SHRUB_BAND_M = (0.5, 2.0)
SMALL_TREE_BAND_M = (2.0, 10.0)
MID_CANOPY_BAND_M = (10.0, 20.0)
TALL_CANOPY_BAND_M = (20.0, 100.0)

#: Below this many valid pixels a height percentile is unstable, so no value is
#: reported. Matches the guard in the legacy patch-metric implementation.
MIN_CANOPY_PIXELS = 10

__all__ = [
    "CANOPY_HEIGHT_THRESHOLD_M",
    "crown_fraction",
    "gap_fraction",
    "gap_persistence",
    "height_band_fraction",
    "height_stats",
    "mid_canopy_fraction",
    "n_valid",
    "shrub_fraction",
    "small_tree_fraction",
    "stand_values",
    "tall_canopy_fraction",
]


def stand_values(chm: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the finite canopy height values inside the stand, as a 1-D array."""
    arr = np.asarray(chm, dtype=np.float32)
    if arr.shape != mask.shape:
        raise ValueError(f"Array shape {arr.shape} does not match mask {mask.shape}")
    return arr[mask & np.isfinite(arr)]


def n_valid(chm: np.ndarray, mask: np.ndarray) -> int:
    """Count the finite canopy height pixels inside the stand."""
    return int(stand_values(chm, mask).size)


def gap_fraction(
    chm: np.ndarray,
    mask: np.ndarray,
    *,
    height_threshold_m: float = CANOPY_HEIGHT_THRESHOLD_M,
) -> float:
    """Proportion of valid in-stand pixels below ``height_threshold_m``.

    Returns NaN when the stand contains no valid pixel.
    """
    values = stand_values(chm, mask)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values < height_threshold_m))


def crown_fraction(
    chm: np.ndarray,
    mask: np.ndarray,
    *,
    height_threshold_m: float = CANOPY_HEIGHT_THRESHOLD_M,
) -> float:
    """Proportion of valid in-stand pixels at or above ``height_threshold_m``."""
    values = stand_values(chm, mask)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values >= height_threshold_m))


def height_band_fraction(
    chm: np.ndarray,
    mask: np.ndarray,
    *,
    lo_m: float,
    hi_m: float,
) -> float:
    """Proportion of valid in-stand pixels in the half-open band ``[lo_m, hi_m)``."""
    values = stand_values(chm, mask)
    if values.size == 0:
        return float("nan")
    return float(np.mean((values >= lo_m) & (values < hi_m)))


def shrub_fraction(
    chm: np.ndarray,
    mask: np.ndarray,
    *,
    lo_m: float = SHRUB_BAND_M[0],
    hi_m: float = SHRUB_BAND_M[1],
) -> float:
    """Low woody cover, 0.5 to 2.0 m by default.

    This is a transient signal. It rises as a young cohort passes through the
    band and falls once the cohort grows past it, so a low value alone does not
    distinguish a site with no shrubs from one whose shrubs became trees.
    """
    return height_band_fraction(chm, mask, lo_m=lo_m, hi_m=hi_m)


def small_tree_fraction(chm: np.ndarray, mask: np.ndarray) -> float:
    """Proportion of in-stand pixels between 2 and 10 m."""
    return height_band_fraction(
        chm, mask, lo_m=SMALL_TREE_BAND_M[0], hi_m=SMALL_TREE_BAND_M[1]
    )


def mid_canopy_fraction(chm: np.ndarray, mask: np.ndarray) -> float:
    """Proportion of in-stand pixels between 10 and 20 m."""
    return height_band_fraction(
        chm, mask, lo_m=MID_CANOPY_BAND_M[0], hi_m=MID_CANOPY_BAND_M[1]
    )


def tall_canopy_fraction(chm: np.ndarray, mask: np.ndarray) -> float:
    """Proportion of in-stand pixels above 20 m."""
    return height_band_fraction(
        chm, mask, lo_m=TALL_CANOPY_BAND_M[0], hi_m=TALL_CANOPY_BAND_M[1]
    )


def height_stats(
    chm: np.ndarray,
    mask: np.ndarray,
    *,
    height_threshold_m: float = CANOPY_HEIGHT_THRESHOLD_M,
    min_pixels: int = MIN_CANOPY_PIXELS,
) -> dict[str, float]:
    """Summarise the canopy height surface inside the stand.

    Statistics are taken over pixels at or above ``height_threshold_m``, so they
    describe the canopy rather than the mixture of canopy and ground. Below
    ``min_pixels`` canopy pixels every statistic is NaN.

    Returns:
        ``height_mean``, ``height_median``, ``height_p90``, ``height_max`` in
        metres, and the dimensionless ``height_cv``.
    """
    nan_result = {
        "height_mean": float("nan"),
        "height_median": float("nan"),
        "height_p90": float("nan"),
        "height_max": float("nan"),
        "height_cv": float("nan"),
    }
    values = stand_values(chm, mask)
    canopy = values[values >= height_threshold_m]
    if canopy.size < min_pixels:
        return nan_result
    mean = float(np.mean(canopy))
    return {
        "height_mean": mean,
        "height_median": float(np.median(canopy)),
        "height_p90": float(np.percentile(canopy, 90)),
        "height_max": float(np.max(canopy)),
        "height_cv": float(np.std(canopy) / mean) if mean > 0 else float("nan"),
    }


def gap_persistence(
    chm_a: np.ndarray,
    chm_b: np.ndarray,
    mask: np.ndarray,
    *,
    height_threshold_m: float = CANOPY_HEIGHT_THRESHOLD_M,
) -> float:
    """Proportion of in-stand pixels that are gap at both dates.

    The denominator is the pixels valid at both dates, so a hole in one date
    does not count against the other. The two arrays must be on an identical
    grid; no resampling is performed, because the error in a comparison across
    dates compounds the error of both inputs and a resampling step adds a third.

    Returns NaN when no in-stand pixel is valid at both dates.

    Raises:
        ValueError: If the two arrays or the mask disagree on shape.
    """
    a = np.asarray(chm_a, dtype=np.float32)
    b = np.asarray(chm_b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"gap_persistence shape mismatch: {a.shape} vs {b.shape}")
    if a.shape != mask.shape:
        raise ValueError(f"Array shape {a.shape} does not match mask {mask.shape}")
    both_valid = mask & np.isfinite(a) & np.isfinite(b)
    n_both = int(both_valid.sum())
    if n_both == 0:
        return float("nan")
    joint_gap = both_valid & (a < height_threshold_m) & (b < height_threshold_m)
    return float(joint_gap.sum()) / float(n_both)
