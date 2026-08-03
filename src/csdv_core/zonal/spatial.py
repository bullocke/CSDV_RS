"""csdv_core.zonal.spatial — pattern metrics inside a stand polygon.

Edge density, linearity and row directionality all read the arrangement of
canopy and gap rather than how much of each there is. Like texture they need a
rectangular grid, so they run over the stand's bounding box with the stand mask
applied, and the in-stand share of that box travels with the result.

Two things go wrong if a masked bounding box is handed straight to the windowed
versions in :mod:`csdv_core.metrics.spatial`.

The stand boundary is itself a canopy-to-nothing transition, so an edge detector
traces the outline of the polygon. On a compact stand that adds a fixed
perimeter; on a long thin one it can dominate. Both edge density and linearity
here erode the stand mask by one pixel and keep only edges strictly inside it,
so the number describes the canopy and not the delineation.

And the windowed edge density divides by the window area, which for a stand is
the bounding box rather than the stand. Dividing by the in-stand area instead
makes the value comparable between a compact stand and a thin one.

The pure kernels are reused from :mod:`csdv_core.metrics.spatial` unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from csdv_core.metrics.spatial import edge_mask, fft_directionality, hough_peak_ratio

logger = logging.getLogger(__name__)

#: Below this in-stand share of the bounding box, row directionality is not
#: reported. The FFT fills invalid pixels with the in-stand mean and windows the
#: whole box, so a sparsely filled box measures the mask, not the canopy.
MIN_DIRECTIONALITY_SUPPORT = 0.5

__all__ = [
    "MIN_DIRECTIONALITY_SUPPORT",
    "SpatialResult",
    "interior_edge_mask",
    "stand_edge_density",
    "stand_linearity",
    "stand_row_directionality",
    "stand_spatial_metrics",
]


@dataclass(frozen=True)
class SpatialResult:
    """A spatial metric with the support it was computed on.

    Attributes:
        value: The metric, or NaN.
        support_fraction: In-stand share of the bounding box.
        reason: Empty when computed, otherwise why the value is NaN.
    """

    value: float
    support_fraction: float
    reason: str = ""


def interior_edge_mask(binary: np.ndarray, stand_mask: np.ndarray) -> np.ndarray:
    """Edges of ``binary`` that lie strictly inside the stand.

    The stand mask is eroded by one pixel before the edge map is intersected
    with it, which removes the transition the polygon boundary creates against
    everything outside.
    """
    inner = ndi.binary_erosion(stand_mask, structure=np.ones((3, 3), dtype=bool))
    return edge_mask(binary & stand_mask) & inner


def stand_edge_density(
    canopy_mask: np.ndarray,
    stand_mask: np.ndarray,
    *,
    pixel_size_m: float,
) -> SpatialResult:
    """Canopy edge length per unit stand area, in 1/m.

    Edge pixels are counted only inside the stand, and the denominator is the
    stand area rather than the bounding box area, so the value does not change
    with the shape of the box.
    """
    n_inside = int(stand_mask.sum())
    support = (
        float(n_inside) / float(stand_mask.size) if stand_mask.size else float("nan")
    )
    area_m2 = float(n_inside) * pixel_size_m**2
    if area_m2 <= 0.0:
        return SpatialResult(float("nan"), support, "stand has no pixels")
    edges = interior_edge_mask(np.asarray(canopy_mask, dtype=bool), stand_mask)
    edge_len_m = float(edges.sum()) * pixel_size_m
    return SpatialResult(edge_len_m / area_m2, support)


def stand_linearity(
    gap_mask: np.ndarray,
    stand_mask: np.ndarray,
    *,
    n_angles: int = 90,
) -> SpatialResult:
    """How linear the gap pattern inside the stand is, on a 0 to 1 scale.

    High values mean the openings line up, which is the signature of a utility
    corridor or another maintained strip. The normalisation is not calibrated,
    so compare values against each other rather than reading them as absolute.
    """
    support = (
        float(stand_mask.sum()) / float(stand_mask.size)
        if stand_mask.size
        else float("nan")
    )
    edges = interior_edge_mask(np.asarray(gap_mask, dtype=bool), stand_mask)
    if not edges.any():
        return SpatialResult(float("nan"), support, "no gap edges inside the stand")
    return SpatialResult(hough_peak_ratio(edges, n_angles=n_angles), support)


def stand_row_directionality(
    image: np.ndarray,
    stand_mask: np.ndarray,
    *,
    n_bins: int = 36,
    min_support: float = MIN_DIRECTIONALITY_SUPPORT,
) -> SpatialResult:
    """How regularly and directionally spaced the canopy is, 0 to 1.

    High values mean crowns are arranged along a preferred direction, which is
    the signature of a row plantation. Reported only where the stand fills at
    least ``min_support`` of its bounding box, because the transform is applied
    to the whole box and a sparsely filled box measures the mask instead. Same
    normalisation caveat as :func:`stand_linearity`.
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.shape != stand_mask.shape:
        raise ValueError(
            f"Array shape {arr.shape} does not match mask {stand_mask.shape}"
        )
    support = (
        float(stand_mask.sum()) / float(stand_mask.size)
        if stand_mask.size
        else float("nan")
    )
    if not np.isfinite(support) or support < min_support:
        return SpatialResult(
            float("nan"),
            support,
            f"support_fraction={support:.2f} < min_support={min_support:.2f}",
        )
    masked = np.where(stand_mask, arr, np.nan)
    return SpatialResult(float(fft_directionality(masked, n_bins=n_bins)), support)


def stand_spatial_metrics(
    chm: np.ndarray,
    stand_mask: np.ndarray,
    *,
    pixel_size_m: float,
    height_threshold_m: float = 2.0,
    n_angles: int = 90,
    n_bins: int = 36,
) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
    """Compute every spatial metric for one stand from its canopy height array.

    Returns:
        ``(metrics, support, reasons)`` keyed by the names the metric registry
        uses, so the values can be matched against a stage or trajectory rule
        without translation.
    """
    arr = np.asarray(chm, dtype=np.float32)
    valid = stand_mask & np.isfinite(arr)
    canopy = valid & (arr >= height_threshold_m)
    gap = valid & (arr < height_threshold_m)

    density = stand_edge_density(canopy, stand_mask, pixel_size_m=pixel_size_m)
    linear = stand_linearity(gap, stand_mask, n_angles=n_angles)
    rows = stand_row_directionality(arr, stand_mask, n_bins=n_bins)

    metrics = {
        "edge_density": density.value,
        "linearity_index": linear.value,
        "row_directionality": rows.value,
    }
    support = {"support_fraction": density.support_fraction}
    reasons = {
        name: result.reason
        for name, result in (
            ("edge_density", density),
            ("linearity_index", linear),
            ("row_directionality", rows),
        )
        if result.reason
    }
    return metrics, support, reasons
