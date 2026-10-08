"""csdv_core.zonal.components — residual canopy and disturbance gaps in a stand.

A partial disturbance leaves a stand with two kinds of ground: canopy that
survived and openings the disturbance created. A single stand-level number
mixes the two. This module fixes the two components from the canopy height
model at the first post-disturbance date (and a pre-disturbance reference when
one exists), then reports what happened inside each component at every later
date.

Components are held constant through time. A later row therefore describes the
same pixels the disturbance sorted, which is what a recovery trajectory needs:
regrowth in the openings, and survival or loss in the residual canopy.

Beyond the two components tracked independently, three descriptions carry
information about how they interact: connected gap patches and their size,
distance from every gap pixel to the nearest residual canopy, and the split of
whole-stand height variance into within-component and between-component parts.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage

from csdv_core.zonal.mask import StandWindow, read_stand_array
from csdv_core.zonal.pixel import CANOPY_HEIGHT_THRESHOLD_M, MIN_CANOPY_PIXELS

logger = logging.getLogger(__name__)

OUTSIDE = 0
RESIDUAL = 1
NEW_GAP = 2
PREEXISTING_GAP = 3
WHOLE = "whole"

COMPONENT_NAMES: dict[int, str] = {
    RESIDUAL: "residual",
    NEW_GAP: "new_gap",
    PREEXISTING_GAP: "preexisting_gap",
}

__all__ = [
    "COMPONENT_NAMES",
    "NEW_GAP",
    "OUTSIDE",
    "PREEXISTING_GAP",
    "RESIDUAL",
    "WHOLE",
    "ComponentSeries",
    "classify_components",
    "component_series",
    "component_stats",
    "crowns_by_component",
    "distance_to_residual",
    "gap_patches",
    "height_by_distance",
    "patch_table",
    "variance_decomposition",
]


def classify_components(
    chm_post: np.ndarray,
    mask: np.ndarray,
    *,
    chm_ref: np.ndarray | None = None,
    threshold_m: float = CANOPY_HEIGHT_THRESHOLD_M,
) -> np.ndarray:
    """Sort in-stand pixels into residual canopy and gaps.

    With a pre-disturbance reference, residual canopy is at or above the
    threshold on both dates, a new gap is above on the reference and below
    afterwards, and a pre-existing gap was already below on the reference.
    Without a reference the first post-disturbance date alone decides: canopy
    is residual, everything else is a new gap.

    Args:
        chm_post: Canopy height at the first post-disturbance date, metres.
        mask: Boolean in-stand mask of the same shape.
        chm_ref: Canopy height at the last pre-disturbance date, or None.
        threshold_m: Canopy height threshold.

    Returns:
        uint8 array of component codes, ``OUTSIDE`` beyond the stand or where
        any input is NaN.
    """
    post = np.asarray(chm_post, dtype=np.float32)
    if post.shape != mask.shape:
        raise ValueError(f"Array shape {post.shape} does not match mask {mask.shape}")
    valid = mask & np.isfinite(post)
    post_canopy = post >= threshold_m
    out = np.zeros(post.shape, dtype=np.uint8)
    if chm_ref is None:
        out[valid & post_canopy] = RESIDUAL
        out[valid & ~post_canopy] = NEW_GAP
        return out
    ref = np.asarray(chm_ref, dtype=np.float32)
    if ref.shape != post.shape:
        raise ValueError(f"Reference shape {ref.shape} does not match {post.shape}")
    valid &= np.isfinite(ref)
    ref_canopy = ref >= threshold_m
    out[valid & ref_canopy & post_canopy] = RESIDUAL
    out[valid & ref_canopy & ~post_canopy] = NEW_GAP
    out[valid & ~ref_canopy] = PREEXISTING_GAP
    return out


def _nan_stats() -> dict[str, float]:
    keys = (
        "canopy_fraction",
        "height_mean",
        "height_median",
        "height_p90",
        "height_cv",
        "canopy_height_median",
        "canopy_height_p90",
        "canopy_height_cv",
    )
    return {key: float("nan") for key in keys}


def component_stats(
    chm: np.ndarray,
    components: np.ndarray,
    code: int | str,
    *,
    threshold_m: float = CANOPY_HEIGHT_THRESHOLD_M,
    min_pixels: int = MIN_CANOPY_PIXELS,
) -> dict[str, float]:
    """Summarise canopy height inside one component at one date.

    ``height_*`` statistics use every valid pixel in the component, so a gap
    starts near zero and rises as it fills. ``canopy_height_*`` statistics use
    only pixels at or above ``threshold_m``, matching
    :func:`csdv_core.zonal.pixel.height_stats`, so the ``WHOLE`` row reproduces
    the stand metrics table.

    Args:
        chm: Canopy height at the date of interest.
        components: Codes from :func:`classify_components`.
        code: A component code, or ``WHOLE`` for every in-stand pixel.
        threshold_m: Canopy height threshold.
        min_pixels: Below this many valid pixels every statistic is NaN.

    Returns:
        A dictionary with ``n_pixels``, ``canopy_fraction`` and the height
        statistics.
    """
    arr = np.asarray(chm, dtype=np.float32)
    if arr.shape != components.shape:
        raise ValueError(f"Array shape {arr.shape} does not match {components.shape}")
    member = components > OUTSIDE if code == WHOLE else components == code
    sel = member & np.isfinite(arr)
    values = arr[sel]
    result: dict[str, float] = {"n_pixels": float(values.size)}
    if values.size < min_pixels:
        result.update(_nan_stats())
        return result
    mean = float(values.mean())
    result["canopy_fraction"] = float(np.mean(values >= threshold_m))
    result["height_mean"] = mean
    result["height_median"] = float(np.median(values))
    result["height_p90"] = float(np.percentile(values, 90))
    result["height_cv"] = float(values.std() / mean) if mean > 0 else float("nan")
    canopy = values[values >= threshold_m]
    if canopy.size < min_pixels:
        result["canopy_height_median"] = float("nan")
        result["canopy_height_p90"] = float("nan")
        result["canopy_height_cv"] = float("nan")
        return result
    canopy_mean = float(canopy.mean())
    result["canopy_height_median"] = float(np.median(canopy))
    result["canopy_height_p90"] = float(np.percentile(canopy, 90))
    result["canopy_height_cv"] = float(canopy.std() / canopy_mean)
    return result


@dataclass(frozen=True)
class ComponentSeries:
    """Component codes for one stand and the per-date statistics inside them.

    Attributes:
        stand_id: Stand identifier.
        components: Code array on the stand window.
        window: The stand window the arrays share.
        chm_by_year: Canopy height arrays on the same window, by year.
        frame: One row per year per component (plus ``WHOLE``).
    """

    stand_id: str
    components: np.ndarray
    window: StandWindow
    chm_by_year: dict[int, np.ndarray]
    frame: pd.DataFrame


def component_series(
    chm_paths: Mapping[int, Path | str],
    geometry: Any,
    *,
    first_post_year: int,
    ref_year: int | None = None,
    threshold_m: float = CANOPY_HEIGHT_THRESHOLD_M,
    scale: float = 1.0,
    stand_id: str = "",
) -> ComponentSeries:
    """Fix the components for a stand and summarise every date inside them.

    Args:
        chm_paths: Canopy height model path by year. Every raster must share
            one grid.
        geometry: Stand geometry in the raster CRS.
        first_post_year: Year whose canopy height sorts the components.
        ref_year: Pre-disturbance year, or None when the record starts after
            the disturbance.
        threshold_m: Canopy height threshold.
        scale: Read scale for the canopy height model.
        stand_id: Carried into the frame.
    """
    if first_post_year not in chm_paths:
        raise KeyError(f"first_post_year {first_post_year} not in chm_paths")
    if ref_year is not None and ref_year not in chm_paths:
        raise KeyError(f"ref_year {ref_year} not in chm_paths")

    chm_by_year: dict[int, np.ndarray] = {}
    window: StandWindow | None = None
    for year in sorted(chm_paths):
        arr, sw = read_stand_array(
            chm_paths[year], geometry, scale=scale, stand_id=stand_id
        )
        if window is None:
            window = sw
        elif arr.shape != window.mask.shape:
            raise ValueError(
                f"{year} block {arr.shape} differs from {window.mask.shape}; "
                "the rasters do not share a grid"
            )
        chm_by_year[year] = arr
    assert window is not None

    components = classify_components(
        chm_by_year[first_post_year],
        window.mask,
        chm_ref=None if ref_year is None else chm_by_year[ref_year],
        threshold_m=threshold_m,
    )
    codes: list[int | str] = [WHOLE] + [
        code for code in COMPONENT_NAMES if np.any(components == code)
    ]
    rows = []
    for year, arr in chm_by_year.items():
        for code in codes:
            stats = component_stats(arr, components, code, threshold_m=threshold_m)
            name = WHOLE if code == WHOLE else COMPONENT_NAMES[int(code)]
            rows.append(
                {"stand_id": stand_id, "year": year, "component": name, **stats}
            )
    frame = pd.DataFrame(rows)
    logger.info(
        "%s: %d pixels sorted into %s",
        stand_id or "stand",
        int((components > OUTSIDE).sum()),
        {COMPONENT_NAMES[c]: int((components == c).sum()) for c in COMPONENT_NAMES},
    )
    return ComponentSeries(stand_id, components, window, chm_by_year, frame)


def gap_patches(
    components: np.ndarray,
    *,
    code: int = NEW_GAP,
    min_area_px: int = 1,
) -> np.ndarray:
    """Label connected patches of one component, 8-connected.

    Returns:
        int32 array, zero where the pixel is not in a kept patch.
    """
    structure = np.ones((3, 3), dtype=bool)
    labels, n = ndimage.label(components == code, structure=structure)
    if n == 0 or min_area_px <= 1:
        return labels.astype(np.int32)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_area_px
    keep[0] = False
    relabel = np.zeros_like(sizes, dtype=np.int32)
    relabel[keep] = np.arange(1, int(keep.sum()) + 1)
    return relabel[labels]


def patch_table(
    labels: np.ndarray,
    chm_by_year: Mapping[int, np.ndarray],
    *,
    pixel_size_m: float,
    threshold_m: float = CANOPY_HEIGHT_THRESHOLD_M,
) -> pd.DataFrame:
    """One row per patch per year: area, fill fraction, and height summary."""
    ids = np.unique(labels[labels > 0])
    rows = []
    for year, chm in chm_by_year.items():
        arr = np.asarray(chm, dtype=np.float32)
        for patch in ids:
            sel = (labels == patch) & np.isfinite(arr)
            values = arr[sel]
            if values.size == 0:
                continue
            rows.append(
                {
                    "patch_id": int(patch),
                    "year": year,
                    "n_pixels": int(values.size),
                    "area_m2": float(values.size * pixel_size_m**2),
                    "canopy_fraction": float(np.mean(values >= threshold_m)),
                    "height_median": float(np.median(values)),
                    "height_p90": float(np.percentile(values, 90)),
                }
            )
    return pd.DataFrame(rows)


def distance_to_residual(components: np.ndarray, pixel_size_m: float) -> np.ndarray:
    """Euclidean distance in metres from every pixel to the nearest residual canopy.

    Zero on residual pixels. Where the stand has no residual canopy every
    distance is infinite.
    """
    residual = components == RESIDUAL
    if not residual.any():
        return np.full(components.shape, np.inf, dtype=np.float32)
    dist = ndimage.distance_transform_edt(~residual)
    return (dist * pixel_size_m).astype(np.float32)


def height_by_distance(
    chm: np.ndarray,
    components: np.ndarray,
    distance: np.ndarray,
    *,
    bins_m: Sequence[float],
    code: int = NEW_GAP,
    threshold_m: float = CANOPY_HEIGHT_THRESHOLD_M,
    min_pixels: int = MIN_CANOPY_PIXELS,
) -> pd.DataFrame:
    """Bin one component's pixels by distance to residual canopy.

    Returns:
        One row per bin with ``bin_lo``, ``bin_hi``, ``n_pixels``,
        ``canopy_fraction``, ``height_median`` and ``height_p90``. Bins with
        fewer than ``min_pixels`` pixels report NaN statistics.
    """
    arr = np.asarray(chm, dtype=np.float32)
    sel = (components == code) & np.isfinite(arr) & np.isfinite(distance)
    edges = np.asarray(bins_m, dtype=float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = sel & (distance >= lo) & (distance < hi)
        values = arr[in_bin]
        row: dict[str, float] = {"bin_lo": lo, "bin_hi": hi, "n_pixels": values.size}
        if values.size < min_pixels:
            row.update(canopy_fraction=np.nan, height_median=np.nan, height_p90=np.nan)
        else:
            row.update(
                canopy_fraction=float(np.mean(values >= threshold_m)),
                height_median=float(np.median(values)),
                height_p90=float(np.percentile(values, 90)),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def variance_decomposition(
    chm: np.ndarray,
    components: np.ndarray,
    *,
    codes: Sequence[int] = (RESIDUAL, NEW_GAP, PREEXISTING_GAP),
) -> dict[str, float]:
    """Split whole-stand height variance into within and between components.

    By the law of total variance, the population variance of height over the
    stand equals the pixel-weighted mean of the within-component variances
    plus the variance of the component means. Each within term is reported
    already weighted, so the parts sum to ``total``.

    Returns:
        ``total``, ``between``, ``within_<name>`` for each code present,
        ``mean`` and ``cv`` of the whole stand.
    """
    arr = np.asarray(chm, dtype=np.float32)
    valid = (components > OUTSIDE) & np.isfinite(arr)
    values = arr[valid]
    out: dict[str, float] = {}
    if values.size == 0:
        return out
    mean = float(values.mean())
    total = float(values.var())
    out["mean"] = mean
    out["total"] = total
    out["cv"] = float(np.sqrt(total) / mean) if mean > 0 else float("nan")
    n = float(values.size)
    between = 0.0
    for code in codes:
        sel = valid & (components == code)
        part = arr[sel]
        if part.size == 0:
            continue
        weight = part.size / n
        out[f"within_{COMPONENT_NAMES[code]}"] = float(weight * part.var())
        between += weight * (float(part.mean()) - mean) ** 2
    out["between"] = float(between)
    return out


def crowns_by_component(
    crowns: Any,
    components: np.ndarray,
    transform: Any,
) -> np.ndarray:
    """Assign each crown the component code under its centroid.

    Args:
        crowns: GeoDataFrame of crown polygons in the raster CRS.
        components: Code array on a block read with ``transform``.
        transform: Affine of that block.

    Returns:
        uint8 array of codes, ``OUTSIDE`` for centroids beyond the block.
    """
    if len(crowns) == 0:
        return np.zeros(0, dtype=np.uint8)
    centroids = crowns.geometry.centroid
    inverse = ~transform
    cols, rows = inverse * (centroids.x.to_numpy(), centroids.y.to_numpy())
    rows = np.floor(rows).astype(int)
    cols = np.floor(cols).astype(int)
    inside = (
        (rows >= 0)
        & (rows < components.shape[0])
        & (cols >= 0)
        & (cols < components.shape[1])
    )
    out = np.zeros(len(crowns), dtype=np.uint8)
    out[inside] = components[rows[inside], cols[inside]]
    return out
