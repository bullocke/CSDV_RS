"""Refine coarse canopy-loss patches into 0.6 m openings and harvest-unit envelopes.

A rule run on 9.6 m cells outlines a disturbance as a staircase of cells. This
module redraws it from a pair of canopy height models at their native pixel
size. *Loss pixels* were canopy before and are open after; a 3 x 3 maximum
filter on the later CHM keeps one-pixel misregistration from reading as loss.
The *opening footprint* is the loss pixels after a small morphological opening.
The *envelope* closes the gaps between openings, fills holes and smooths the
outline, which approximates how interpreters draw a harvest unit (residual
trees included).

Inputs: two CHM arrays in metres (NaN = nodata) on one grid, plus optional
seed and forest masks. Outputs: boolean masks, shapely geometries and a
:class:`RefineResult` with QA numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import rasterio.features
from affine import Affine
from scipy import ndimage
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


@dataclass(frozen=True)
class RefineParams:
    """Thresholds for :func:`refine` (heights in m, areas in m2, radii in m)."""

    canopy_m: float = 10.0
    open_m: float = 5.0
    drop_m: float = 8.0
    tolerance_px: int = 1
    open_radius_m: float = 1.0
    min_opening_m2: float = 25.0
    close_radius_m: float = 8.0
    fill_holes_m2: float = 2500.0
    min_envelope_m2: float = 1000.0
    smooth_m: float = 3.0
    simplify_m: float = 0.6


@dataclass
class RefineResult:
    """Masks, outlines and QA numbers for one refined patch."""

    loss: np.ndarray
    openings: np.ndarray
    envelope: np.ndarray
    opening_geom: BaseGeometry
    envelope_geom: BaseGeometry
    stats: dict[str, float] = field(default_factory=dict)


def loss_pixels(
    h0: np.ndarray,
    h1: np.ndarray,
    *,
    canopy_m: float = 10.0,
    open_m: float = 5.0,
    drop_m: float = 8.0,
    tolerance_px: int = 1,
) -> np.ndarray:
    """Pixels that were canopy at the first date and lost it by the second.

    A pixel counts when ``h0 >= canopy_m`` and the highest later height within
    ``tolerance_px`` is below ``open_m`` or at least ``drop_m`` lower than
    ``h0``. Taking the neighbourhood maximum of ``h1`` means a crown that only
    shifted by a pixel between dates is not read as loss.
    """
    size = 2 * tolerance_px + 1
    h1max = ndimage.maximum_filter(np.nan_to_num(h1, nan=np.inf), size=size)
    ok = np.isfinite(h0) & np.isfinite(h1max)
    h0f = np.nan_to_num(h0)
    return ok & (h0f >= canopy_m) & ((h1max < open_m) | (h1max - h0f <= -drop_m))


def _disk_open(mask: np.ndarray, radius_px: float) -> np.ndarray:
    """Morphological opening with a Euclidean disk (erode, then dilate)."""
    if radius_px <= 0 or not mask.any():
        return mask.copy()
    eroded = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1] > radius_px
    return (
        ndimage.distance_transform_edt(~eroded) <= radius_px if eroded.any() else eroded
    )


def _disk_close(mask: np.ndarray, radius_px: float) -> np.ndarray:
    """Morphological closing with a Euclidean disk (dilate, then erode)."""
    if radius_px <= 0 or not mask.any():
        return mask.copy()
    pad = int(np.ceil(radius_px)) + 1
    m = np.pad(mask, pad)
    dilated = ndimage.distance_transform_edt(~m) <= radius_px
    closed = ndimage.distance_transform_edt(dilated) > radius_px
    return closed[pad:-pad, pad:-pad]


def _drop_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0 or min_px <= 1:
        return mask.copy()
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    keep = np.concatenate([[False], sizes >= min_px])
    return keep[lab]


def _fill_small_holes(mask: np.ndarray, max_px: int) -> np.ndarray:
    holes = ~mask
    lab, n = ndimage.label(holes)
    if n == 0:
        return mask.copy()
    sizes = ndimage.sum(holes, lab, index=np.arange(1, n + 1))
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    small = np.concatenate([[False], sizes <= max_px])
    small[border] = False
    return mask | small[lab]


def opening_footprint(
    loss: np.ndarray,
    pixel_m: float,
    *,
    open_radius_m: float = 1.0,
    min_area_m2: float = 25.0,
) -> np.ndarray:
    """Loss pixels cleaned of crown-edge slivers and specks."""
    opened = _disk_open(loss, open_radius_m / pixel_m)
    return _drop_small(opened, int(round(min_area_m2 / pixel_m**2)))


def envelope_mask(
    openings: np.ndarray,
    pixel_m: float,
    *,
    close_radius_m: float = 8.0,
    fill_holes_m2: float = 2500.0,
    min_area_m2: float = 1000.0,
    seed: np.ndarray | None = None,
    clip: np.ndarray | None = None,
) -> np.ndarray:
    """Close the gaps between openings into a unit outline.

    Args:
        openings: Opening footprint.
        pixel_m: Pixel size.
        close_radius_m: Gaps up to about twice this width are bridged.
        fill_holes_m2: Holes up to this area are filled (residual trees).
        min_area_m2: Components below this area are dropped.
        seed: Keep only components that touch this mask (e.g. the coarse patch).
        clip: Restrict the result to this mask (e.g. forest at the first date).
    """
    env = _disk_close(openings, close_radius_m / pixel_m)
    env = _fill_small_holes(env, int(round(fill_holes_m2 / pixel_m**2)))
    if clip is not None:
        env &= clip
    env = _drop_small(env, int(round(min_area_m2 / pixel_m**2)))
    if seed is not None and env.any():
        lab, n = ndimage.label(env, structure=np.ones((3, 3)))
        hit = np.unique(lab[seed & env])
        env = np.isin(lab, hit[hit > 0])
    return env


def polygonize(
    mask: np.ndarray,
    transform: Affine,
    *,
    smooth_m: float = 0.0,
    simplify_m: float = 0.0,
) -> BaseGeometry:
    """Union of the mask's polygons, optionally smoothed and simplified."""
    parts = [
        shape(g)
        for g, v in rasterio.features.shapes(
            mask.astype("uint8"), mask=mask, transform=transform
        )
        if v == 1
    ]
    geom = (
        unary_union(parts) if parts else shape({"type": "Polygon", "coordinates": []})
    )
    if smooth_m > 0 and not geom.is_empty:
        geom = geom.buffer(smooth_m, join_style="round").buffer(
            -smooth_m, join_style="round"
        )
    if simplify_m > 0 and not geom.is_empty:
        geom = geom.simplify(simplify_m, preserve_topology=True)
    return geom


def refine(
    h0: np.ndarray,
    h1: np.ndarray,
    transform: Affine,
    *,
    params: RefineParams | None = None,
    seed: np.ndarray | None = None,
    search: np.ndarray | None = None,
    clip: np.ndarray | None = None,
) -> RefineResult:
    """Redraw one canopy-loss patch at pixel size.

    Args:
        h0, h1: CHM before and after (m, NaN nodata), same grid.
        transform: Affine transform of that grid.
        params: Thresholds.
        seed: Coarse patch mask. Envelope components must touch it.
        search: Pixels allowed to count as loss (e.g. the patch buffered,
            minus mapped polygons). Defaults to the whole window.
        clip: Mask the envelope is restricted to (e.g. forest at the first date).
    """
    params = params or RefineParams()
    px = abs(transform.a)
    loss = loss_pixels(
        h0,
        h1,
        canopy_m=params.canopy_m,
        open_m=params.open_m,
        drop_m=params.drop_m,
        tolerance_px=params.tolerance_px,
    )
    if search is not None:
        loss &= search
    openings = opening_footprint(
        loss, px, open_radius_m=params.open_radius_m, min_area_m2=params.min_opening_m2
    )
    env = envelope_mask(
        openings,
        px,
        close_radius_m=params.close_radius_m,
        fill_holes_m2=params.fill_holes_m2,
        min_area_m2=params.min_envelope_m2,
        seed=seed,
        clip=None if clip is None else (clip | openings),
    )
    if search is not None:
        env &= search
    openings &= env
    env_geom = polygonize(
        env, transform, smooth_m=params.smooth_m, simplify_m=params.simplify_m
    )
    open_geom = polygonize(openings, transform)
    a_px = px**2
    canopy0 = np.nan_to_num(h0) >= params.canopy_m
    stats = {
        "envelope_ha": float(env_geom.area / 1e4),
        "openings_ha": float(openings.sum() * a_px / 1e4),
        "opening_share": float(openings.sum() / max(env.sum(), 1)),
        "canopy0_share": float((canopy0 & env).sum() / max(env.sum(), 1)),
        "n_openings": int(ndimage.label(openings, structure=np.ones((3, 3)))[1]),
    }
    if seed is not None:
        inter = (env & seed).sum()
        stats["iou_seed"] = float(inter / max((env | seed).sum(), 1))
    return RefineResult(loss, openings, env, open_geom, env_geom, stats)
