"""csdv_core.segmentation.chm_watershed — Python crown segmentation.

Seeded watershed segmentation of a canopy height model, following
Dalponte and Coomes (2016) closely enough to compare against the lidR
implementation in :mod:`csdv_core.segmentation.lidr_bridge`.

Algorithm:

    1. Smooth the CHM with a mean filter of a radius given in metres.
    2. Mask pixels below ``min_height_m``.
    3. Locate tree tops. A pixel is a tree top when it is the highest pixel
       inside a circle whose diameter comes from that pixel's own height,
       which is what lidR's ``lmf`` does.
    4. Watershed the inverted CHM from those tree tops.
    5. Bound each crown's extent. Drop pixels below ``th_cr`` times the height
       at the tree top, keep the part still joined to the tree top, then apply
       the radius ceiling.
    6. Vectorize, and record the tree top position and height alongside the
       area and diameter.

Two things here were wrong before and are worth stating plainly, because both
inflated crown size by a large factor.

``skimage.peak_local_max(min_distance=d)`` takes a minimum *separation*, while
lidR's ``lmf(ws=...)`` takes a window *diameter*. The old code passed the
window diameter as the separation, so it spread tree tops twice as far apart as
the equation intended. It also evaluated the window once for the whole raster
at the scene mean height, so seed placement carried no height dependence at
all. Both are fixed here, and because the window is now per-pixel, blocks no
longer need a shared scene mean to stay comparable.

Nothing bounded crown extent, so the watershed assigned every canopy pixel to
some tree top and mean crown area was pinned at ``10000 / density``. ``th_cr``
is the rule that releases it.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import geopandas as gpd
import numpy as np
from rasterio.features import shapes as rio_shapes
from scipy import ndimage as ndi
from shapely.geometry import shape as shapely_shape
from skimage.segmentation import watershed

from csdv_core.preprocess.chm import mask_below, smooth_chm
from csdv_core.segmentation.params import (
    DEFAULT_PARAMS,
    SegmentationParams,
    WindowFunction,
)

logger = logging.getLogger(__name__)

#: Columns every crown frame carries.
CROWN_COLUMNS = (
    "segment_id",
    "area_m2",
    "crown_diam_m",
    "apex_h_m",
    "seed_x",
    "seed_y",
    "capped_frac",
)

__all__ = [
    "CROWN_COLUMNS",
    "segment_crowns",
    "locate_seeds",
    "smooth_kernel_px",
]


def smooth_kernel_px(smooth_radius_m: float, pixel_size_m: float) -> int:
    """Odd mean-filter side length in pixels for a radius given in metres.

    Returns 1, a no-op, for a zero radius. Keeping the radius in metres is what
    makes one parameter set behave the same on a 0.6 m and a 1 m CHM.
    """
    if smooth_radius_m <= 0:
        return 1
    r = int(round(float(smooth_radius_m) / float(pixel_size_m)))
    return 2 * max(r, 0) + 1


def _disk(radius_px: int) -> np.ndarray:
    """Boolean disk footprint, matching lidR's circular search window."""
    y, x = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    return (x * x + y * y) <= radius_px * radius_px + 1e-9


def _label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """``scipy.ndimage.label`` with a return type the checker can follow."""
    labelled, count = ndi.label(mask)  # type: ignore[misc]
    return np.asarray(labelled), int(count)


def locate_seeds(
    smoothed: np.ndarray,
    valid: np.ndarray,
    pixel_size_m: float,
    window: WindowFunction,
    *,
    min_separation_m: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Find tree tops with a height-dependent search window.

    A pixel is a tree top when no pixel within ``window(h)/2`` metres of it is
    higher, where ``h`` is that pixel's own smoothed height. This is lidR's
    ``lmf`` rule with a circular window.

    The window radius is quantized to whole pixels, which lets one maximum
    filter per distinct radius answer the question for every pixel at once.
    The number of distinct radii is small, so this stays cheap.

    Args:
        smoothed: Smoothed CHM in metres, NaN outside the canopy.
        valid: True where ``smoothed`` is canopy.
        pixel_size_m: Raster pixel size in metres.
        window: Search window diameter as a function of height.
        min_separation_m: Floor on separation between tree tops, so a flat
            crown top does not become several trees.

    Returns:
        ``(rows, cols)`` index arrays of the tree top pixels.
    """
    if not valid.any():
        empty = np.zeros(0, dtype=np.intp)
        return empty, empty

    heights = np.where(valid, smoothed, -np.inf).astype("float32")

    # Radius in whole pixels, from each pixel's own height. Diameter over two.
    radius_px = window(np.where(valid, smoothed, 0.0)) / 2.0 / float(pixel_size_m)
    floor_px = max(1, int(round(float(min_separation_m) / float(pixel_size_m))))
    radius_px = np.clip(np.rint(radius_px), floor_px, None).astype(np.int16)
    radius_px[~valid] = 0

    is_top = np.zeros(smoothed.shape, dtype=bool)
    for r in np.unique(radius_px[valid]):
        r = int(r)
        if r < 1:
            continue
        band = valid & (radius_px == r)
        if not band.any():
            continue
        local_max = ndi.maximum_filter(
            heights, footprint=_disk(r), mode="constant", cval=-np.inf
        )
        is_top |= band & (heights >= local_max)

    if not is_top.any():
        empty = np.zeros(0, dtype=np.intp)
        return empty, empty

    # A flat top yields a plateau of tied maxima. Collapse each plateau to one
    # seed at its centre of mass, matching the old ndi.label collapse.
    groups, n_groups = _label(is_top)
    if n_groups == 0:
        empty = np.zeros(0, dtype=np.intp)
        return empty, empty
    centres = ndi.center_of_mass(is_top, groups, np.arange(1, n_groups + 1))
    coords = np.rint(np.asarray(centres, dtype="float64")).astype(np.intp)
    rows = np.clip(coords[:, 0], 0, smoothed.shape[0] - 1)
    cols = np.clip(coords[:, 1], 0, smoothed.shape[1] - 1)

    # The centre of mass of an L-shaped plateau can land off the plateau. Pull
    # any such seed back onto its own group.
    off = groups[rows, cols] != np.arange(1, n_groups + 1)
    if off.any():
        fallback = ndi.maximum_position(
            heights, groups, np.arange(1, n_groups + 1)[off]
        )
        fallback = np.asarray(fallback, dtype=np.intp).reshape(-1, 2)
        rows[off] = fallback[:, 0]
        cols[off] = fallback[:, 1]
    return rows, cols


def _empty_crowns(crs: Any) -> gpd.GeoDataFrame:
    """An empty crown frame carrying the full schema."""
    return gpd.GeoDataFrame(
        {c: [] for c in CROWN_COLUMNS},
        geometry=[],
        crs=crs,
    )


def _bound_extent(
    labels: np.ndarray,
    smoothed: np.ndarray,
    seed_rows: np.ndarray,
    seed_cols: np.ndarray,
    apex: np.ndarray,
    *,
    th_cr: float,
    max_radius_px: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim each crown to its own canopy, and cap its radius.

    Returns the trimmed label array and, per label, the fraction of area the
    radius ceiling removed. A crown the ceiling shaped carries a censored
    diameter, which is a lower bound rather than a measurement, so any
    allometric fit has to be able to find and exclude it. The fraction rather
    than a flag separates a crown merely grazed by the ceiling from one the
    ceiling defined.
    """
    n_labels = int(labels.max())
    capped = np.zeros(n_labels + 1, dtype="float32")
    if n_labels == 0:
        return labels, capped[1:]

    keep = labels > 0

    if th_cr > 0:
        # apex is indexed 1..n_labels; prepend a dummy for label 0.
        apex_by_label = np.concatenate([[np.inf], apex])
        floor = th_cr * apex_by_label[labels]
        keep &= smoothed >= floor

    if max_radius_px is not None:
        rr, cc = np.indices(labels.shape, dtype="float32")
        sr = np.concatenate([[0.0], seed_rows.astype("float32")])
        sc = np.concatenate([[0.0], seed_cols.astype("float32")])
        d2 = (rr - sr[labels]) ** 2 + (cc - sc[labels]) ** 2
        within = d2 <= (max_radius_px * max_radius_px)
        removed = keep & ~within
        if removed.any():
            bins = np.arange(n_labels + 2)
            lost = np.histogram(labels[removed], bins=bins)[0]
            total = np.histogram(labels[keep], bins=bins)[0]
            with np.errstate(invalid="ignore", divide="ignore"):
                capped = np.where(total > 0, lost / np.maximum(total, 1), 0.0).astype(
                    "float32"
                )
        keep &= within

    if th_cr <= 0 and max_radius_px is None:
        return labels, capped[1:]

    # Trimming can disconnect a crown from its own tree top. Keep only the
    # piece still joined to the seed. Components are found on the trimmed mask,
    # then paired with the original label so two crowns that touch are never
    # merged into one component.
    trimmed = np.where(keep, labels, 0)
    comp, _ = _label(keep)
    pair = comp.astype(np.int64) * (n_labels + 1) + trimmed
    seed_pair = pair[seed_rows, seed_cols]
    # A seed whose own pixel was trimmed away cannot happen for th_cr < 1, but
    # guard anyway so a bad parameter fails loudly rather than silently.
    seed_pair = seed_pair[trimmed[seed_rows, seed_cols] > 0]
    connected = np.isin(pair, seed_pair)
    return np.where(connected, trimmed, 0), capped[1:]


def segment_crowns(
    chm: np.ndarray,
    transform: Any,
    crs: Any,
    *,
    params: SegmentationParams = DEFAULT_PARAMS,
) -> gpd.GeoDataFrame:
    """Segment individual tree crowns from a canopy height model.

    Args:
        chm: 2-D CHM array in metres. NaN allowed for nodata.
        transform: Affine transform mapping pixel indices to ``crs``.
        crs: CRS for the output GeoDataFrame.
        params: Segmentation configuration. Every value is in metres, so one
            set transfers between resolutions unchanged.

    Returns:
        GeoDataFrame with ``segment_id``, ``area_m2``, ``crown_diam_m``,
        ``apex_h_m``, ``seed_x``, ``seed_y`` and ``capped``.

        ``crown_diam_m`` is the area-equivalent circle diameter,
        ``2 * sqrt(area / pi)``. ``apex_h_m`` is the smoothed height at the
        tree top, never the maximum over the segment. The maximum over a
        segment grows with segment size, so using it would build a
        height-to-size relationship into the measurement.
    """
    px = float(abs(transform.a))
    kernel = smooth_kernel_px(params.smooth_radius_m, px)
    smoothed = smooth_chm(np.asarray(chm, dtype="float32"), kernel=kernel)
    masked = mask_below(smoothed, threshold_m=params.min_height_m)
    valid = ~np.isnan(masked)
    if not valid.any():
        logger.warning("segment_crowns: no pixels above %.2f m", params.min_height_m)
        return _empty_crowns(crs)

    seed_rows, seed_cols = locate_seeds(
        masked,
        valid,
        px,
        params.window,
        min_separation_m=params.min_separation_m,
    )
    if seed_rows.size == 0:
        logger.warning("segment_crowns: no tree tops detected")
        return _empty_crowns(crs)

    markers = np.zeros(masked.shape, dtype="int32")
    markers[seed_rows, seed_cols] = np.arange(1, seed_rows.size + 1)

    inv = np.where(valid, -masked, np.inf).astype("float32")
    labels = watershed(inv, markers=markers, mask=valid)

    apex = masked[seed_rows, seed_cols].astype("float64")
    max_radius_px = (
        None
        if params.max_crown_radius_m is None
        else float(params.max_crown_radius_m) / px
    )
    labels, capped = _bound_extent(
        labels,
        masked,
        seed_rows,
        seed_cols,
        apex,
        th_cr=float(params.th_cr),
        max_radius_px=max_radius_px,
    )

    # One connected component per label, so rio_shapes emits one polygon per
    # crown and no dissolve is needed. Connectivity must match ndi.label's
    # default cross structure.
    polys: list[Any] = []
    seg_ids: list[int] = []
    for geom, val in rio_shapes(
        labels.astype("int32"), mask=labels > 0, transform=transform, connectivity=4
    ):
        polys.append(shapely_shape(geom))
        seg_ids.append(int(val))
    if not polys:
        return _empty_crowns(crs)

    gdf = gpd.GeoDataFrame({"segment_id": seg_ids}, geometry=polys, crs=crs)
    if gdf["segment_id"].duplicated().any():
        gdf = gdf.dissolve(by="segment_id", as_index=False)

    idx = gdf["segment_id"].to_numpy() - 1
    seed_x, seed_y = transform * (seed_cols + 0.5, seed_rows + 0.5)
    gdf["area_m2"] = gdf.geometry.area
    gdf["apex_h_m"] = apex[idx]
    gdf["seed_x"] = np.asarray(seed_x, dtype="float64")[idx]
    gdf["seed_y"] = np.asarray(seed_y, dtype="float64")[idx]
    gdf["capped_frac"] = capped[idx]
    gdf = gdf[gdf["area_m2"] >= float(params.min_crown_area_m2)].copy()
    if gdf.empty:
        return _empty_crowns(crs)
    gdf["crown_diam_m"] = 2.0 * np.sqrt(gdf["area_m2"].to_numpy() / math.pi)
    gdf.reset_index(drop=True, inplace=True)
    gdf = gdf[list(CROWN_COLUMNS) + ["geometry"]]

    logger.info(
        "segment_crowns: %d crowns, mean diam %.2f m, %.1f%% shaped by the cap",
        len(gdf),
        float(gdf["crown_diam_m"].mean()),
        100.0 * float((gdf["capped_frac"] > 0.05).mean()),
    )
    return gdf
