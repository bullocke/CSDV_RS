"""csdv_core.segmentation.chm_watershed — Python crown segmentation.

Dalponte-approximate watershed segmentation of a canopy height model.
For full Dalponte-Coomes (2016) seed expansion use the R lidR bridge in
:mod:`csdv_core.segmentation.lidr_bridge`. This Python path is the
canonical Colab fallback and is used by the integration test against the
PoC reference.

Algorithm:

    1. Smooth the CHM with a 3x3 mean kernel.
    2. Mask pixels below ``min_height_m``.
    3. Detect tree tops as local maxima (variable footprint scaled by
       canopy height: ``ws = clip(2 + 0.5 * h, 3, 12)`` m, expressed in
       pixels and applied as the minimum peak separation).
    4. Watershed-segment the inverted smoothed CHM, seeded by the tree
       tops, restricted to the masked CHM.
    5. Vectorize segment IDs and compute ``area_m2`` and
       ``crown_diam_m = 2 * sqrt(area / pi)``.
"""

from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd
import numpy as np
from rasterio.features import shapes as rio_shapes
from scipy import ndimage as ndi
from shapely.geometry import shape as shapely_shape
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from csdv_core.preprocess.chm import mask_below, smooth_chm

logger = logging.getLogger(__name__)


def _peak_distance_px(
    min_distance_m: float, pixel_size_m: float, mean_height_m: float
) -> int:
    """Compute the local-max footprint in pixels.

    Uses ``ws = clip(2 + 0.5 * h, 3, 12)`` m from the legacy R script,
    but bounded below by ``min_distance_m``.
    """
    ws_m = max(min_distance_m, min(12.0, 2.0 + 0.5 * mean_height_m))
    return max(1, int(round(ws_m / pixel_size_m)))


def segment_crowns(
    chm: np.ndarray,
    transform: Any,
    crs: Any,
    *,
    min_height_m: float = 2.0,
    smooth_kernel: int = 3,
    min_peak_distance_m: float = 3.0,
    min_crown_area_m2: float = 1.0,
    mean_height_m: float | None = None,
) -> gpd.GeoDataFrame:
    """Segment individual tree crowns from a CHM.

    Args:
        chm: 2-D CHM array in metres. NaN allowed.
        transform: Affine transform mapping pixel indices to ``crs``.
        crs: CRS string or rasterio CRS for the output GeoDataFrame.
        min_height_m: Pixels below this height are excluded.
        smooth_kernel: Side length of the mean-filter kernel.
        min_peak_distance_m: Minimum separation between local maxima.
        min_crown_area_m2: Polygons smaller than this are dropped.
        mean_height_m: Canopy height used to size the local-maximum footprint.
            Defaults to the mean of the masked input, which makes the footprint
            depend on how much surrounding forest the caller happened to pass
            in. Pass one value computed over the whole scene when segmenting it
            in blocks or per stand, so that every crown is found under the same
            rule and crown statistics stay comparable between stands.

    Returns:
        GeoDataFrame with columns ``segment_id, area_m2, crown_diam_m``
        and polygon geometries in ``crs``.
    """
    px = float(abs(transform.a))
    smoothed = smooth_chm(np.asarray(chm, dtype="float32"), kernel=smooth_kernel)
    masked = mask_below(smoothed, threshold_m=min_height_m)
    valid = ~np.isnan(masked)
    if not valid.any():
        logger.warning("segment_crowns: no pixels above %.2f m", min_height_m)
        return gpd.GeoDataFrame(
            {"segment_id": [], "area_m2": [], "crown_diam_m": []},
            geometry=[],
            crs=crs,
        )

    mean_h = (
        float(np.nanmean(masked)) if mean_height_m is None else float(mean_height_m)
    )
    min_dist_px = _peak_distance_px(min_peak_distance_m, px, mean_h)

    # Local maxima -> tree-top markers.
    peaks = peak_local_max(
        np.where(valid, masked, 0.0),
        min_distance=min_dist_px,
        exclude_border=False,
        labels=valid.astype("uint8"),
    )
    if peaks.size == 0:
        logger.warning("segment_crowns: no local maxima detected")
        return gpd.GeoDataFrame(
            {"segment_id": [], "area_m2": [], "crown_diam_m": []},
            geometry=[],
            crs=crs,
        )

    markers = np.zeros(masked.shape, dtype="int32")
    for i, (r, c) in enumerate(peaks, start=1):
        markers[r, c] = i
    markers, _ = ndi.label(markers > 0)

    # Watershed on inverted CHM, restricted to valid pixels.
    inv = np.where(valid, -masked, np.inf).astype("float32")
    labels = watershed(inv, markers=markers, mask=valid)

    # Vectorize.
    polys = []
    seg_ids = []
    for geom, val in rio_shapes(
        labels.astype("int32"), mask=labels > 0, transform=transform
    ):
        polys.append(shapely_shape(geom))
        seg_ids.append(int(val))

    if not polys:
        return gpd.GeoDataFrame(
            {"segment_id": [], "area_m2": [], "crown_diam_m": []},
            geometry=[],
            crs=crs,
        )

    gdf = gpd.GeoDataFrame({"segment_id": seg_ids}, geometry=polys, crs=crs)
    # Dissolve multi-piece labels so each segment_id is a single polygon.
    gdf = gdf.dissolve(by="segment_id", as_index=False)
    gdf["area_m2"] = gdf.geometry.area
    gdf = gdf[gdf["area_m2"] >= float(min_crown_area_m2)].copy()
    gdf["crown_diam_m"] = 2.0 * np.sqrt(gdf["area_m2"].to_numpy() / np.pi)
    gdf.reset_index(drop=True, inplace=True)
    logger.info(
        "segment_crowns: %d crowns (mean diam=%.2f m)",
        len(gdf),
        float(gdf["crown_diam_m"].mean()) if len(gdf) else float("nan"),
    )
    return gdf


__all__ = ["segment_crowns"]
