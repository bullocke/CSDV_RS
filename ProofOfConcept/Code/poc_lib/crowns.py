"""
poc_lib/crowns.py — Crown polygon utilities for CSDV proof-of-concept analyses.

Provides rasterization of crown polygons to binary masks, IoU computation, and
windowed statistics over crown diameter distributions. All functions expect
crown GeoPackages produced by 02_crown_segmentation.R with a 'crown_diam_m'
column (diameter estimated from polygon area assuming circular crown shape).
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import Affine
from shapely.geometry import mapping

logger = logging.getLogger(__name__)


def rasterize_crowns(crowns_path: Path, ref_chm_path: Path) -> np.ndarray:
    """Rasterize crown polygons to a binary mask matching the reference CHM grid.

    The output mask has 1 inside any crown polygon and 0 outside, aligned to
    the exact pixel grid of *ref_chm_path*. Crown polygons are reprojected to
    the reference CRS before rasterization.

    Parameters
    ----------
    crowns_path : Path
        Crown polygon GeoPackage (any CRS; reprojected internally).
    ref_chm_path : Path
        Reference raster that defines the output grid, shape, and CRS.

    Returns
    -------
    mask : np.ndarray
        uint8 array of shape (height, width). Returns zeros if GeoPackage empty.
    """
    with rasterio.open(ref_chm_path) as src:
        h, w = src.height, src.width
        transform = src.transform
        ref_crs = src.crs

    crowns = gpd.read_file(crowns_path).to_crs(ref_crs)
    if len(crowns) == 0:
        logger.warning("Empty crown GeoPackage: %s", crowns_path)
        return np.zeros((h, w), dtype=np.uint8)

    shapes = (
        (mapping(geom), 1)
        for geom in crowns.geometry
        if geom is not None and not geom.is_empty
    )
    return rio_rasterize(
        shapes,
        out_shape=(h, w),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )


def iou_stats(neon_mask: np.ndarray, naip_mask: np.ndarray) -> dict[str, float]:
    """Compute pixel-level IoU, precision, recall, and F1 between two binary masks.

    NEON is treated as the reference (ground truth). Both masks must have the
    same shape (use rasterize_crowns with the same ref_chm_path for both).

    Parameters
    ----------
    neon_mask : np.ndarray
        Binary mask from NEON crown segmentation (0/1, uint8 or bool).
    naip_mask : np.ndarray
        Binary mask from NAIP CHM crown segmentation (0/1, uint8 or bool).

    Returns
    -------
    dict with keys:
        iou       : Intersection over Union = TP / (TP + FP + FN)
        precision : TP / (TP + FP)
        recall    : TP / (TP + FN)
        f1        : 2 * precision * recall / (precision + recall)
        tp, fp, fn, tn : raw pixel counts (float)
    """
    neon = neon_mask.astype(bool).ravel()
    naip = naip_mask.astype(bool).ravel()

    tp = float(np.sum(neon & naip))
    fp = float(np.sum(~neon & naip))
    fn = float(np.sum(neon & ~naip))
    tn = float(np.sum(~neon & ~naip))

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def crown_stats_per_window(
    crowns_path: Path,
    ref_chm_path: Path,
    window_m: float,
    stat: str = "cv",
) -> tuple[np.ndarray | None, Affine | None, str | None]:
    """Compute a windowed statistic over crown diameter distributions.

    Replicates the R tapply aggregation from 02_crown_segmentation.R at any
    window size, for any summary statistic. Crown centroids are used to assign
    crowns to windows; windows with fewer than 3 crowns return NaN.

    Parameters
    ----------
    crowns_path : Path
        Crown polygon GeoPackage with 'crown_diam_m' column.
    ref_chm_path : Path
        Reference CHM defining the output extent and CRS.
    window_m : float
        Window side length in meters.
    stat : str
        Summary statistic. One of: "cv", "p90", "mean", "median", "count", "std".
        "cv" = coefficient of variation (σ/μ), the standard crown heterogeneity metric.
        "p90" = 90th percentile (highgrading indicator per summary_v1.md §4.3).

    Returns
    -------
    grid : np.ndarray or None
        2D float32 array. None if GeoPackage is empty or missing 'crown_diam_m'.
    transform : Affine or None
    crs : str or None

    Raises
    ------
    ValueError
        If *stat* is not one of the supported options.
    """
    valid_stats = {"cv", "p90", "mean", "median", "count", "std"}
    if stat not in valid_stats:
        raise ValueError(f"stat must be one of {sorted(valid_stats)}, got '{stat}'")

    crowns = gpd.read_file(crowns_path)
    if len(crowns) == 0 or "crown_diam_m" not in crowns.columns:
        logger.warning("No crowns or missing 'crown_diam_m': %s", crowns_path)
        return None, None, None

    with rasterio.open(ref_chm_path) as src:
        bounds = src.bounds
        ref_crs = src.crs
        pixel_size = abs(src.transform.a)

    crowns = crowns.to_crs(ref_crs)
    centroids = crowns.geometry.centroid
    diams = crowns["crown_diam_m"].values.astype(np.float64)

    xmin, ymax = bounds.left, bounds.top
    # Use floor division to discard partial edge cells, consistent with
    # gap_fraction() which uses (shape // window_px).
    window_px = max(1, int(round(window_m / pixel_size)))
    n_cols = int((bounds.right - xmin) / pixel_size) // window_px
    n_rows = int((ymax - bounds.bottom) / pixel_size) // window_px

    col_idx = ((centroids.x.values - xmin) / window_m).astype(int)
    row_idx = ((ymax - centroids.y.values) / window_m).astype(int)

    in_bounds = (
        (col_idx >= 0) & (col_idx < n_cols) & (row_idx >= 0) & (row_idx < n_rows)
    )
    col_idx = col_idx[in_bounds]
    row_idx = row_idx[in_bounds]
    diams = diams[in_bounds]

    cell_ids = row_idx * n_cols + col_idx
    result_flat = np.full(n_rows * n_cols, np.nan, dtype=np.float32)

    for cell_id in np.unique(cell_ids):
        mask = cell_ids == cell_id
        d = diams[mask]
        n = mask.sum()
        if stat == "count":
            result_flat[cell_id] = float(n)
            continue
        if n < 3:
            continue
        if stat == "cv":
            mu = d.mean()
            result_flat[cell_id] = float(d.std() / mu) if mu > 0 else np.nan
        elif stat == "p90":
            result_flat[cell_id] = float(np.percentile(d, 90))
        elif stat == "mean":
            result_flat[cell_id] = float(d.mean())
        elif stat == "median":
            result_flat[cell_id] = float(np.median(d))
        elif stat == "std":
            result_flat[cell_id] = float(d.std())

    grid = result_flat.reshape(n_rows, n_cols)
    transform = Affine(window_m, 0.0, xmin, 0.0, -window_m, ymax)
    logger.info(
        "Crown %s: %d × %d windows (%.0f m), non-NaN = %d",
        stat, n_rows, n_cols, window_m, int(np.sum(~np.isnan(grid))),
    )
    return grid, transform, str(ref_crs)
