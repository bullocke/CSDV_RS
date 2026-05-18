"""
poc_lib/metrics.py — Structural metric computation for CSDV proof-of-concept.

All functions accept CHM or crown polygon inputs and return (array, Affine, crs)
tuples suitable for passing directly to save_raster().

Metrics
-------
gap_fraction      : fraction of pixels below a height threshold per window
crown_fraction    : complement of gap fraction (canopy cover fraction)
crown_width_p90   : 90th-percentile crown diameter per window (highgrading indicator)
metric_difference : pixel-aligned NAIP − NEON difference raster

Parameter choices
-----------------
- Height threshold (2 m): matches the gap pixel definition in CSDV Classification
  V5 Full.md. Pixels with canopy height < 2 m are classified as gaps.
- Window sizes (25, 50, 100 m): not specified in V5 docs; chosen to span the
  range from stand-scale (~1 canopy-width) to coarse inventory-plot scale.
  For operational use, 50 m is a reasonable default; 25 m better resolves
  within-stand heterogeneity at the cost of noisier estimates.
- Crown Width P90 (50 m window): P90 rather than mean is sensitive to the
  removal of the largest trees (highgrading signature per summary_v1.md §4.3).
  A 50 m window typically contains 5–30 crowns in closed canopy, enough for a
  stable 90th-percentile estimate.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# save_raster (re-exported here so callers only need to import poc_lib.metrics)
# ---------------------------------------------------------------------------

def save_raster(
    array: np.ndarray,
    path: Path,
    transform: Affine,
    crs: str,
    nodata: float = -9999.0,
) -> None:
    """Write a 2D float32 array to a single-band GeoTIFF.

    NaN values are replaced with *nodata* before writing.

    Parameters
    ----------
    array : np.ndarray
        2D array to save (will be cast to float32).
    path : Path
        Output file path. Parent directory is created if needed.
    transform : Affine
        Rasterio Affine transform.
    crs : str
        CRS string (e.g. "EPSG:5070").
    nodata : float
        NoData sentinel value. Default -9999.0.
    """
    arr = array.astype(np.float32)
    arr[np.isnan(arr)] = nodata
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(arr, 1)
    logger.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# Gap fraction
# ---------------------------------------------------------------------------

def gap_fraction(
    chm_path: Path,
    window_m: float = 25.0,
    height_threshold_m: float = 2.0,
) -> tuple[np.ndarray, Affine, str]:
    """Compute gap fraction within non-overlapping spatial windows.

    Gap pixels are defined as those with canopy height < *height_threshold_m*.
    The 2 m threshold is taken from the CSDV Classification V5 Full.md
    definition of gap pixels (pixels with no woody canopy contribution above
    2 m are treated as inter-crown gaps or open ground).

    Parameters
    ----------
    chm_path : Path
        Canopy height model GeoTIFF (height in meters, float32 or convertible).
    window_m : float
        Window side length in meters (non-overlapping square tiles).
        Not specified in V5 docs; 25 m is a reasonable stand-scale default.
    height_threshold_m : float
        Pixels below this height are counted as gaps. Default 2 m (V5-defined).

    Returns
    -------
    gap_frac : np.ndarray
        2D float32 array of gap fraction values (0–1).
    out_transform : Affine
        Affine transform for the output raster (one pixel = one window).
    crs : str
        CRS string from the input raster.
    """
    with rasterio.open(chm_path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        pixel_size = abs(src.transform.a)
        ox, oy = src.transform.c, src.transform.f
        crs = str(src.crs)
    if nodata is not None:
        data[data == nodata] = np.nan

    window_px = max(1, int(round(window_m / pixel_size)))
    rows = data.shape[0] // window_px
    cols = data.shape[1] // window_px

    gap_frac = np.full((rows, cols), np.nan, dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            tile = data[
                r * window_px : (r + 1) * window_px,
                c * window_px : (c + 1) * window_px,
            ]
            valid = tile[~np.isnan(tile)]
            if valid.size == 0:
                continue
            gap_frac[r, c] = float(np.mean(valid < height_threshold_m))

    out_transform = Affine(window_m, 0.0, ox, 0.0, -window_m, oy)
    logger.info(
        "Gap fraction: %d × %d windows (%.0f m, %.2f m pixel, threshold=%.1f m)",
        rows, cols, window_m, pixel_size, height_threshold_m,
    )
    return gap_frac, out_transform, crs


# ---------------------------------------------------------------------------
# Crown fraction (complement of gap fraction)
# ---------------------------------------------------------------------------

def crown_fraction(
    chm_path: Path,
    window_m: float = 25.0,
    height_threshold_m: float = 2.0,
) -> tuple[np.ndarray, Affine, str]:
    """Compute crown cover fraction within non-overlapping spatial windows.

    Crown fraction = 1 − gap fraction. Pixels with height >= *height_threshold_m*
    are counted as canopy cover. This is equivalent to crown closure or canopy
    cover fraction, a standard metric in forest inventory and remote sensing.

    Parameters
    ----------
    chm_path : Path
        Canopy height model GeoTIFF (height in meters).
    window_m : float
        Window side length in meters. Default 25 m.
    height_threshold_m : float
        Pixels at or above this height are counted as crown. Default 2 m.

    Returns
    -------
    crown_frac : np.ndarray
        2D float32 array of crown fraction values (0–1).
    out_transform : Affine
    crs : str
    """
    gf, transform, crs = gap_fraction(chm_path, window_m, height_threshold_m)
    crown_frac = 1.0 - gf
    crown_frac[np.isnan(gf)] = np.nan
    return crown_frac.astype(np.float32), transform, crs


# ---------------------------------------------------------------------------
# Crown width P90
# ---------------------------------------------------------------------------

def crown_width_p90(
    crowns_path: Path,
    ref_chm_path: Path,
    window_m: float = 50.0,
) -> tuple[np.ndarray | None, Affine | None, str | None]:
    """Compute the 90th-percentile crown diameter per spatial window.

    Crown Width P90 is sensitive to the removal of the largest-diameter trees
    (a diagnostic signature of highgrading per CSDV summary_v1.md §4.3). Unlike
    crown CV, it does not require a minimum of 3 crowns per window, but cells
    with fewer than 3 crowns still return NaN to avoid single-crown instability.

    The crown GeoPackage must contain a 'crown_diam_m' column (produced by
    02_crown_segmentation.R). Crown centroids are used to assign crowns to
    windows.

    Parameters
    ----------
    crowns_path : Path
        Crown polygon GeoPackage with 'crown_diam_m' column.
    ref_chm_path : Path
        Reference CHM used to set the output extent and CRS.
    window_m : float
        Window side length in meters. Default 50 m.

    Returns
    -------
    p90_grid : np.ndarray or None
        2D float32 array of P90 crown diameter values (meters). None if no crowns.
    out_transform : Affine or None
    crs : str or None
    """
    crowns = gpd.read_file(crowns_path)
    if len(crowns) == 0 or "crown_diam_m" not in crowns.columns:
        logger.warning("No crowns or missing 'crown_diam_m' column: %s", crowns_path)
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

    valid = (col_idx >= 0) & (col_idx < n_cols) & (row_idx >= 0) & (row_idx < n_rows)
    col_idx = col_idx[valid]
    row_idx = row_idx[valid]
    diams = diams[valid]

    cell_ids = row_idx * n_cols + col_idx
    p90_flat = np.full(n_rows * n_cols, np.nan, dtype=np.float32)

    for cell_id in np.unique(cell_ids):
        mask = cell_ids == cell_id
        if mask.sum() < 3:
            continue
        p90_flat[cell_id] = float(np.percentile(diams[mask], 90))

    p90_grid = p90_flat.reshape(n_rows, n_cols)
    transform = Affine(window_m, 0.0, xmin, 0.0, -window_m, ymax)
    logger.info(
        "Crown Width P90: %d × %d windows (%.0f m), non-NaN cells = %d",
        n_rows, n_cols, window_m, int(np.sum(~np.isnan(p90_grid))),
    )
    return p90_grid, transform, str(ref_crs)


# ---------------------------------------------------------------------------
# Metric difference (NAIP − NEON)
# ---------------------------------------------------------------------------

def metric_difference(
    neon_path: Path,
    naip_path: Path,
) -> tuple[np.ndarray, Affine, str]:
    """Compute pixel-aligned difference: NAIP CHM metric − NEON CHM metric.

    If the two rasters have different shapes, the NAIP metric is reprojected
    onto the NEON grid using nearest-neighbor resampling before subtraction.
    This preserves the NEON grid as the reference (NEON ALS is the benchmark).

    Parameters
    ----------
    neon_path : Path
        NEON-derived metric GeoTIFF (reference grid).
    naip_path : Path
        NAIP-derived metric GeoTIFF (reprojected to NEON grid if needed).

    Returns
    -------
    diff : np.ndarray
        2D float32 difference array (NAIP − NEON), NaN where either is NaN.
    transform : Affine
        Affine transform of the NEON (reference) grid.
    crs : str
        CRS of the NEON (reference) grid.
    """
    with rasterio.open(neon_path) as src_neon:
        neon_data = src_neon.read(1).astype(np.float32)
        neon_nodata = src_neon.nodata
        neon_transform = src_neon.transform
        neon_crs = src_neon.crs
        neon_shape = (src_neon.height, src_neon.width)

    if neon_nodata is not None:
        neon_data[neon_data == neon_nodata] = np.nan

    with rasterio.open(naip_path) as src_naip:
        naip_data_raw = src_naip.read(1).astype(np.float32)
        naip_nodata = src_naip.nodata
        naip_crs = src_naip.crs
        naip_transform = src_naip.transform
        naip_shape = (src_naip.height, src_naip.width)

    if naip_nodata is not None:
        naip_data_raw[naip_data_raw == naip_nodata] = np.nan

    # Reproject NAIP onto NEON grid if grids differ
    if naip_shape != neon_shape or naip_transform != neon_transform:
        logger.info("Reprojecting NAIP metric onto NEON grid for difference computation")
        naip_aligned = np.full(neon_shape, np.nan, dtype=np.float32)
        reproject(
            source=naip_data_raw,
            destination=naip_aligned,
            src_transform=naip_transform,
            src_crs=naip_crs,
            dst_transform=neon_transform,
            dst_crs=neon_crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
    else:
        naip_aligned = naip_data_raw

    diff = naip_aligned - neon_data
    # Mask where either source is NaN
    diff[np.isnan(neon_data) | np.isnan(naip_aligned)] = np.nan

    valid = diff[~np.isnan(diff)]
    if valid.size > 0:
        logger.info(
            "Difference (NAIP−NEON): RMSE=%.4f, bias=%.4f, N=%d",
            float(np.sqrt(np.mean(valid**2))),
            float(np.mean(valid)),
            valid.size,
        )
    return diff.astype(np.float32), neon_transform, str(neon_crs)
