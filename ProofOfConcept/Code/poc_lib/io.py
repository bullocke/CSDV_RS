"""
poc_lib/io.py — Raster and vector I/O helpers for CSDV proof-of-concept analyses.

All functions operate on Path objects, work with EPSG:5070 (NAD83 Conus Albers)
by default, and use rasterio for all raster I/O (no raw GDAL bindings).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.mask import mask as rasterio_mask
from rasterio.transform import Affine
from shapely.geometry import box

logger = logging.getLogger(__name__)


def find_latest_tif(
    directory: Path,
    prefix: str,
    exclude_suffixes: tuple[str, ...] = ("_zoom", "_deepzoom", "_mediumzoom"),
) -> Path:
    """Return the most-recently-modified .tif matching *prefix*, excluding
    filenames that contain any of *exclude_suffixes* in their stem.

    Parameters
    ----------
    directory : Path
        Directory to search.
    prefix : str
        Filename prefix (glob pattern: ``{prefix}*.tif``).
    exclude_suffixes : tuple of str
        Stems containing these strings are excluded. Default skips zoom variants.

    Returns
    -------
    Path
        Path to the most recent matching file.

    Raises
    ------
    FileNotFoundError
        If no matching file is found.
    """
    tifs = sorted(
        [
            p
            for p in directory.glob(f"{prefix}*.tif")
            if not any(s in p.stem for s in exclude_suffixes)
        ],
        key=lambda p: p.stat().st_mtime,
    )
    if not tifs:
        raise FileNotFoundError(
            f"No .tif matching '{prefix}*' (excluding {exclude_suffixes}) in {directory}"
        )
    return tifs[-1]


def clip_raster_to_bbox(
    src_path: Path,
    bbox: tuple[float, float, float, float],
    out_path: Path,
) -> None:
    """Clip any raster (single- or multi-band) to a geographic bounding box.

    Parameters
    ----------
    src_path : Path
        Source raster GeoTIFF (any number of bands).
    bbox : (west, south, east, north)
        Bounding box in the raster's CRS (typically EPSG:5070 meters).
    out_path : Path
        Output GeoTIFF path. Parent directory is created if needed.
    """
    west, south, east, north = bbox
    geom = box(west, south, east, north)
    with rasterio.open(src_path) as src:
        out_data, out_transform = rasterio_mask(src, [geom], crop=True, all_touched=True)
        profile = src.profile.copy()
        profile.update(
            height=out_data.shape[1],
            width=out_data.shape[2],
            transform=out_transform,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_data)
    logger.info("Clipped → %s (%.1f MB)", out_path.name, out_path.stat().st_size / 1e6)


def clip_and_convert_naip_chm(
    src_path: Path,
    bbox: tuple[float, float, float, float],
    out_path: Path,
    scale: float = 0.01,
) -> None:
    """Clip the NAIP CHM (UInt16, stored as height_m × 100) and convert to
    float32 meters in one step.

    The Morford et al. 2025 NAIP CHM is delivered as UInt16 with a scale factor
    of 0.01 (i.e., stored_value × 0.01 = height in meters). This function clips
    to *bbox* and writes a float32 GeoTIFF in meters, which is directly usable
    by compute_gap_fraction() and 02_crown_segmentation.R.

    Parameters
    ----------
    src_path : Path
        NAIP CHM GeoTIFF (UInt16).
    bbox : (west, south, east, north)
        Bounding box in the raster's CRS.
    out_path : Path
        Output float32 GeoTIFF path.
    scale : float
        Multiplicative scale factor. Default 0.01 (UInt16 → meters).
    """
    west, south, east, north = bbox
    geom = box(west, south, east, north)
    with rasterio.open(src_path) as src:
        out_data, out_transform = rasterio_mask(src, [geom], crop=True, all_touched=True)
        src_nodata = src.nodata
        profile = src.profile.copy()

    arr = out_data[0].astype(np.float32) * scale
    if src_nodata is not None:
        arr[out_data[0] == src_nodata] = -9999.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(
        height=arr.shape[0],
        width=arr.shape[1],
        transform=out_transform,
        dtype="float32",
        nodata=-9999.0,
        count=1,
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr[np.newaxis])
    logger.info(
        "NAIP CHM clip (meters) → %s (%.1f MB)",
        out_path.name,
        out_path.stat().st_size / 1e6,
    )


def read_band(
    path: Path,
    band: int = 1,
    nodata_to_nan: bool = True,
) -> tuple[np.ndarray, Affine, str]:
    """Read a single raster band as float32.

    Parameters
    ----------
    path : Path
        GeoTIFF path.
    band : int
        1-based band index. Default 1.
    nodata_to_nan : bool
        If True, replace nodata values with NaN. Default True.

    Returns
    -------
    data : np.ndarray
        2D float32 array, shape (height, width).
    transform : Affine
        Rasterio Affine transform.
    crs : str
        CRS string (e.g. "EPSG:5070").
    """
    with rasterio.open(path) as src:
        data = src.read(band).astype(np.float32)
        transform = src.transform
        crs = str(src.crs)
        nodata = src.nodata
    if nodata_to_nan and nodata is not None:
        data[data == nodata] = np.nan
    return data, transform, crs
