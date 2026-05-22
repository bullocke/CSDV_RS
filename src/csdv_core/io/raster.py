"""csdv_core.io.raster — rasterio-based raster I/O helpers.

Ported from ``legacy/proof_of_concept/Code/poc_lib/io.py``. Keeps the
single-band reader, bbox clipper, NAIP-CHM uint16-to-meters conversion,
and a latest-by-name file selector. A small writer is added so tests can
round-trip rasters in-memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import box

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RasterRead:
    """Result of :func:`read_band`."""

    data: np.ndarray
    transform: Any
    crs: Any
    nodata: float | None


def read_band(
    path: Path | str,
    band: int = 1,
    nodata_to_nan: bool = True,
) -> RasterRead:
    """Read a single band from a raster.

    Args:
        path: Raster file path.
        band: 1-based band index.
        nodata_to_nan: If True, replace nodata pixels with ``np.nan`` and
            cast to float32 when needed.

    Returns:
        :class:`RasterRead` with array, transform, CRS, and nodata value.
    """
    with rasterio.open(path) as src:
        arr = src.read(band)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
    if nodata_to_nan and nodata is not None:
        arr = arr.astype("float32", copy=False)
        arr = np.where(arr == nodata, np.nan, arr)
    return RasterRead(data=arr, transform=transform, crs=crs, nodata=nodata)


def write_raster(
    path: Path | str,
    data: np.ndarray,
    *,
    transform: Any,
    crs: Any,
    nodata: float | None = None,
    dtype: str | None = None,
) -> None:
    """Write a single-band or multi-band raster.

    Args:
        path: Output path.
        data: 2-D ``(rows, cols)`` or 3-D ``(bands, rows, cols)`` array.
        transform: Affine transform.
        crs: Coordinate reference system.
        nodata: Optional nodata sentinel written into the file metadata.
        dtype: Optional dtype string. Defaults to ``data.dtype``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim == 2:
        count = 1
        height, width = data.shape
        bands = data[np.newaxis, :, :]
    elif data.ndim == 3:
        count, height, width = data.shape
        bands = data
    else:
        raise ValueError(f"data must be 2-D or 3-D, got shape {data.shape!r}")
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": dtype or str(data.dtype),
        "transform": transform,
        "crs": crs,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(bands)


def clip_to_bbox(
    src_path: Path | str,
    bbox: tuple[float, float, float, float],
    out_path: Path | str,
) -> Path:
    """Clip ``src_path`` to ``bbox`` (in source CRS) and write to ``out_path``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    geom = [box(*bbox).__geo_interface__]
    with rasterio.open(src_path) as src:
        clipped, transform = rio_mask(src, geom, crop=True)
        profile = src.profile.copy()
        profile.update(
            height=clipped.shape[1],
            width=clipped.shape[2],
            transform=transform,
        )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(clipped)
    return out_path


def clip_and_convert_naip_chm(
    src_path: Path | str,
    bbox: tuple[float, float, float, float],
    out_path: Path | str,
    scale: float = 0.01,
) -> Path:
    """Clip a NAIP-CHM raster and convert uint16 storage to float32 meters.

    NAIP-CHM (Morford et al., 2025) is distributed as uint16 with a scale
    factor of 0.01 m. This helper clips to ``bbox``, applies the scale,
    and writes float32 with nodata = -9999.0.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    geom = [box(*bbox).__geo_interface__]
    with rasterio.open(src_path) as src:
        clipped, transform = rio_mask(src, geom, crop=True)
        src_nodata = src.nodata
        profile = src.profile.copy()
    arr = clipped.astype("float32") * float(scale)
    if src_nodata is not None:
        arr = np.where(clipped == src_nodata, -9999.0, arr)
    profile.update(
        dtype="float32",
        nodata=-9999.0,
        height=arr.shape[1],
        width=arr.shape[2],
        transform=transform,
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype("float32"))
    return out_path


def find_latest_tif(
    directory: Path | str,
    prefix: str,
    exclude_suffixes: tuple[str, ...] = ("_zoom", "_deepzoom", "_mediumzoom"),
) -> Path | None:
    """Return the most recently named ``.tif`` in ``directory`` starting with ``prefix``.

    Excludes files whose stems end with any of ``exclude_suffixes``. Sorted
    lexicographically; timestamped filenames pick the latest acquisition.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None
    candidates = [
        p
        for p in directory.glob(f"{prefix}*.tif")
        if not any(p.stem.endswith(s) for s in exclude_suffixes)
    ]
    if not candidates:
        return None
    return sorted(candidates)[-1]
