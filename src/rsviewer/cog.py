"""rsviewer.cog: write Cloud Optimized GeoTIFFs for fast tiling.

The tile server reads one tile per request. Large rasters without internal
overviews make every zoomed-out tile read the full-resolution pixels, which is
slow (about 2 s per tile for a 0.6 m NAIP quarter quad, against about 0.2 s
from a COG). :func:`file_to_cog` converts such files once, and
:func:`array_to_cog` writes in-memory results (e.g. analysis outputs held in
``.npz`` files) as georeferenced COGs. :func:`is_stale` lets callers rebuild a
cached COG only when its sources change.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

logger = logging.getLogger(__name__)

#: GDAL settings for conversions. The block cache is kept small for WSL.
GDAL_CONFIG: dict[str, Any] = {"GDAL_NUM_THREADS": "ALL_CPUS", "GDAL_CACHEMAX": 512}


def _profile(dtype: str, blocksize: int) -> dict[str, Any]:
    prof = cog_profiles.get("deflate")
    prof.update(blockxsize=blocksize, blockysize=blocksize)
    prof["predictor"] = 3 if np.dtype(dtype).kind == "f" else 2
    return prof


def is_stale(dst: Path, *sources: Path) -> bool:
    """Whether ``dst`` is missing or older than any existing source file."""
    dst = Path(dst)
    if not dst.exists():
        return True
    t = dst.stat().st_mtime
    return any(Path(s).exists() and Path(s).stat().st_mtime > t for s in sources)


def file_to_cog(
    src: Path,
    dst: Path,
    *,
    resampling: str = "average",
    blocksize: int = 512,
    overwrite: bool = False,
) -> Path:
    """Convert a GeoTIFF to a COG with internal overviews.

    Args:
        src: Input raster.
        dst: Output COG path. Parent folders are created.
        resampling: Overview resampling (``"average"`` for imagery and heights,
            ``"nearest"`` for classes).
        blocksize: Internal tile size in pixels.
        overwrite: Rebuild even if ``dst`` is newer than ``src``.

    Returns:
        ``dst``.

    Raises:
        FileNotFoundError: ``src`` does not exist.
    """
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise FileNotFoundError(src)
    if not overwrite and not is_stale(dst, src):
        logger.debug("COG up to date: %s", dst)
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src) as ds:
        prof = _profile(ds.dtypes[0], blocksize)
    tmp = dst.with_suffix(".tmp.tif")
    cog_translate(
        str(src),
        str(tmp),
        prof,
        overview_resampling=resampling,
        quiet=True,
        allow_intermediate_compression=True,
        config=GDAL_CONFIG,
    )
    tmp.replace(dst)
    logger.info("Wrote COG %s (%.0f MB)", dst, dst.stat().st_size / 1e6)
    return dst


def array_to_cog(
    arr: np.ndarray,
    dst: Path,
    *,
    transform: Any,
    crs: Any,
    nodata: float | None = None,
    band_names: Sequence[str] | None = None,
    resampling: str = "nearest",
    blocksize: int = 256,
) -> Path:
    """Write a (H, W) or (B, H, W) array as a COG.

    Args:
        arr: Raster values. NaN is replaced by ``nodata`` for float arrays.
        dst: Output path. Parent folders are created.
        transform: Affine transform of the grid.
        crs: Coordinate reference system (anything rasterio accepts).
        nodata: Nodata value written to the file.
        band_names: Optional band descriptions, one per band.
        resampling: Overview resampling. Keep ``"nearest"`` for classes.
        blocksize: Internal tile size in pixels.

    Returns:
        ``dst``.

    Raises:
        ValueError: ``arr`` is not 2-D or 3-D, or ``band_names`` has the wrong
            length.
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        a = a[None]
    if a.ndim != 3:
        raise ValueError(f"expected a 2-D or 3-D array, got shape {arr.shape}")
    if band_names is not None and len(band_names) != a.shape[0]:
        raise ValueError("band_names must have one entry per band")
    if nodata is not None and a.dtype.kind == "f":
        a = np.where(np.isfinite(a), a, nodata).astype(a.dtype)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff",
        width=a.shape[2],
        height=a.shape[1],
        count=a.shape[0],
        dtype=a.dtype.name,
        crs=crs,
        transform=transform,
        nodata=nodata,
    )
    tmp = dst.with_suffix(".tmp.tif")
    with MemoryFile() as mem:
        with mem.open(**profile) as ds:
            ds.write(a)
            for i, name in enumerate(band_names or (), start=1):
                ds.set_band_description(i, str(name))
        with mem.open() as ds:
            cog_translate(
                ds,
                str(tmp),
                _profile(a.dtype.name, blocksize),
                overview_resampling=resampling,
                quiet=True,
                allow_intermediate_compression=True,
                in_memory=True,
                config=GDAL_CONFIG,
            )
    tmp.replace(dst)
    logger.debug("Wrote COG %s", dst)
    return dst
