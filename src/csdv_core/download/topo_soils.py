"""csdv_core.download.topo_soils — 3DEP DEM acquisition.

The Planetary Computer serves the USGS 3DEP seamless DEM as cloud-optimised
GeoTIFFs in one degree tiles, at 10 m and 30 m. The pattern mirrors
:mod:`csdv_core.download.naip_pc`: search for the tiles over a box, then read
them through a warped virtual raster targeted at the grid the caller wants, so
the elevation is resampled once.

SSURGO acquisition is not implemented yet.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from csdv_core.download.naip_pc import PC_STAC_URL

logger = logging.getLogger(__name__)

DEM_COLLECTION = "3dep-seamless"
DEM_ASSET = "data"
DEM_NODATA = -9999.0

__all__ = [
    "DEM_COLLECTION",
    "DEM_NODATA",
    "dem_to_grid",
    "search_3dep_items",
]


def search_3dep_items(
    bbox_4326: tuple[float, float, float, float],
    *,
    gsd: int = 10,
    catalog: Any | None = None,
) -> list[Any]:
    """Find the 3DEP seamless tiles intersecting ``bbox_4326`` at ``gsd`` metres.

    Raises:
        ValueError: If no tile at that resolution covers the box.
    """
    import planetary_computer
    import pystac_client

    if catalog is None:
        catalog = pystac_client.Client.open(
            PC_STAC_URL, modifier=planetary_computer.sign_inplace
        )
    search = catalog.search(collections=[DEM_COLLECTION], bbox=list(bbox_4326))
    items = [item for item in search.items() if item.properties.get("gsd") == gsd]
    if not items:
        raise ValueError(f"No {gsd} m 3DEP tiles over {bbox_4326}")
    logger.info("%d 3DEP tiles at %d m", len(items), gsd)
    return items


def dem_to_grid(
    items: Sequence[Any],
    out_path: Path | str,
    *,
    dst_crs: str,
    dst_transform: Any,
    dst_width: int,
    dst_height: int,
    resampling: str = "cubic",
) -> Path:
    """Warp 3DEP tiles onto a target grid and write a float32 GeoTIFF in metres.

    Cubic resampling keeps the surface smooth when the target grid is finer
    than the source, which matters for a hillshade or a rendered surface.
    Bilinear leaves visible facets at the source cell size.
    """
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dem = np.full((dst_height, dst_width), np.nan, dtype="float32")
    with rasterio.Env(GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="3"):
        for item in items:
            with (
                rasterio.open(item.assets[DEM_ASSET].href) as src,
                WarpedVRT(
                    src,
                    crs=dst_crs,
                    transform=dst_transform,
                    width=dst_width,
                    height=dst_height,
                    resampling=Resampling[resampling],
                    dst_nodata=DEM_NODATA,
                ) as vrt,
            ):
                block = vrt.read(1).astype("float32")
            valid = (block != DEM_NODATA) & np.isfinite(block)
            fill = valid & np.isnan(dem)
            dem[fill] = block[fill]

    coverage = float(np.isfinite(dem).mean())
    if coverage < 0.999:
        logger.warning("DEM covers only %.3f of the target grid", coverage)
    profile = {
        "driver": "GTiff",
        "height": dst_height,
        "width": dst_width,
        "count": 1,
        "dtype": "float32",
        "crs": dst_crs,
        "transform": dst_transform,
        "nodata": DEM_NODATA,
        "compress": "DEFLATE",
        "tiled": True,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(np.where(np.isfinite(dem), dem, DEM_NODATA).astype("float32"), 1)
    logger.info(
        "Wrote %s, elevation %.0f to %.0f m",
        out_path.name,
        float(np.nanmin(dem)),
        float(np.nanmax(dem)),
    )
    return out_path
