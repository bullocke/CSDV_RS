"""csdv_core.download.naip_rest — NAIP from the USDA FPAC image service.

The Planetary Computer archive lags the NAIP programme by a year or more. The
USDA image service carries the newest acquisitions, so it fills that gap. The
service caps the size of one export, so a target grid is requested in tiles and
pasted together. Each tile is requested in the target CRS on the target pixel
corners, which keeps the result aligned with a canopy height model on that grid.
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

NAIP_REST_URL = (
    "https://apps.geo.fpac.usda.gov/geo-imagery/rest/services/"
    "naip/conus_naip/ImageServer"
)

__all__ = ["NAIP_REST_URL", "export_naip_to_grid", "find_quads"]


def find_quads(
    bbox_4326: tuple[float, float, float, float], *, year: int
) -> list[dict[str, Any]]:
    """Return the catalogue attributes of the quads over ``bbox_4326`` in ``year``."""
    import requests

    params = {
        "where": f"year_ts = {int(year)}",
        "geometry": ",".join(f"{v:.6f}" for v in bbox_4326),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "OBJECTID,Name,year_ts,ST,QQDATE",
        "returnGeometry": "false",
        "f": "json",
    }
    response = requests.get(f"{NAIP_REST_URL}/query", params=params, timeout=120)
    response.raise_for_status()
    return [feature["attributes"] for feature in response.json().get("features", [])]


def export_naip_to_grid(
    out_path: Path | str,
    *,
    epsg: int,
    dst_transform: Any,
    dst_width: int,
    dst_height: int,
    object_ids: list[int] | None = None,
    tile_px: int = 2048,
    attempts: int = 4,
) -> Path:
    """Export the service onto a north-up target grid as a four-band GeoTIFF.

    Args:
        out_path: Destination GeoTIFF.
        epsg: EPSG code of the target grid.
        dst_transform: Target affine transform.
        dst_width: Target width in pixels.
        dst_height: Target height in pixels.
        object_ids: Catalogue rasters to lock the mosaic to, from
            :func:`find_quads`. Without it the service chooses, and may return
            a different year.
        tile_px: Tile edge per request. The service caps one export at
            15000 by 4100 pixels.
        attempts: Tries per tile before giving up.
    """
    import numpy as np
    import rasterio
    import requests

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res = dst_transform.a
    x0, y0 = dst_transform.c, dst_transform.f
    mosaic = np.zeros((4, dst_height, dst_width), dtype="uint8")

    base = {
        "bboxSR": str(epsg),
        "imageSR": str(epsg),
        "format": "tiff",
        "pixelType": "U8",
        "interpolation": "RSP_NearestNeighbor",
        "f": "image",
    }
    if object_ids:
        ids = ",".join(str(i) for i in object_ids)
        base["mosaicRule"] = (
            f'{{"mosaicMethod":"esriMosaicLockRaster","lockRasterIds":[{ids}]}}'
        )

    n_tiles = -(-dst_height // tile_px) * -(-dst_width // tile_px)
    done = 0
    for row in range(0, dst_height, tile_px):
        for col in range(0, dst_width, tile_px):
            h = min(tile_px, dst_height - row)
            w = min(tile_px, dst_width - col)
            minx, maxy = x0 + col * res, y0 - row * res
            params = {
                **base,
                "bbox": f"{minx},{maxy - h * res},{minx + w * res},{maxy}",
                "size": f"{w},{h}",
            }
            for attempt in range(1, attempts + 1):
                try:
                    response = requests.get(
                        f"{NAIP_REST_URL}/exportImage", params=params, timeout=300
                    )
                    response.raise_for_status()
                    with rasterio.open(io.BytesIO(response.content)) as src:
                        block = src.read()
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt == attempts:
                        raise
                    logger.warning("Tile %d,%d failed (%s), retrying", row, col, exc)
                    time.sleep(5 * attempt)
            mosaic[: block.shape[0], row : row + h, col : col + w] = block[:4, :h, :w]
            done += 1
            logger.info("Tile %d of %d", done, n_tiles)

    profile = {
        "driver": "GTiff",
        "height": dst_height,
        "width": dst_width,
        "count": 4,
        "dtype": "uint8",
        "crs": f"EPSG:{epsg}",
        "transform": dst_transform,
        "compress": "DEFLATE",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic)
        dst.descriptions = ("red", "green", "blue", "nir")
    logger.info("Wrote %s", out_path)
    return out_path
