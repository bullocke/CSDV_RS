"""csdv_core.download.naip_pc — NAIP from the Microsoft Planetary Computer.

The Planetary Computer serves NAIP as public cloud-optimised GeoTIFFs, one per
digital orthophoto quarter quadrangle, in the imagery's native UTM zone at its
native resolution. For a study module that is often the least troublesome way to
get the imagery: no account, no export quota, no reprojection to undo, and the
files land in the same CRS as a state's photo-interpretation delivery.

Two details matter enough to be worth stating.

Access tokens expire after about an hour, so a job spanning several dates must
sign its links date by date rather than all at once up front.

And every date has to land on one grid. A difference between two dates is only
meaningful pixel for pixel, so each year's quads are read through a warped
virtual raster targeted at a shared grid, which makes the mosaic step a paste
and keeps resampling to a single pass.

Earth Engine remains the route to imagery older than the Planetary Computer
archive; see :mod:`csdv_core.download.naip_gee`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
NAIP_COLLECTION = "naip"
NAIP_ASSET = "image"
NAIP_BANDS = ("red", "green", "blue", "nir")

__all__ = [
    "NAIP_BANDS",
    "NAIP_COLLECTION",
    "PC_STAC_URL",
    "GridSpecOut",
    "grid_from_bounds",
    "median_date_tag",
    "naip_mosaic",
    "search_naip_items",
    "snap_bounds",
]


class GridSpecOut(tuple):
    """``(transform, width, height)`` for a target grid, with named access."""

    __slots__ = ()

    def __new__(cls, transform: Any, width: int, height: int) -> GridSpecOut:
        return super().__new__(cls, (transform, int(width), int(height)))

    @property
    def transform(self) -> Any:
        return self[0]

    @property
    def width(self) -> int:
        return self[1]

    @property
    def height(self) -> int:
        return self[2]


def snap_bounds(
    bounds: tuple[float, float, float, float],
    *,
    snap_m: float,
    pad_m: float = 0.0,
) -> tuple[float, float, float, float]:
    """Widen ``bounds`` by ``pad_m`` and snap outward to a multiple of ``snap_m``.

    Snapping to a common multiple of every resolution in play, 3 m for NAIP's
    0.6 m and 1.0 m products, makes the two grids nest so a coarse date and a
    fine date share pixel corners.
    """
    if snap_m <= 0:
        raise ValueError(f"snap_m must be positive, got {snap_m}")
    minx, miny, maxx, maxy = bounds
    return (
        math.floor((minx - pad_m) / snap_m) * snap_m,
        math.floor((miny - pad_m) / snap_m) * snap_m,
        math.ceil((maxx + pad_m) / snap_m) * snap_m,
        math.ceil((maxy + pad_m) / snap_m) * snap_m,
    )


def grid_from_bounds(
    bounds: tuple[float, float, float, float],
    resolution: float,
) -> GridSpecOut:
    """Return the transform and shape of a north-up grid covering ``bounds``."""
    from affine import Affine

    minx, miny, maxx, maxy = bounds
    width = int(round((maxx - minx) / resolution))
    height = int(round((maxy - miny) / resolution))
    if width <= 0 or height <= 0:
        raise ValueError(f"Bounds {bounds} give an empty grid at {resolution} m")
    transform = Affine(resolution, 0.0, minx, 0.0, -resolution, maxy)
    return GridSpecOut(transform, width, height)


def median_date_tag(datetimes: Sequence[str | datetime]) -> str:
    """Return the median acquisition date of a set of items as ``YYYYMMDD``.

    A mapping module usually spans several quads, and the quads are not always
    flown on the same day. The canopy height model reads day of year from the
    filename, so the mosaic gets one representative date. Record every
    individual date in the run's provenance, because the model conditions on
    that single value for the whole mosaic.
    """
    if not datetimes:
        raise ValueError("No acquisition dates supplied")
    parsed = sorted(
        (
            value
            if isinstance(value, datetime)
            else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        )
        for value in datetimes
    )
    return parsed[len(parsed) // 2].strftime("%Y%m%d")


def search_naip_items(
    bbox_4326: tuple[float, float, float, float],
    *,
    year: int,
    catalog: Any | None = None,
) -> list[Any]:
    """Find every NAIP quad intersecting ``bbox_4326`` in ``year``.

    A catalogue opened with the signing modifier returns items whose asset
    links already carry a token. Open it inside the per-year loop rather than
    once for a whole job, because the tokens expire.

    Raises:
        ValueError: If the year has no imagery over the box.
    """
    import planetary_computer
    import pystac_client

    if catalog is None:
        catalog = pystac_client.Client.open(
            PC_STAC_URL, modifier=planetary_computer.sign_inplace
        )
    search = catalog.search(
        collections=[NAIP_COLLECTION],
        bbox=list(bbox_4326),
        datetime=f"{year}-01-01/{year}-12-31",
    )
    items = list(search.items())
    if not items:
        raise ValueError(f"No NAIP imagery for {year} over {bbox_4326}")
    resolutions = {item.properties.get("gsd") for item in items}
    logger.info(
        "%d: %d NAIP quads, ground sample distance %s",
        year,
        len(items),
        sorted(r for r in resolutions if r is not None),
    )
    return items


def naip_mosaic(
    items: Sequence[Any],
    out_path: Path | str,
    *,
    dst_crs: str,
    dst_transform: Any,
    dst_width: int,
    dst_height: int,
    resampling: str = "bilinear",
    compress: str = "DEFLATE",
) -> tuple[Path, dict[str, Any]]:
    """Mosaic NAIP quads onto a target grid and write a four-band GeoTIFF.

    Each quad is opened through a warped virtual raster already targeted at the
    output grid, so the merge is a paste and every pixel is resampled once.

    Args:
        items: STAC items, already signed.
        out_path: Destination GeoTIFF.
        dst_crs: Target CRS.
        dst_transform: Target affine transform.
        dst_width: Target width in pixels.
        dst_height: Target height in pixels.
        resampling: Resampling method for the warp.
        compress: GeoTIFF compression.

    Returns:
        ``(path, provenance)``. The provenance dictionary records the item
        identifiers, every acquisition date, the ground sample distances found
        and the grid written, which is what a later reader needs to judge
        whether two dates are comparable.
    """
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    method = Resampling[resampling]

    mosaic = np.zeros((4, dst_height, dst_width), dtype="uint8")
    written = np.zeros((dst_height, dst_width), dtype=bool)
    full = Window(col_off=0, row_off=0, width=dst_width, height=dst_height)

    env = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        "GDAL_HTTP_MAX_RETRY": "5",
        "GDAL_HTTP_RETRY_DELAY": "3",
    }
    with rasterio.Env(**env):
        for item in items:
            href = item.assets[NAIP_ASSET].href
            with rasterio.open(href) as src:
                # Warp only into the part of the output this quad can reach.
                # Reading the whole grid once per quad would multiply the work
                # by the number of quads for no benefit.
                quad = transform_bounds(src.crs, dst_crs, *src.bounds, densify_pts=21)
                window = from_bounds(*quad, transform=dst_transform)
                window = window.round_offsets().round_lengths()
                try:
                    window = window.intersection(full)
                except rasterio.errors.WindowError:
                    logger.debug("%s falls outside the target grid", item.id)
                    continue
                if window.width < 1 or window.height < 1:
                    continue
                sub_transform = rasterio.windows.transform(window, dst_transform)
                with WarpedVRT(
                    src,
                    crs=dst_crs,
                    transform=sub_transform,
                    width=int(window.width),
                    height=int(window.height),
                    resampling=method,
                ) as vrt:
                    block = vrt.read()

            rows = slice(int(window.row_off), int(window.row_off + window.height))
            cols = slice(int(window.col_off), int(window.col_off + window.width))
            covered = block.any(axis=0)
            fill = covered & ~written[rows, cols]
            target = mosaic[:, rows, cols]
            target[:, fill] = block[:, fill]
            mosaic[:, rows, cols] = target
            written[rows, cols] |= covered

    coverage = float(written.mean())
    profile = {
        "driver": "GTiff",
        "height": dst_height,
        "width": dst_width,
        "count": 4,
        "dtype": "uint8",
        "crs": dst_crs,
        "transform": dst_transform,
        "compress": compress,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic)
        dst.descriptions = NAIP_BANDS

    dates = [str(item.properties["datetime"]) for item in items]
    provenance = {
        "n_items": len(items),
        "item_ids": [item.id for item in items],
        "acq_dates": sorted(dates),
        "date_tag": median_date_tag(dates),
        "gsd": sorted({item.properties.get("gsd") for item in items} - {None}),
        "crs": str(dst_crs),
        "transform": list(dst_transform)[:6],
        "shape": [dst_height, dst_width],
        "coverage_fraction": coverage,
        "path": str(out_path),
    }
    if coverage < 0.999:
        logger.warning(
            "%s covers only %.3f of the target grid; check the search bounds",
            out_path.name,
            coverage,
        )
    logger.info("Wrote %s from %d quads", out_path.name, len(items))
    return out_path, provenance
