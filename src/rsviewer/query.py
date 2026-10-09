"""rsviewer.query: map clicks to raster pixels.

Leaflet reports clicks as latitude and longitude. Click handlers usually need
the row and column of a raster in its own CRS, or the values of every band at
that point. These helpers keep the CRS handling in one place.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import Transformer

logger = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def _from_lonlat(crs_wkt: str) -> Transformer:
    return Transformer.from_crs("EPSG:4326", crs_wkt, always_xy=True)


def lonlat_to_xy(lon: float, lat: float, crs: Any) -> tuple[float, float]:
    """Project a WGS84 point into ``crs``."""
    wkt = rasterio.crs.CRS.from_user_input(crs).to_wkt()
    return _from_lonlat(wkt).transform(lon, lat)


def lonlat_to_rowcol(
    lon: float,
    lat: float,
    transform: Any,
    crs: Any,
    shape: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    """Row and column of the pixel containing a WGS84 point.

    Args:
        lon: Longitude in degrees.
        lat: Latitude in degrees.
        transform: Affine transform of the grid.
        crs: CRS of the grid.
        shape: ``(rows, cols)``. When given, points outside return ``None``.

    Returns:
        ``(row, col)``, or ``None`` outside ``shape``.
    """
    x, y = lonlat_to_xy(lon, lat, crs)
    col, row = ~transform * (x, y)
    r, c = int(np.floor(row)), int(np.floor(col))
    if shape is not None and not (0 <= r < shape[0] and 0 <= c < shape[1]):
        return None
    return r, c


@dataclass(frozen=True)
class PointSample:
    """Values of every band of a raster at one pixel."""

    row: int
    col: int
    values: np.ndarray
    band_names: tuple[str, ...]


def sample_raster(
    path: Path, lon: float, lat: float, *, scale: float = 1.0
) -> PointSample | None:
    """Read all bands of a raster at a WGS84 point.

    Nodata becomes NaN and values are multiplied by ``scale``.

    Args:
        path: Raster path.
        lon: Longitude in degrees.
        lat: Latitude in degrees.
        scale: Multiplier from raw to display units.

    Returns:
        The sample, or ``None`` when the point is outside the raster.
    """
    with rasterio.open(path) as ds:
        rc = lonlat_to_rowcol(lon, lat, ds.transform, ds.crs, (ds.height, ds.width))
        if rc is None:
            return None
        window = rasterio.windows.Window(rc[1], rc[0], 1, 1)
        v = ds.read(window=window).reshape(-1).astype("float64")
        if ds.nodata is not None:
            v[v == ds.nodata] = np.nan
        names = tuple(d or f"band {i}" for i, d in enumerate(ds.descriptions, 1))
    return PointSample(rc[0], rc[1], v * scale, names)
