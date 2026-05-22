"""csdv_core.io.vector — geopandas-based vector I/O helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd

logger = logging.getLogger(__name__)


def read_vector(path: Path | str) -> gpd.GeoDataFrame:
    """Read a vector file via :func:`geopandas.read_file`."""
    return gpd.read_file(path)


def write_vector(
    gdf: gpd.GeoDataFrame,
    path: Path | str,
    driver: str | None = None,
) -> Path:
    """Write a GeoDataFrame, inferring driver from extension if not given.

    Args:
        gdf: Frame to write.
        path: Output file path.
        driver: Optional Fiona driver name (e.g. ``"GPKG"``, ``"GeoJSON"``).
            If omitted, the driver is inferred from the file extension.

    Returns:
        ``Path(path)``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if driver is None:
        ext = path.suffix.lower()
        driver = {
            ".gpkg": "GPKG",
            ".geojson": "GeoJSON",
            ".json": "GeoJSON",
            ".shp": "ESRI Shapefile",
            ".parquet": "Parquet",
        }.get(ext)
    if driver == "Parquet":
        gdf.to_parquet(path)
    else:
        gdf.to_file(path, driver=driver)
    return path
