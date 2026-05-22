"""Tests for csdv_core.io.vector."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point

from csdv_core.io.vector import read_vector, write_vector


def _toy_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"name": ["a", "b"], "geometry": [Point(0, 0), Point(1, 1)]},
        crs="EPSG:5070",
    )


def test_geojson_roundtrip(tmp_path: Path) -> None:
    gdf = _toy_gdf()
    out = tmp_path / "pts.geojson"
    write_vector(gdf, out)
    back = read_vector(out)
    assert len(back) == 2
    assert set(back["name"]) == {"a", "b"}


def test_gpkg_roundtrip(tmp_path: Path) -> None:
    gdf = _toy_gdf()
    out = tmp_path / "pts.gpkg"
    write_vector(gdf, out)
    back = read_vector(out)
    assert len(back) == 2
