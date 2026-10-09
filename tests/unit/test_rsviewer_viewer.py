"""Smoke tests for rsviewer.viewer (no browser needed)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box

pytest.importorskip("leafmap")
pytest.importorskip("localtileserver")

from pyproj import Transformer  # noqa: E402

from rsviewer import (  # noqa: E402
    Category,
    RasterLayer,
    RasterStyle,
    VectorLayer,
    VectorStyle,
    Viewer,
)
from rsviewer.cog import array_to_cog  # noqa: E402
from rsviewer.query import lonlat_to_rowcol  # noqa: E402

TRANSFORM = from_origin(559425.0, 4330917.0, 9.6, 9.6)
CATS = {1: Category("a", "#ff0000"), 2: Category("b", "#00ff00")}


@pytest.fixture()
def files(tmp_path: Path) -> tuple[Path, Path]:
    arr = np.zeros((64, 64), "uint8")
    arr[10:20, 10:20] = 1
    arr[30:40, 30:40] = 2
    tif = array_to_cog(
        arr, tmp_path / "cls.tif", transform=TRANSFORM, crs=26916, nodata=0
    )
    x0, y0 = TRANSFORM * (10, 20)
    x1, y1 = TRANSFORM * (20, 10)
    gdf = gpd.GeoDataFrame(
        {"kind": ["a"], "id": [7]}, geometry=[box(x0, y0, x1, y1)], crs=26916
    )
    gpkg = tmp_path / "v.gpkg"
    gdf.to_file(gpkg, layer="polys")
    return tif, gpkg


def _click(v: Viewer, row: int, col: int) -> None:
    x, y = TRANSFORM * (col + 0.5, row + 0.5)
    lon, lat = Transformer.from_crs(26916, 4326, always_xy=True).transform(x, y)
    v._interaction(type="click", coordinates=[lat, lon])


def test_viewer_add_replace_inspect_and_click(files) -> None:
    tif, gpkg = files
    v = Viewer(title="t")
    try:
        assert v.add(RasterLayer("cls", path=tif, style=RasterStyle(categories=CATS)))
        assert v.add(
            VectorLayer(
                "polys",
                path=gpkg,
                layer="polys",
                style=VectorStyle(by="kind", categories=CATS and {"a": CATS[1]}),
            )
        )
        assert not v.add(RasterLayer("missing", path=tif.with_name("nope.tif")))
        assert v.names == ["cls", "polys"]
        # Raster sits below the vector layer.
        layers = list(v.map.layers)
        assert layers.index(v._entries["cls"].layer) < layers.index(
            v._entries["polys"].layer
        )

        seen = []
        v.on_click(lambda lon, lat: seen.append((lon, lat)))
        _click(v, 15, 15)
        assert len(seen) == 1
        assert "a" in v.info.value and "cls" in v.info.value

        v.set_visible("cls", False)
        v.set_opacity("cls", 0.4)
        old = v._entries["cls"].layer
        assert v.replace(
            "cls", RasterLayer("cls", path=tif, style=RasterStyle(categories=CATS))
        )
        new = v._entries["cls"].layer
        assert new is not old and new in v.map.layers and old not in v.map.layers
        assert new.visible is False and new.opacity == 0.4

        v.zoom_to("polys")
        v.remove("polys")
        assert v.names == ["cls"]
    finally:
        v.close()


def test_failing_click_handler_is_reported_not_raised(files) -> None:
    tif, _ = files
    v = Viewer()
    try:
        v.add(RasterLayer("cls", path=tif, style=RasterStyle(categories=CATS)))

        def broken(lon: float, lat: float) -> None:
            raise RuntimeError("boom")

        v.on_click(broken)
        _click(v, 15, 15)
        assert "boom" in v.messages.value or "failed" in v.messages.value
    finally:
        v.close()


def test_lonlat_helper_matches_grid() -> None:
    x, y = TRANSFORM * (3.5, 2.5)
    lon, lat = Transformer.from_crs(26916, 4326, always_xy=True).transform(x, y)
    assert lonlat_to_rowcol(lon, lat, TRANSFORM, 26916) == (2, 3)
