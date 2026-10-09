"""Tests for rsviewer.query."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from rsviewer.query import lonlat_to_rowcol, sample_raster

TRANSFORM = from_origin(559425.0, 4330917.0, 9.6, 9.6)
TO_LONLAT = Transformer.from_crs("EPSG:26916", "EPSG:4326", always_xy=True)


def _cell_center_lonlat(row: int, col: int) -> tuple[float, float]:
    x, y = TRANSFORM * (col + 0.5, row + 0.5)
    return TO_LONLAT.transform(x, y)


def test_lonlat_to_rowcol_round_trips_cell_centers() -> None:
    for row, col in [(0, 0), (181, 142), (726, 568)]:
        lon, lat = _cell_center_lonlat(row, col)
        assert lonlat_to_rowcol(lon, lat, TRANSFORM, "EPSG:26916", (727, 569)) == (
            row,
            col,
        )


def test_lonlat_to_rowcol_outside_returns_none() -> None:
    lon, lat = _cell_center_lonlat(-3, 10)
    assert lonlat_to_rowcol(lon, lat, TRANSFORM, "EPSG:26916", (727, 569)) is None
    # Without a shape the caller gets the raw (negative) row.
    assert lonlat_to_rowcol(lon, lat, TRANSFORM, "EPSG:26916")[0] == -3


def test_sample_raster_scales_and_masks_nodata(tmp_path: Path) -> None:
    arr = np.full((3, 20, 20), 150, "int16")
    arr[1, 5, 7] = -32768
    path = tmp_path / "z.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=20,
        height=20,
        count=3,
        dtype="int16",
        crs="EPSG:26916",
        transform=TRANSFORM,
        nodata=-32768,
    ) as ds:
        ds.write(arr)
        ds.set_band_description(1, "2014")
    lon, lat = _cell_center_lonlat(5, 7)
    smp = sample_raster(path, lon, lat, scale=0.01)
    assert (smp.row, smp.col) == (5, 7)
    np.testing.assert_allclose(smp.values, [1.5, np.nan, 1.5])
    assert smp.band_names[0] == "2014"
    far_lon, far_lat = _cell_center_lonlat(50, 50)
    assert sample_raster(path, far_lon, far_lat) is None
