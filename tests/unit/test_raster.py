"""Tests for csdv_core.io.raster."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from csdv_core.io.raster import (
    clip_to_bbox,
    find_latest_tif,
    read_band,
    write_raster,
)


def test_read_band_replaces_nodata_with_nan(synthetic_chm: Path) -> None:
    r = read_band(synthetic_chm)
    assert r.data.dtype == np.float32
    assert np.isnan(r.data[0, 0])
    assert not np.isnan(r.data[25, 25])
    assert r.crs is not None
    assert r.crs.to_epsg() == 5070


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    data = np.arange(100, dtype="float32").reshape(10, 10)
    transform = from_origin(0.0, 100.0, 1.0, 1.0)
    out = tmp_path / "round.tif"
    write_raster(out, data, transform=transform, crs="EPSG:5070", nodata=-1.0)
    r = read_band(out)
    np.testing.assert_array_equal(r.data, data)


def test_clip_to_bbox_reduces_extent(synthetic_chm: Path, tmp_path: Path) -> None:
    with rasterio.open(synthetic_chm) as src:
        b = src.bounds
    # Clip to the upper-left quarter.
    midx = (b.left + b.right) / 2
    midy = (b.bottom + b.top) / 2
    out = tmp_path / "clip.tif"
    clip_to_bbox(synthetic_chm, (b.left, midy, midx, b.top), out)
    with rasterio.open(out) as dst:
        assert dst.width < 50
        assert dst.height < 50


def test_find_latest_tif(tmp_path: Path) -> None:
    (tmp_path / "NAIP_SCBI_2018.tif").touch()
    (tmp_path / "NAIP_SCBI_2023.tif").touch()
    (tmp_path / "NAIP_SCBI_2023_zoom.tif").touch()
    (tmp_path / "OTHER.tif").touch()
    result = find_latest_tif(tmp_path, "NAIP_SCBI")
    assert result is not None
    assert result.name == "NAIP_SCBI_2023.tif"


def test_find_latest_tif_no_match(tmp_path: Path) -> None:
    assert find_latest_tif(tmp_path, "DOES_NOT_EXIST") is None
