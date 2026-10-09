"""Tests for rsviewer.cog."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

pytest.importorskip("rio_cogeo")
from rio_cogeo.cogeo import cog_validate  # noqa: E402

from rsviewer.cog import array_to_cog, file_to_cog, is_stale  # noqa: E402

TRANSFORM = from_origin(559425.0, 4330917.0, 9.6, 9.6)


def test_array_to_cog_writes_valid_cog_with_metadata(tmp_path: Path) -> None:
    arr = np.arange(3 * 600 * 500, dtype="int16").reshape(3, 600, 500)
    out = array_to_cog(
        arr,
        tmp_path / "a" / "x.tif",
        transform=TRANSFORM,
        crs="EPSG:26916",
        nodata=-32768,
        band_names=["2014", "2016", "2018"],
    )
    assert cog_validate(str(out))[0]
    with rasterio.open(out) as ds:
        assert ds.crs.to_epsg() == 26916
        assert ds.transform == TRANSFORM
        assert ds.nodata == -32768
        assert ds.descriptions == ("2014", "2016", "2018")
        assert ds.overviews(1)
        np.testing.assert_array_equal(ds.read(), arr)
    assert not (tmp_path / "a" / "x.tmp.tif").exists()


def test_array_to_cog_float_nan_becomes_nodata(tmp_path: Path) -> None:
    arr = np.ones((40, 30), "float32")
    arr[0, 0] = np.nan
    out = array_to_cog(
        arr, tmp_path / "f.tif", transform=TRANSFORM, crs=26916, nodata=-9999
    )
    with rasterio.open(out) as ds:
        a = ds.read(1)
    assert a[0, 0] == -9999
    assert a[1, 1] == 1


def test_array_to_cog_rejects_bad_shapes(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        array_to_cog(np.zeros(5), tmp_path / "x.tif", transform=TRANSFORM, crs=26916)
    with pytest.raises(ValueError):
        array_to_cog(
            np.zeros((2, 4, 4)),
            tmp_path / "x.tif",
            transform=TRANSFORM,
            crs=26916,
            band_names=["one"],
        )


def test_file_to_cog_and_staleness(synthetic_chm: Path, tmp_path: Path) -> None:
    dst = tmp_path / "chm_cog.tif"
    assert is_stale(dst, synthetic_chm)
    file_to_cog(synthetic_chm, dst)
    assert cog_validate(str(dst))[0]
    assert not is_stale(dst, synthetic_chm)
    later = time.time() + 10
    os.utime(synthetic_chm, (later, later))
    assert is_stale(dst, synthetic_chm)


def test_file_to_cog_missing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        file_to_cog(tmp_path / "nope.tif", tmp_path / "out.tif")
