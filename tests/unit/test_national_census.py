"""Unit tests for csdv_core.national.census.

Uses a small synthetic COG with internal overviews, written to a temporary
directory, so no network access is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from csdv_core.national.census import (
    canopy_metrics,
    census_tile,
    load_census,
    run_census,
)


def _write_tile(path: Path, *, forest_cols: int = 256, size: int = 512) -> Path:
    """uint16 cm CHM, left ``forest_cols`` columns at 20 m, rest bare, with overviews."""
    data = np.zeros((size, size), dtype="uint16")
    data[:, :forest_cols] = 2000
    data[:16, :16] = 65535
    profile = {
        "driver": "GTiff",
        "height": size,
        "width": size,
        "count": 1,
        "dtype": "uint16",
        "nodata": 65535,
        "crs": "EPSG:26918",
        "transform": from_origin(300000.0, 4300000.0, 0.6, 0.6),
        "tiled": True,
        "blockxsize": 128,
        "blockysize": 128,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.build_overviews([2, 4, 8, 16, 32], Resampling.average)
    return path


def test_canopy_metrics_fractions_over_valid_pixels() -> None:
    h = np.array([[np.nan, 1.0, 3.0, 20.0]])
    m = canopy_metrics(h)
    assert m["valid_frac"] == pytest.approx(0.75)
    assert m["frac_ge2"] == pytest.approx(2 / 3)
    assert m["frac_ge5"] == pytest.approx(1 / 3)
    assert m["frac_ge15"] == pytest.approx(1 / 3)
    assert m["mean_h_ge2"] == pytest.approx(11.5)


def test_canopy_metrics_all_nodata() -> None:
    m = canopy_metrics(np.full((3, 3), np.nan))
    assert m["valid_frac"] == 0.0
    assert np.isnan(m["frac_ge2"])


def test_census_tile_reads_requested_overview(tmp_path: Path) -> None:
    tile = _write_tile(tmp_path / "t.tif")
    res = census_tile("t", str(tile), decimation=8, scale_factor=0.01)
    assert res.error == ""
    assert res.decimation == 8
    assert res.pixel_m == pytest.approx(4.8)
    assert (res.full_width, res.full_height) == (512, 512)
    assert res.frac_ge2 == pytest.approx(0.5, abs=0.02)
    assert res.mean_h_ge2 == pytest.approx(20.0, abs=0.5)


def test_census_tile_records_failure(tmp_path: Path) -> None:
    res = census_tile("missing", str(tmp_path / "nope.tif"))
    assert res.error
    assert np.isnan(res.frac_ge2)


def test_run_census_resumes(tmp_path: Path) -> None:
    tiles = pd.DataFrame(
        {
            "tile_id": ["a", "b", "c"],
            "chm_url": [
                str(_write_tile(tmp_path / f"{t}.tif", forest_cols=c))
                for t, c in (("a", 0), ("b", 256), ("c", 512))
            ],
            "scale_factor": 0.01,
        }
    )
    out = tmp_path / "census"
    run_census(tiles, out, workers=2, chunk_size=2, decimation=8, limit=2)
    assert len(load_census(out)) == 2
    run_census(tiles, out, workers=2, chunk_size=2, decimation=8)
    cen = load_census(out).set_index("tile_id")
    assert sorted(cen.index) == ["a", "b", "c"]
    assert cen.loc["a", "frac_ge2"] == pytest.approx(0.0)
    assert cen.loc["c", "frac_ge2"] == pytest.approx(1.0)
    assert cen.loc["b", "full_px"] == 512 * 512
