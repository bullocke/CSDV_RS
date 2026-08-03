"""Tests for the stand metric orchestrator."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from csdv_core.zonal.compute import (
    DateInputs,
    assert_common_grid,
    compute_module_metrics,
    compute_stand_record,
)

TRANSFORM = from_origin(west=0.0, north=120.0, xsize=0.6, ysize=0.6)
SHAPE = (200, 200)  # 120 x 120 m
CRS = "EPSG:26916"


def _write(
    path: Path,
    data: np.ndarray,
    *,
    count: int = 1,
    dtype: str = "float32",
    nodata: float | None = None,
    transform=TRANSFORM,
) -> Path:
    profile = {
        "driver": "GTiff",
        "height": data.shape[-2],
        "width": data.shape[-1],
        "count": count,
        "dtype": dtype,
        "transform": transform,
        "crs": CRS,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        if count == 1:
            dst.write(data.astype(dtype), 1)
        else:
            dst.write(data.astype(dtype))
    return path


def _chm(gap_fraction: float, seed: int = 0) -> np.ndarray:
    """A CHM where a given share of pixels is below 2 m."""
    rng = np.random.default_rng(seed)
    arr = rng.uniform(8.0, 22.0, size=SHAPE).astype(np.float32)
    n_gap = int(round(gap_fraction * arr.size))
    flat = rng.permutation(arr.size)[:n_gap]
    arr.ravel()[flat] = rng.uniform(0.0, 1.9, size=n_gap).astype(np.float32)
    return arr


@pytest.fixture()
def series(tmp_path: Path) -> list[DateInputs]:
    """Three dates on one grid, canopy closing over time."""
    rng = np.random.default_rng(42)
    dates = []
    for i, (year, gap) in enumerate([(2016, 0.60), (2018, 0.35), (2020, 0.15)]):
        chm_path = _write(tmp_path / f"chm_{year}.tif", _chm(gap, seed=i))
        naip = rng.integers(20, 220, size=(4, *SHAPE)).astype("uint8")
        naip_path = _write(tmp_path / f"naip_{year}.tif", naip, count=4, dtype="uint8")
        dates.append(
            DateInputs(
                year=year,
                date=f"{year}-07-01",
                native_res_m=0.6 if year >= 2016 else 1.0,
                chm_path=chm_path,
                naip_path=naip_path,
            )
        )
    return dates


@pytest.fixture()
def stands() -> gpd.GeoDataFrame:
    geoms = [box(10.0, 10.0, 60.0, 60.0), box(70.0, 70.0, 110.0, 110.0)]
    return gpd.GeoDataFrame(
        {"stand_id": ["S1", "S2"], "area_m2": [g.area for g in geoms]},
        geometry=geoms,
        crs=CRS,
    )


def test_assert_common_grid_accepts_matching_rasters(series):
    assert_common_grid([d.chm_path for d in series])


def test_assert_common_grid_rejects_a_shifted_raster(tmp_path: Path, series):
    shifted = _write(
        tmp_path / "shifted.tif",
        _chm(0.2),
        transform=from_origin(1.0, 120.0, 0.6, 0.6),
    )
    with pytest.raises(ValueError, match="not on the same grid"):
        assert_common_grid([series[0].chm_path, shifted])


def test_record_carries_the_core_metrics(series, stands):
    record = compute_stand_record("S1", stands.geometry[0], series[1])
    assert record.stand_id == "S1"
    assert record.year == 2018
    assert record.native_res_m == pytest.approx(0.6)
    assert 0.0 <= record.value("gap_fraction") <= 1.0
    assert record.value("crown_fraction") == pytest.approx(
        1.0 - record.value("gap_fraction")
    )
    assert np.isfinite(record.value("glcm_texture"))
    assert np.isfinite(record.value("edge_density"))


def test_first_date_has_no_gap_persistence_and_says_why(series, stands):
    record = compute_stand_record("S1", stands.geometry[0], series[0])
    assert np.isnan(record.value("gap_persistence"))
    assert "first date" in record.reasons["gap_persistence"]


def test_gap_persistence_uses_the_previous_date(series, stands):
    record = compute_stand_record(
        "S1", stands.geometry[0], series[1], prev_chm_path=series[0].chm_path
    )
    assert 0.0 <= record.value("gap_persistence") <= 1.0
    assert "gap_persistence" not in record.reasons


def test_crown_metrics_report_a_reason_when_no_segmentation_is_supplied(series, stands):
    record = compute_stand_record("S1", stands.geometry[0], series[0])
    assert np.isnan(record.value("crown_cv"))
    assert record.reasons["crown_cv"] == "no crown segmentation supplied"


def test_crown_metrics_use_the_supplied_crowns(series, stands):
    crowns = gpd.GeoDataFrame(
        {
            "segment_id": range(8),
            "crown_diam_m": [3.0, 4.0, 5.0, 6.0, 3.5, 4.5, 5.5, 6.5],
        },
        geometry=[Point(15.0 + 5 * i, 20.0).buffer(2.0) for i in range(8)],
        crs=CRS,
    )
    record = compute_stand_record("S1", stands.geometry[0], series[0], crowns=crowns)
    assert record.support["n_crowns"] == pytest.approx(8.0)
    assert np.isfinite(record.value("crown_cv"))
    assert "crown_cv" not in record.reasons


def test_texture_reports_a_reason_without_naip(series, stands):
    no_naip = DateInputs(
        year=series[0].year,
        date=series[0].date,
        native_res_m=0.6,
        chm_path=series[0].chm_path,
    )
    record = compute_stand_record("S1", stands.geometry[0], no_naip)
    assert np.isnan(record.value("glcm_texture"))
    assert record.reasons["glcm_texture"] == "no NAIP image supplied"


def test_nodata_fraction_is_reported(tmp_path: Path, stands):
    holed = _chm(0.3)
    # The stand covers world x 10..60, y 10..60, which is rows 100..183 and
    # cols 17..100 on this grid.
    holed[110:150, 30:70] = -9999.0
    path = _write(tmp_path / "holed.tif", holed, nodata=-9999.0)
    date = DateInputs(2018, "2018-07-01", 0.6, path)
    record = compute_stand_record("S1", stands.geometry[0], date)
    assert record.support["nodata_fraction"] > 0.0


def test_a_misaligned_previous_date_raises(tmp_path: Path, series, stands):
    shifted = _write(
        tmp_path / "prev_shifted.tif",
        _chm(0.4),
        transform=from_origin(1.0, 120.0, 0.6, 0.6),
    )
    with pytest.raises(ValueError, match="not on the same grid"):
        compute_stand_record("S1", stands.geometry[0], series[1], prev_chm_path=shifted)


def test_module_metrics_produce_one_row_per_stand_per_date(series, stands):
    frame = compute_module_metrics(stands, series)
    assert len(frame) == len(stands) * len(series)
    assert set(frame["stand_id"]) == {"S1", "S2"}
    assert sorted(frame["year"].unique()) == [2016, 2018, 2020]


def test_module_metrics_track_a_closing_canopy(series, stands):
    frame = compute_module_metrics(stands, series)
    s1 = frame[frame["stand_id"] == "S1"].sort_values("year")
    assert list(s1["gap_fraction"]) == sorted(s1["gap_fraction"], reverse=True)
    # Change metrics come along, and the first date of the stand is NaN.
    assert np.isnan(s1.iloc[0]["d_gap_fraction"])
    assert s1.iloc[1]["d_gap_fraction"] < 0


def test_module_metrics_can_skip_change_metrics(series, stands):
    frame = compute_module_metrics(stands, series, with_change_metrics=False)
    assert "d_gap_fraction" not in frame.columns


def test_a_stand_outside_the_raster_is_skipped_not_fatal(series, stands):
    far = gpd.GeoDataFrame(
        {"stand_id": ["OFF"], "area_m2": [100.0]},
        geometry=[box(9000.0, 9000.0, 9100.0, 9100.0)],
        crs=CRS,
    )
    frame = compute_module_metrics(
        gpd.GeoDataFrame(pd.concat([stands, far], ignore_index=True), crs=CRS),
        series,
    )
    assert "OFF" not in set(frame["stand_id"])
    assert len(frame) == len(stands) * len(series)
