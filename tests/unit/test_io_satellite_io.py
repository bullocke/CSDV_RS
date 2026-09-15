"""Tests for csdv_core.io.satellite_io."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from csdv_core.io.satellite_io import (
    ANNUAL_NAME,
    MANIFEST_NAME,
    OBSERVATIONS_NAME,
    join_satellite_metrics,
    read_annual,
    read_manifest,
    read_observations,
    write_annual,
    write_observations,
)
from csdv_core.satellite.annual import REASON_COLUMN, SUPPORT_PREFIX
from csdv_core.satellite.extract import OBSERVATION_COLUMNS


def _observations(image_ids: list[str], *, stand_id: str = "S1") -> pd.DataFrame:
    n = len(image_ids)
    frame = pd.DataFrame(
        {
            "stand_id": [stand_id] * n,
            "image_id": image_ids,
            "sensor": ["L5"] * n,
            "date": ["2005-07-04"] * n,
            "year": [2005] * n,
            "doy": [185] * n,
            "time_ms": list(range(1_120_435_200_000, 1_120_435_200_000 + n)),
            "wrs_path": [21] * n,
            "wrs_row": [33] * n,
            "scene_cloud_cover": [4.0] * n,
            "n_pixels": [36] * n,
            "pixel_weight_sum": [32.4] * n,
            "area_m2": [32400.0] * n,
            "expected_pixels": [36.0] * n,
            "coverage_fraction": [0.9] * n,
        }
    )
    return frame.astype(OBSERVATION_COLUMNS)


_PROVENANCE = {
    "sensors": ["L5", "L7"],
    "indices": ["ndvi"],
    "start_year": 2005,
    "end_year": 2005,
    "chunks_requested": ["2005"],
    "chunks_failed": {},
    "scale_m": 30.0,
    "pixel_rule": "ee_area_weighted",
    "crs": "native scene projection (no reprojection)",
}


def test_observation_round_trip_preserves_dtypes(tmp_path: Path) -> None:
    frame = _observations(["a", "b"])
    parquet, manifest = write_observations(frame, tmp_path, _PROVENANCE)
    assert parquet.name == OBSERVATIONS_NAME and manifest.name == MANIFEST_NAME
    back = read_observations(tmp_path)
    for name, dtype in OBSERVATION_COLUMNS.items():
        assert str(back[name].dtype) == dtype, f"{name} came back as {back[name].dtype}"
    pd.testing.assert_frame_equal(frame, back)


def test_manifest_records_the_rules_the_product_was_made_under(tmp_path: Path) -> None:
    write_observations(_observations(["a"]), tmp_path, _PROVENANCE)
    manifest = read_manifest(tmp_path)
    # The pixel rule differs from csdv_core.zonal.mask, so a product on disk has
    # to say which rule made it.
    assert manifest["pixel_rule"] == "ee_area_weighted"
    assert manifest["scale_m"] == 30.0
    assert manifest["sensors"] == ["L5", "L7"]
    assert manifest["n_observations"] == 1


def test_appending_deduplicates_on_stand_and_scene(tmp_path: Path) -> None:
    write_observations(_observations(["a", "b"]), tmp_path, _PROVENANCE)
    # Refetching a year that partly failed must not double its rows.
    write_observations(_observations(["b", "c"]), tmp_path, _PROVENANCE)
    back = read_observations(tmp_path)
    assert sorted(back["image_id"]) == ["a", "b", "c"]
    assert read_manifest(tmp_path)["n_observations"] == 3


def test_appending_widens_the_year_span_and_clears_a_recovered_chunk(
    tmp_path: Path,
) -> None:
    first = {**_PROVENANCE, "chunks_failed": {"2005": "Computation timed out."}}
    write_observations(_observations(["a"]), tmp_path, first)
    assert read_manifest(tmp_path)["chunks_failed"] == {
        "2005": "Computation timed out."
    }

    second = {
        **_PROVENANCE,
        "start_year": 2006,
        "end_year": 2006,
        "chunks_requested": ["2005", "2006"],
    }
    write_observations(_observations(["b"]), tmp_path, second)
    manifest = read_manifest(tmp_path)
    assert manifest["chunks_failed"] == {}  # 2005 succeeded this time
    assert manifest["start_year"] == 2005 and manifest["end_year"] == 2006
    assert manifest["chunks_requested"] == ["2005", "2006"]


def test_replace_mode_does_not_append(tmp_path: Path) -> None:
    write_observations(_observations(["a", "b"]), tmp_path, _PROVENANCE)
    write_observations(_observations(["c"]), tmp_path, _PROVENANCE, append=False)
    assert read_observations(tmp_path)["image_id"].tolist() == ["c"]


# --------------------------------------------------------------------------
# The join
# --------------------------------------------------------------------------
def _annual() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stand_id": ["S1", "S1", "S2", "S2"],
            "year": [2018, 2020, 2018, 2020],
            "area_m2": [32400.0] * 4,
            "ndvi_mean": [0.88, 0.90, 0.40, np.nan],
            "ndvi_seasonal_amplitude": [0.55, 0.58, 0.72, np.nan],
            "ndvi_trend": [0.01, 0.01, 0.06, np.nan],
            f"{SUPPORT_PREFIX}n_observations": [14, 16, 14, 0],
            REASON_COLUMN: [
                "",
                "",
                "",
                "ndvi_mean: n_obs=0 in DOY 152-258 < min_obs=3",
            ],
        }
    )


def _stand_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stand_id": ["S1", "S1", "S2", "S2"],
            "year": [2018, 2020, 2018, 2020],
            "date": ["2018-08-12", "2020-06-12"] * 2,
            "gap_fraction": [0.05, 0.04, 0.80, 0.60],
            "unavailable": ["", "crown_cv: too few crowns", "", ""],
        }
    )


def test_join_attaches_the_metrics_on_stand_and_year() -> None:
    merged = join_satellite_metrics(_stand_metrics(), _annual())
    assert len(merged) == 4
    row = merged[(merged["stand_id"] == "S1") & (merged["year"] == 2018)].iloc[0]
    assert row["ndvi_mean"] == pytest.approx(0.88)
    assert row["ndvi_seasonal_amplitude"] == pytest.approx(0.55)
    assert row[f"{SUPPORT_PREFIX}n_observations"] == 14


def test_join_drops_no_stand_date_and_leaves_missing_years_nan() -> None:
    annual = _annual().drop(index=[1]).reset_index(drop=True)  # remove S1 2020
    merged = join_satellite_metrics(_stand_metrics(), annual)
    assert len(merged) == 4
    row = merged[(merged["stand_id"] == "S1") & (merged["year"] == 2020)].iloc[0]
    assert np.isnan(row["ndvi_mean"])


def test_join_concatenates_the_two_reason_columns_rather_than_overwriting() -> None:
    merged = join_satellite_metrics(_stand_metrics(), _annual())
    kept = merged[(merged["stand_id"] == "S1") & (merged["year"] == 2020)].iloc[0]
    assert "crown_cv: too few crowns" in kept["unavailable"]

    added = merged[(merged["stand_id"] == "S2") & (merged["year"] == 2020)].iloc[0]
    assert "ndvi_mean" in added["unavailable"]
    assert REASON_COLUMN not in merged.columns


def test_join_refuses_to_define_a_metric_name_twice() -> None:
    stand_metrics = _stand_metrics().assign(ndvi_mean=0.5)
    with pytest.raises(ValueError, match="ndvi_mean"):
        join_satellite_metrics(stand_metrics, _annual())


def test_annual_round_trip(tmp_path: Path) -> None:
    path = write_annual(_annual(), tmp_path)
    assert path.name == ANNUAL_NAME
    pd.testing.assert_frame_equal(_annual(), read_annual(tmp_path))
