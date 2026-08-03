"""Tests for the worked-example stand screen."""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import box

from csdv_core.examples.screen import ACRE_M2, ScreenCriteria, screen_stands

YEARS = [2012, 2014, 2016, 2018, 2020, 2022]


def _stands(rows: list[dict[str, object]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        [
            {
                "stand_id": r.get("stand_id", f"S{i}"),
                "dist_label": r.get("dist_label", "Clearcut"),
                "dist_group": r.get("dist_group", "Harvest"),
                "area_m2": float(r.get("acres", 10.0)) * ACRE_M2,
                "bbox_fill": r.get("bbox_fill", 0.5),
                "LastImageryPreDist": r.get("last_pre", 2016),
                "FirstImageryPostDist": r.get("first_post", 2017),
                "WithinMappingarea": r.get("within", 1),
                "AdditionalDisturbance": r.get("additional", 0),
            }
            for i, r in enumerate(rows)
        ],
        geometry=[box(0, 0, 1, 1) for _ in rows],
        crs="EPSG:26916",
    )


def test_a_clean_stand_passes():
    out = screen_stands(_stands([{}]), ScreenCriteria(years=YEARS))
    assert bool(out.loc[0, "passes"])
    assert out.loc[0, "fails"] == ""


def test_counts_dates_either_side_of_the_disturbance():
    out = screen_stands(
        _stands([{"last_pre": 2016, "first_post": 2017}]), ScreenCriteria(years=YEARS)
    )
    assert out.loc[0, "n_pre_dates"] == 3  # 2012, 2014, 2016
    assert out.loc[0, "n_post_dates"] == 3  # 2018, 2020, 2022


def test_a_disturbance_before_the_record_fails_and_says_so():
    out = screen_stands(
        _stands([{"last_pre": 2007, "first_post": 2008}]), ScreenCriteria(years=YEARS)
    )
    assert not bool(out.loc[0, "passes"])
    assert "no imagery before the disturbance" in out.loc[0, "fails"]
    # It still has a full recovery limb, which the counts record.
    assert out.loc[0, "n_post_dates"] == 6


def test_the_pre_date_requirement_can_be_relaxed():
    stands = _stands([{"last_pre": 2007, "first_post": 2008}])
    out = screen_stands(stands, ScreenCriteria(years=YEARS, require_pre_date=False))
    assert bool(out.loc[0, "passes"])


def test_a_small_stand_fails():
    out = screen_stands(_stands([{"acres": 1.2}]), ScreenCriteria(years=YEARS))
    assert "minimum mapping unit" in out.loc[0, "fails"]


def test_a_truncated_footprint_fails():
    out = screen_stands(_stands([{"within": 0}]), ScreenCriteria(years=YEARS))
    assert "truncated" in out.loc[0, "fails"]


def test_a_second_disturbance_fails_by_default():
    out = screen_stands(_stands([{"additional": 1}]), ScreenCriteria(years=YEARS))
    assert "second disturbance" in out.loc[0, "fails"]


def test_the_single_event_rule_can_be_relaxed_for_salvage():
    """Salvage after a windthrow is a second disturbance, and showing it is the
    point of that example rather than a reason to drop the stand."""
    stands = _stands([{"additional": 1}])
    out = screen_stands(stands, ScreenCriteria(years=YEARS, require_single_event=False))
    assert bool(out.loc[0, "passes"])


def test_every_failed_criterion_is_named():
    out = screen_stands(
        _stands([{"acres": 1.0, "within": 0, "additional": 1}]),
        ScreenCriteria(years=YEARS),
    )
    fails = out.loc[0, "fails"]
    assert "minimum mapping unit" in fails
    assert "truncated" in fails
    assert "second disturbance" in fails


def test_results_are_sorted_with_passing_stands_first():
    out = screen_stands(
        _stands(
            [
                {"stand_id": "small", "acres": 1.0},
                {"stand_id": "big", "acres": 100.0},
                {"stand_id": "mid", "acres": 20.0},
            ]
        ),
        ScreenCriteria(years=YEARS),
    )
    assert list(out["stand_id"]) == ["big", "mid", "small"]


def test_the_real_delivery_screens_without_error():
    from pathlib import Path

    from csdv_core.io.stands import read_ais_stands

    gdb = Path(
        "data/calibration/Indiana-ElkinsvilleNE_revised.gdb/"
        "Indiana-ElkinsvilleNE_revised.gdb"
    )
    if not gdb.exists():
        pytest.skip("Calibration delivery not present")
    out = screen_stands(read_ais_stands(gdb), ScreenCriteria(years=YEARS))
    assert len(out) == 40
    assert out["passes"].sum() > 0
