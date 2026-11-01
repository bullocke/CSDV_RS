"""Tests for the review sample (equal allocation, stable IDs, extension)."""

from __future__ import annotations

import pytest

from csdv_core.review.sample import (
    STORE_CRS,
    STRATUM_OPPORTUNISTIC,
    draw_sample,
    extend_sample,
    list_runs,
    opportunistic_row,
)

REVIEW_TAG = "default"
REVIEW_UNITS = [("q4407159", "EPSG:26919", 12), ("q3912112", "EPSG:26910", 5)]
REVIEW_YEARS = [2015, 2018, 2021]


def test_list_runs_reads_years_labels_and_crs(review_exports):
    runs = {r.unit: r for r in list_runs(review_exports, REVIEW_TAG)}
    assert set(runs) == {u for u, _, _ in REVIEW_UNITS}
    run = runs["q4407159"]
    assert run.years == REVIEW_YEARS
    assert run.intervals == ["2015-2018", "2018-2021"]
    assert run.label == "Site_q4407159"
    assert "26919" in run.crs


def test_equal_allocation_and_ids(review_exports):
    s = draw_sample(review_exports, REVIEW_TAG, 4)
    counts = s.groupby("unit").size().to_dict()
    # The small unit has 5 objects, so both give 4.
    assert counts == {"q4407159": 4, "q3912112": 4}
    assert s["sample_id"].tolist() == [f"R{i:03d}" for i in range(1, 9)]
    assert s["order"].tolist() == list(range(1, 9))
    assert s.crs.to_string() == STORE_CRS
    assert s.loc[s["unit"] == "q4407159", "incl_prob"].eq(4 / 12).all()
    assert s.loc[s["unit"] == "q3912112", "incl_prob"].eq(4 / 5).all()


def test_units_with_fewer_objects_give_all(review_exports):
    s = draw_sample(review_exports, REVIEW_TAG, 8)
    assert (s["unit"] == "q3912112").sum() == 5
    assert s.loc[s["unit"] == "q3912112", "incl_prob"].eq(1.0).all()


def test_ids_interleave_units(review_exports):
    s = draw_sample(review_exports, REVIEW_TAG, 4)
    # A shuffle across units, not one block per unit.
    blocks = (s["unit"] != s["unit"].shift()).sum()
    assert blocks > 2


def test_extend_keeps_ids_and_matches_single_draw(review_exports):
    first = draw_sample(review_exports, REVIEW_TAG, 3)
    grown = extend_sample(first, review_exports, REVIEW_TAG, 6)
    single = draw_sample(review_exports, REVIEW_TAG, 6)
    head = grown.iloc[: len(first)]
    assert head["sample_id"].tolist() == first["sample_id"].tolist()
    assert head["object_id"].tolist() == first["object_id"].tolist()
    assert set(zip(grown["unit"], grown["object_id"], strict=True)) == set(
        zip(single["unit"], single["object_id"], strict=True)
    )
    assert grown["batch"].max() == 2
    assert grown["sample_id"].is_unique
    big = grown["unit"] == "q4407159"
    assert grown.loc[big, "incl_prob"].eq(6 / 12).all()


def test_extend_to_same_size_adds_nothing(review_exports):
    first = draw_sample(review_exports, REVIEW_TAG, 3)
    again = extend_sample(first, review_exports, REVIEW_TAG, 3)
    assert len(again) == len(first)


def test_assessed_geometry_prefers_refined_outline(review_exports):
    s = draw_sample(review_exports, REVIEW_TAG, 20)
    refined = s["geom_src"] == "refined"
    # Odd object IDs (1, 3, ...) have refined outlines in the fixture.
    assert (s.loc[refined, "object_id"] % 2 == 1).all()
    assert (s.loc[~refined, "object_id"] % 2 == 0).all()
    assert (s.loc[refined, "assessed_ha"] < s.loc[refined, "area_ha"]).all()
    assert (
        s.loc[~refined, "assessed_ha"]
        .round(4)
        .eq(s.loc[~refined, "area_ha"].round(4))
        .all()
    )


def test_exclude_growth(review_exports):
    s = draw_sample(review_exports, REVIEW_TAG, 20, exclude_growth=True)
    assert not s["growth"].any()
    assert s.loc[s["unit"] == "q4407159", "n_objects_unit"].eq(8).all()


def test_opportunistic_row(review_exports):
    s = draw_sample(review_exports, REVIEW_TAG, 2)
    run = list_runs(review_exports, REVIEW_TAG)[0]
    row = opportunistic_row(s, run, 3)
    assert row["sample_id"].iloc[0] == "X001"
    assert row["stratum"].iloc[0] == STRATUM_OPPORTUNISTIC
    assert row["incl_prob"].isna().all()
    with pytest.raises(KeyError):
        opportunistic_row(s, run, 999)
