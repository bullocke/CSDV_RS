"""Tests for the review GeoPackage store."""

from __future__ import annotations

import pytest
from shapely.geometry import box

from csdv_core.review.sample import (
    draw_sample,
    extend_sample,
    list_runs,
    opportunistic_row,
)
from csdv_core.review.store import Interpretation, ReviewStore

REVIEW_TAG = "default"


@pytest.fixture()
def store(review_exports, tmp_path):
    sample = draw_sample(review_exports, REVIEW_TAG, 3)
    return ReviewStore.create(tmp_path / "ref" / "review_default.gpkg", sample)


def test_create_refuses_existing(store):
    with pytest.raises(FileExistsError):
        ReviewStore.create(store.path, store.sample)


def test_save_reload_and_first_open(store):
    assert store.first_open() == "R001"
    store.save(
        Interpretation(
            "R001",
            "confirmed",
            "clearcut",
            "2015-2018",
            "too_large",
            context="unit_edge,part_of_larger_event",
            confidence=3,
            notes="slash piles",
            seconds_spent=12.5,
        )
    )
    store.save(Interpretation("R002", "rejected", "artefact_seam"))
    again = ReviewStore(store.path)
    rec = again.get("R001")
    assert rec.verdict == "confirmed"
    assert rec.confidence == 3
    assert rec.context == "unit_edge,part_of_larger_event"
    assert rec.seconds_spent == 12.5
    assert rec.saved_at
    assert again.get("R002").confidence is None
    assert again.first_open() == "R003"
    assert again.progress() == (2, 6)
    assert again.next_open("R006") == "R003"


def test_resave_replaces_label(store):
    store.save(Interpretation("R001", "likely"))
    store.save(Interpretation("R001", "rejected"))
    assert len(store.labels) == 1
    assert ReviewStore(store.path).get("R001").verdict == "rejected"


def test_redrawn_geometry_round_trip_and_clear(store):
    lonlat = store.sample.to_crs(4326).geometry.iloc[0].buffer(-1e-5)
    store.save(Interpretation("R001", "confirmed"), geometry=lonlat)
    again = ReviewStore(store.path)
    assert again.get("R001").redrawn
    back = again.redrawn_geometry("R001")
    assert back.symmetric_difference(lonlat).area / lonlat.area < 1e-3
    # Saving without a drawing keeps it, clear_geometry removes it.
    again.save(Interpretation("R001", "confirmed"))
    assert again.redrawn_geometry("R001") is not None
    again.save(Interpretation("R001", "confirmed"), clear_geometry=True)
    assert ReviewStore(store.path).redrawn_geometry("R001") is None
    assert not ReviewStore(store.path).get("R001").redrawn


def test_log_keeps_every_save(store):
    store.save(Interpretation("R001", "likely"))
    store.save(Interpretation("R001", "rejected"), geometry=box(-71, 44, -70.99, 44.01))
    lines = store.log_path.read_text().strip().splitlines()
    assert len(lines) == 3  # header and two saves
    assert "MULTIPOLYGON" in lines[-1]


def test_external_extension_is_picked_up(store, review_exports):
    store.save(Interpretation("R001", "likely"))
    other = ReviewStore(store.path)
    other.set_sample(extend_sample(other.sample, review_exports, REVIEW_TAG, 5))
    # The first store reloads before saving, so neither change is lost.
    store.save(Interpretation("R002", "rejected"))
    final = ReviewStore(store.path)
    assert len(final.sample) == 10
    assert final.done_ids() == {"R001", "R002"}


def test_opportunistic_rows_are_not_in_the_random_progress(store, review_exports):
    run = list_runs(review_exports, REVIEW_TAG)[0]
    store.add_sample_rows(opportunistic_row(store.sample, run, 1))
    again = ReviewStore(store.path)
    assert again.ordered_ids()[-1] == "X001"
    assert again.progress() == (0, 6)
    assert "X001" not in again.ordered_ids(include_opportunistic=False)


def test_unknown_values_are_refused():
    with pytest.raises(ValueError):
        Interpretation("R001", "yes")
    with pytest.raises(ValueError):
        Interpretation("R001", "confirmed", change_type="bulldozer")
    with pytest.raises(ValueError):
        Interpretation("R001", "confirmed", context="cloudy")


def test_save_unknown_id(store):
    with pytest.raises(KeyError):
        store.save(Interpretation("R999", "confirmed"))
