"""Tests for the comment points of the map viewer."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pytest

from csdv_core.view.notes import NoteStore


def _log(store: NoteStore) -> list[dict]:
    return [json.loads(line) for line in store.log_path.read_text().splitlines()]


def test_put_writes_a_geojson_that_geopandas_reads(tmp_path: Path) -> None:
    store = NoteStore(tmp_path / "notes" / "pair.geojson")
    feature = store.put(
        "n1",
        -71.68,
        41.56,
        {"unit": "q4107127", "tag": "missed", "comment": " a gap ", "junk": 1},
    )
    assert feature["properties"]["comment"] == "a gap"
    assert "junk" not in feature["properties"]
    frame = gpd.read_file(store.path)
    assert len(frame) == 1
    assert frame.crs.to_epsg() == 4326
    assert frame.geometry.iloc[0].x == pytest.approx(-71.68)
    assert frame["tag"].iloc[0] == "missed"
    assert not store.path.with_name("pair.geojson.tmp").exists()


def test_put_replaces_and_keeps_the_time_of_creation(tmp_path: Path) -> None:
    store = NoteStore(tmp_path / "pair.geojson")
    first = store.put("n1", 1.0, 2.0, {"comment": "one"})
    second = store.put("n1", 1.0, 2.0, {"comment": "two", "tag": "good"})
    assert len(store) == 1
    assert second["properties"]["created"] == first["properties"]["created"]
    assert second["properties"]["comment"] == "two"
    assert [r["action"] for r in _log(store)] == ["put", "put"]


def test_an_empty_note_is_not_kept(tmp_path: Path) -> None:
    store = NoteStore(tmp_path / "pair.geojson")
    assert store.put("n1", 1.0, 2.0, {"comment": "  ", "tag": ""}) is None
    assert len(store) == 0
    assert not store.path.exists()
    store.put("n1", 1.0, 2.0, {"comment": "text"})
    assert store.put("n1", 1.0, 2.0, {"comment": ""}) is None
    assert len(store) == 0
    assert [r["action"] for r in _log(store)] == ["put", "delete"]
    assert _log(store)[-1]["note"]["comment"] == "text"


def test_delete_and_reload(tmp_path: Path) -> None:
    path = tmp_path / "pair.geojson"
    store = NoteStore(path)
    store.put("n1", 1.0, 2.0, {"comment": "one"})
    store.put("n2", 3.0, 4.0, {"tag": "good"})
    assert store.delete("n1") is True
    assert store.delete("n1") is False
    again = NoteStore(path)
    assert [f["properties"]["id"] for f in again.collection()["features"]] == ["n2"]


def test_bad_values_are_refused(tmp_path: Path) -> None:
    store = NoteStore(tmp_path / "pair.geojson")
    with pytest.raises(ValueError):
        store.put("n1", 1.0, 2.0, {"tag": "nonsense"})
    with pytest.raises(ValueError):
        store.put("n1", 500.0, 2.0, {"comment": "x"})


def test_a_change_on_disk_is_read_before_the_next_write(tmp_path: Path) -> None:
    path = tmp_path / "pair.geojson"
    store = NoteStore(path)
    store.put("n1", 1.0, 2.0, {"comment": "one"})
    other = NoteStore(path)
    other.put("n2", 3.0, 4.0, {"comment": "two"})
    store.put("n3", 5.0, 6.0, {"comment": "three"})
    ids = {f["properties"]["id"] for f in store.collection()["features"]}
    assert ids == {"n1", "n2", "n3"}
