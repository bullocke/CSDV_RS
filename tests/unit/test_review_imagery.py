"""Tests for the NAIP tile proxy of the review app (mosaics that expire)."""

from __future__ import annotations

import json
import time

import pytest

from csdv_core.review import imagery


class _Response:
    def __init__(self, status: int, payload: dict | None = None, content=b"") -> None:
        self.status_code = status
        self._payload = payload or {}
        self.content = content
        self.headers = {"content-type": "image/jpeg"}

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


@pytest.fixture()
def fake_pc(tmp_path, monkeypatch):
    """A fake data API: registrations and tiles, with controllable expiry."""
    cache = tmp_path / "pc_items.json"
    cache.write_text(json.dumps({"q1": {"2019": ["a", "b"]}}))
    monkeypatch.setattr(imagery, "default_pc_cache", lambda: cache)
    monkeypatch.setattr(imagery, "_MOSAICS", imagery._Mosaics())
    imagery.pc_tile.cache_clear()
    state = {"n": 0, "expiry": time.time() + 86400, "expired": set()}

    def post(url, json=None, timeout=None):
        state["n"] += 1
        return _Response(200, {"searchid": f"pc1.{int(state['expiry'])}.h{state['n']}"})

    def get(url, params=None, timeout=None):
        search_id = url.split("/mosaic/")[1].split("/")[0]
        if search_id in state["expired"]:
            return _Response(410)
        if url.endswith("/0/0/0"):
            return _Response(404)
        return _Response(200, content=search_id.encode())

    monkeypatch.setattr(imagery.requests, "post", post)
    monkeypatch.setattr(imagery.requests, "get", get)
    yield state
    imagery.pc_tile.cache_clear()


def test_registers_once_while_valid(fake_pc):
    assert imagery.pc_tile("q1", 2019, 15, 1, 1)[0].endswith(b".h1")
    assert imagery.pc_tile("q1", 2019, 15, 1, 2)[0].endswith(b".h1")
    assert fake_pc["n"] == 1


def test_renews_before_stated_expiry(fake_pc):
    fake_pc["expiry"] = time.time() + 60  # inside the renewal margin
    imagery.pc_tile("q1", 2019, 15, 1, 1)
    fake_pc["expiry"] = time.time() + 86400
    tile = imagery.pc_tile("q1", 2019, 15, 1, 2)
    assert fake_pc["n"] == 2
    assert tile[0].endswith(b".h2")


def test_recovers_from_410(fake_pc):
    imagery.pc_tile("q1", 2019, 15, 1, 1)
    fake_pc["expired"].add(f"pc1.{int(fake_pc['expiry'])}.h1")
    tile = imagery.pc_tile("q1", 2019, 15, 1, 3)
    assert tile[0].endswith(b".h2")


def test_outside_and_unknown(fake_pc):
    assert imagery.pc_tile("q1", 2019, 0, 0, 0) is None
    with pytest.raises(KeyError):
        imagery.pc_tile("q1", 2011, 15, 1, 1)


def test_sources_point_at_the_proxy(fake_pc, tmp_path):
    record = {
        "bbox_4326": [-71.0, 44.0, -70.9, 44.1],
        "years": {"2019": {"source": "pc"}, "2025": {"source": "rest"}},
    }
    sources = imagery.naip_sources(
        "q1", [2019, 2025], record, imagery.default_pc_cache()
    )
    assert sources[2019].url == "/tiles/pc/q1/2019/{z}/{x}/{y}"
    assert sources[2019].kind == "pc"
    assert sources[2025].kind == "usda"
