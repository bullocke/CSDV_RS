"""Tests for the review app components, rendered without a browser."""

from __future__ import annotations

import pytest

solara = pytest.importorskip("solara")
ipyleaflet = pytest.importorskip("ipyleaflet")

import ipyvuetify as v  # noqa: E402

from csdv_core.review import app  # noqa: E402
from csdv_core.review.mapview import ReviewMap  # noqa: E402
from csdv_core.review.sample import draw_sample, list_runs  # noqa: E402
from csdv_core.review.store import ReviewStore  # noqa: E402

REVIEW_TAG = "default"


@pytest.fixture()
def setup(review_exports, tmp_path, monkeypatch):
    store = ReviewStore.create(
        tmp_path / "review_default.gpkg", draw_sample(review_exports, REVIEW_TAG, 2)
    )
    runs = {r.unit: r for r in list_runs(review_exports, REVIEW_TAG)}
    monkeypatch.setattr(app, "_store", lambda: store)
    monkeypatch.setattr(app, "_runs", lambda: runs)
    return store


def _buttons(rc, label):
    return [
        b
        for b in rc.find(v.Btn).widgets
        if any(isinstance(c, str) and c.startswith(label) for c in b.children)
    ]


def test_form_saves_verdict_and_drawing(setup):
    store = setup
    focus = app.load_focus(store, "R001")
    ctrl = ReviewMap()
    saved: list[bool] = []
    _, rc = solara.render(
        app.FormPanel(
            sid="R001",
            focus=focus,
            ctrl=ctrl,
            on_saved=saved.append,
            on_key=lambda k: None,
        ),
        handle_error=False,
    )
    # Saving without a verdict is refused.
    _buttons(rc, "Save (s)")[0].click()
    assert store.get("R001") is None and not saved
    # The first toggle group is the verdict.
    rc.find(v.BtnToggle).widgets[0].v_model = "rejected"
    ctrl.load_drawing(focus.geom_4326.buffer(-1e-5))
    _buttons(rc, "Save + next")[0].click()
    rec = store.get("R001")
    assert rec.verdict == "rejected"
    assert rec.ref_interval == ""
    assert rec.redrawn
    assert saved == [True]


def test_focus_context_and_overlay_urls(setup):
    focus = app.load_focus(setup, "R001")
    assert focus.focus_utm.crs == focus.outlines.crs
    assert int(focus.row["object_id"]) in set(focus.outlines["object_id"])
    unit = focus.run.unit
    assert app.overlay_url("CHM change", unit, 2018, 2015).endswith(
        f"/tiles/dchm/{unit}/2015/2018/{{z}}/{{x}}/{{y}}.png"
    )
    assert app.overlay_url("CHM change", unit, 2015, None) is None
    assert app.overlay_url("none", unit, 2015, None) is None
