"""Outline choice for accepted candidates, and the view rule."""

from __future__ import annotations

import pytest
from shapely.geometry import box

from csdv_core.config import load_label
from csdv_core.label import routing as R
from csdv_core.label.outlines import (
    choose_outline,
    is_crisp,
    needs_polygon,
    parse_polygon,
    polygon_usable,
)
from csdv_core.label.routing import Routed
from csdv_core.label.views import View, hybrid_view

CFG = load_label()
CAND = box(0, 0, 100, 100)


def routed(
    fits=("about_right", "about_right"),
    shares=("over_75", "over_75"),
    change_type="clearcut",
    outcome=R.LOSS_FULL,
) -> Routed:
    return Routed(
        status="accepted",
        outcomes=[R.INTACT, outcome],
        first_loss=1,
        change_type=change_type,
        outline_fit=fits,
        affected_share=shares,
    )


def test_needs_polygon() -> None:
    assert not needs_polygon(routed())
    assert needs_polygon(routed(fits=("about_right", "too_small")))
    assert needs_polygon(routed(fits=("about_right", None)))
    assert needs_polygon(routed(shares=("under_25", "over_75")))


def test_crisp_types() -> None:
    assert is_crisp(routed(change_type="mining"), CFG.outlines)
    assert not is_crisp(routed(change_type="wind"), CFG.outlines)
    assert not is_crisp(routed(change_type=None), CFG.outlines)
    assert is_crisp(routed(change_type="fire"), CFG.outlines)
    assert not is_crisp(
        routed(change_type="fire", outcome=R.LOSS_PARTIAL), CFG.outlines
    )


def test_polygon_usable() -> None:
    cfg = CFG.outlines
    assert polygon_usable(box(-20, -20, 120, 120), CAND, cfg)
    assert not polygon_usable(None, CAND, cfg)
    assert not polygon_usable(box(60, 0, 200, 100), CAND, cfg)  # covers 40%
    assert not polygon_usable(box(-200, -200, 300, 300), CAND, cfg)  # 25 times
    assert not polygon_usable(box(0, 0, 20, 20), CAND, cfg)  # under 0.1 ha


def test_about_right_keeps_the_candidate() -> None:
    event = choose_outline(
        CAND,
        source="di",
        geom_src="refined",
        routed=routed(),
        polygon=None,
        cfg=CFG.outlines,
    )
    assert event.core.equals(CAND) and event.extension.is_empty
    assert event.quality == "precise" and event.source == "di_refined"
    cells = choose_outline(
        CAND,
        source="di",
        geom_src="cells",
        routed=routed(),
        polygon=None,
        cfg=CFG.outlines,
    )
    assert cells.quality == "envelope"
    diffuse = choose_outline(
        CAND,
        source="di",
        geom_src="refined",
        routed=routed(change_type="wind"),
        polygon=None,
        cfg=CFG.outlines,
    )
    assert diffuse.quality == "envelope" and not diffuse.crisp


def test_too_small_adds_an_extension() -> None:
    poly = box(-50, 0, 150, 100)
    event = choose_outline(
        CAND,
        source="di",
        geom_src="cells",
        routed=routed(fits=("too_small", "about_right")),
        polygon=poly,
        cfg=CFG.outlines,
    )
    assert event.core.equals(CAND) and event.trimmed.is_empty
    assert event.extension.area == pytest.approx(100 * 100)
    assert event.polygon_used and event.quality == "envelope"


def test_wrong_shape_trims_the_candidate() -> None:
    poly = box(-50, 0, 60, 100)
    event = choose_outline(
        CAND,
        source="chm",
        geom_src="cells",
        routed=routed(fits=("wrong_shape", "about_right")),
        polygon=poly,
        cfg=CFG.outlines,
    )
    assert event.core.area == pytest.approx(60 * 100)
    assert event.trimmed.area == pytest.approx(40 * 100)
    assert event.extension.area == pytest.approx(50 * 100)
    assert event.source == "chm_cells+polygon"


def test_without_a_usable_polygon() -> None:
    doubt = choose_outline(
        CAND,
        source="di",
        geom_src="cells",
        routed=routed(fits=("too_large", "about_right")),
        polygon=None,
        cfg=CFG.outlines,
    )
    assert doubt.core.is_empty and doubt.trimmed.equals(CAND)
    small = choose_outline(
        CAND,
        source="di",
        geom_src="cells",
        routed=routed(fits=("too_small", "too_small")),
        polygon=box(0, 0, 5, 5),
        cfg=CFG.outlines,
    )
    assert small.core.equals(CAND) and small.wide_margin and not small.polygon_used


def test_parse_polygon_maps_image_units_to_the_ground() -> None:
    view = View((1000.0, 2000.0, 1384.0, 2384.0), 640)
    answer = {"polygon": [[0, 0], [500, 0], [500, 500], [0, 500]]}
    poly = parse_polygon(answer, view)
    assert poly.bounds == pytest.approx((1000.0, 2192.0, 1192.0, 2384.0))
    assert parse_polygon({"polygon": [[0, 0], [1, 1]]}, view) is None
    assert parse_polygon("no", view) is None
    assert view.to_px(1192.0, 2192.0) == pytest.approx((320.0, 320.0))


STACK = (0.0, 0.0, 998.4, 998.4)


def test_view_zooms_in_on_a_small_candidate() -> None:
    view = hybrid_view(box(500, 500, 530, 530), STACK, CFG.views, key="a")
    assert view.edge_m == pytest.approx(150.0) and view.px == 640
    x0, y0, x1, y1 = view.bounds
    assert x0 <= 500 and x1 >= 530 and y0 <= 500 and y1 >= 530


def test_view_standard_and_enlarged() -> None:
    view = hybrid_view(box(400, 400, 600, 550), STACK, CFG.views, key="b")
    assert view.edge_m == pytest.approx(384.0) and view.px == 640
    large = hybrid_view(box(200, 300, 700, 500), STACK, CFG.views, key="c")
    assert large.edge_m == pytest.approx(625.2, abs=0.6)
    assert large.px == round(large.edge_m / 0.6)
    assert large.m_per_px == pytest.approx(0.6, abs=0.01)


def test_view_never_leaves_the_stack() -> None:
    edge = hybrid_view(box(0, 0, 60, 60), STACK, CFG.views, key="d")
    assert edge.bounds[0] >= 0 and edge.bounds[1] >= 0
    huge = hybrid_view(box(-500, -500, 2000, 2000), STACK, CFG.views, key="e")
    assert huge.bounds == pytest.approx(STACK)
    assert huge.px == CFG.views.max_px


def test_view_offset_is_fixed_by_the_key() -> None:
    geom = box(500, 500, 530, 530)
    first = hybrid_view(geom, STACK, CFG.views, key="x")
    assert first == hybrid_view(geom, STACK, CFG.views, key="x")
    assert first != hybrid_view(geom, STACK, CFG.views, key="y")
