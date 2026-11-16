"""Tests of csdv_core.label.fire: the agent of a loss inside a fire perimeter."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from csdv_core.config import load_label
from csdv_core.label.fire import FROM_PERIMETER, FirePerimeters, settle_fire_agent
from csdv_core.label.routing import Routed

DATES = ["2016-06-01", "2018-06-01", "2020-06-01"]
EPSG = 26910


def _perimeters(ig_date: str = "2019-08-01", share: float = 0.5) -> FirePerimeters:
    frame = gpd.GeoDataFrame(
        {"ig_date": [pd.Timestamp(ig_date)]},
        geometry=[box(0, 0, 100, 100)],
        crs=EPSG,
    ).to_crs(4269)
    return FirePerimeters(frame, min_share=share)


def _routed(types=("fire", "insect_disease"), agent=None, first=1) -> Routed:
    return Routed(
        status="accepted",
        outcomes=["intact", "loss_full"],
        first_loss=first,
        severity="stand_replacing",
        agent=agent,
        change_types=types,
    )


def test_disputed_agent_inside_a_fire_of_its_interval_becomes_fire():
    out = settle_fire_agent(_routed(), box(10, 10, 60, 60), EPSG, DATES, _perimeters())
    assert out.agent == "fire" and out.agent_source == FROM_PERIMETER
    # The type stays unset and the loss is untouched.
    assert out.change_type is None and out.outcomes == ["intact", "loss_full"]


@pytest.mark.parametrize(
    "routed, geom, ig_date",
    [
        (
            _routed(types=("clearcut", "insect_disease")),
            box(10, 10, 60, 60),
            "2019-08-01",
        ),
        (
            _routed(agent="wind", types=("wind", "wind")),
            box(10, 10, 60, 60),
            "2019-08-01",
        ),
        (_routed(), box(10, 10, 60, 60), "2017-08-01"),
        (_routed(), box(80, 80, 180, 180), "2019-08-01"),
        (_routed(first=None), box(10, 10, 60, 60), "2019-08-01"),
    ],
    ids=[
        "no-model-said-fire",
        "agreed-agent",
        "fire-in-other-interval",
        "mostly-outside",
        "no-loss",
    ],
)
def test_other_cases_are_left_alone(routed, geom, ig_date):
    out = settle_fire_agent(routed, geom, EPSG, DATES, _perimeters(ig_date))
    assert out is routed


def test_config_has_the_rule_off_by_default():
    routing = load_label().routing
    assert routing.fire_perimeters is None
    assert routing.fire_perimeter_share == 0.5


def test_perimeters_read_from_a_file(tmp_path):
    path = tmp_path / "perims.gpkg"
    gpd.GeoDataFrame(
        {"ig_date": ["2019-08-01"], "incid_name": ["TEST"]},
        geometry=[box(0, 0, 100, 100)],
        crs=EPSG,
    ).to_file(path)
    perims = FirePerimeters.read(path, 0.6)
    assert perims.min_share == 0.6
    assert perims.share_inside(box(50, 0, 150, 100), EPSG, DATES[1], DATES[2]) == (
        pytest.approx(0.5)
    )
