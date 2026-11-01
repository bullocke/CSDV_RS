"""Tests for the review scores and the review tile server."""

from __future__ import annotations

import math

import pytest
from pyproj import Transformer

from csdv_core.review.sample import draw_sample
from csdv_core.review.store import Interpretation, ReviewStore
from csdv_core.review.summary import summarize, timing_of

REVIEW_TAG = "default"


@pytest.mark.parametrize(
    ("detected", "reference", "expected"),
    [
        ("2015-2018", "2015-2018", "match"),
        ("2018-2021", "2015-2018", "detected_late"),
        ("2015-2018", "2018-2021", "detected_early"),
        ("2015-2018", "unsure", "unsure"),
        ("2015-2018", "", "unsure"),
    ],
)
def test_timing_of(detected, reference, expected):
    assert timing_of(detected, reference) == expected


@pytest.fixture()
def labelled(review_exports, tmp_path):
    sample = draw_sample(review_exports, REVIEW_TAG, 4)
    store = ReviewStore.create(tmp_path / "review_default.gpkg", sample)
    big = sample[sample["unit"] == "q4407159"]["sample_id"].tolist()
    small = sample[sample["unit"] == "q3912112"]["sample_id"].tolist()
    # Big unit (incl_prob 4/12): 3 true, 1 rejected. Small unit (4/5): 1 true,
    # 2 rejected, 1 uncertain.
    for sid, verdict in zip(
        big, ["confirmed", "likely", "confirmed", "rejected"], strict=True
    ):
        interval = sample.loc[sample["sample_id"] == sid, "interval"].iloc[0]
        store.save(
            Interpretation(sid, verdict, ref_interval=interval, outline="about_right")
        )
    for sid, verdict in zip(
        small, ["confirmed", "rejected", "rejected", "uncertain"], strict=True
    ):
        store.save(Interpretation(sid, verdict, ref_interval="unsure"))
    return store


def test_per_unit_and_pooled_precision(labelled):
    table = summarize(labelled).table.set_index("unit")
    assert table.loc["q4407159", "precision"] == pytest.approx(0.75)
    assert table.loc["q3912112", "precision"] == pytest.approx(1 / 3)
    assert table.loc["q3912112", "n_uncertain"] == 1
    # Weights 3 (big unit) and 1.25 (small unit) per object.
    pooled = (3 * 3 + 1.25 * 1) / (4 * 3 + 3 * 1.25)
    assert table.loc["pooled", "precision"] == pytest.approx(pooled)
    assert table.loc["site_mean", "precision"] == pytest.approx((0.75 + 1 / 3) / 2)
    assert table.loc["q4407159", "timing_match"] == pytest.approx(1.0)
    assert table.loc["q4407159", "outline_ok"] == pytest.approx(1.0)


def test_redrawn_outline_scores(labelled):
    sid = labelled.labels["sample_id"].iloc[0]
    row = labelled.sample[labelled.sample["sample_id"] == sid]
    half = row.to_crs(4326).geometry.iloc[0]
    minx, miny, maxx, maxy = half.bounds
    from shapely.geometry import box

    left_half = half.intersection(box(minx, miny, (minx + maxx) / 2, maxy))
    labelled.save(Interpretation(sid, "confirmed"), geometry=left_half)
    df = summarize(labelled).labelled.set_index("sample_id")
    assert df.loc[sid, "iou"] == pytest.approx(0.5, abs=0.02)
    assert df.loc[sid, "area_ratio"] == pytest.approx(2.0, abs=0.08)


def test_summary_files(labelled, tmp_path):
    written = summarize(labelled).write(tmp_path, "review_default")
    assert all(p.exists() for p in written)
    assert "| pooled |" in written[1].read_text()


def test_tile_routes(review_exports):
    pytest.importorskip("httpx")
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from csdv_core.review.config import ReviewConfig
    from csdv_core.review.server import tile_routes

    cfg = ReviewConfig(review_exports, REVIEW_TAG, review_exports / "x.gpkg", "me")
    client = TestClient(Starlette(routes=tile_routes(cfg)))
    # A point inside the fixture rasters (UTM 19N, 300 km E, 4500 km N).
    lon, lat = Transformer.from_crs(26919, 4326, always_xy=True).transform(
        300100.0, 4499900.0
    )
    z = 15
    n = 2**z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    unit = "q4407159"
    ok = client.get(f"/tiles/ql/{unit}/chm_2018_3m/{z}/{x}/{y}.png")
    assert ok.status_code == 200 and ok.headers["content-type"] == "image/png"
    change = client.get(f"/tiles/dchm/{unit}/2015/2021/{z}/{x}/{y}.png")
    assert change.status_code == 200
    onset = client.get(f"/tiles/run/{unit}/onset/{z}/{x}/{y}.png")
    assert onset.status_code == 200
    assert client.get(f"/tiles/ql/{unit}/chm_2018_3m/{z}/0/0.png").status_code == 204
    assert (
        client.get(f"/tiles/ql/{unit}/chm_1999_3m/{z}/{x}/{y}.png").status_code == 404
    )
    assert client.get(f"/tiles/run/{unit}/other/{z}/{x}/{y}.png").status_code == 404
