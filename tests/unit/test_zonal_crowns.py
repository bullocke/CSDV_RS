"""Tests for crown assignment and diameter statistics inside a stand."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from csdv_core.zonal.crowns import (
    crown_diameter_stats,
    crowns_in_stand,
    segment_scene_crowns,
)


def _crowns(centres_and_diams: list[tuple[float, float, float]]) -> gpd.GeoDataFrame:
    """Build crown polygons as circles of a given diameter at given centres."""
    geoms = [Point(x, y).buffer(d / 2.0) for x, y, d in centres_and_diams]
    return gpd.GeoDataFrame(
        {
            "segment_id": range(len(geoms)),
            "crown_diam_m": [d for _, _, d in centres_and_diams],
        },
        geometry=geoms,
        crs="EPSG:26916",
    )


def test_each_crown_lands_in_exactly_one_stand():
    """Crowns straddling a shared boundary go to the stand holding the centre."""
    left = box(0.0, 0.0, 50.0, 50.0)
    right = box(50.0, 0.0, 100.0, 50.0)
    # Three crowns, all overlapping the boundary at x = 50.
    crowns = _crowns([(46.0, 25.0, 12.0), (50.5, 25.0, 12.0), (54.0, 25.0, 12.0)])
    in_left = crowns_in_stand(crowns, left)
    in_right = crowns_in_stand(crowns, right)
    assert len(in_left) == 1
    assert len(in_right) == 2
    assert len(in_left) + len(in_right) == len(crowns)


def test_crowns_outside_the_stand_are_excluded():
    stand = box(0.0, 0.0, 50.0, 50.0)
    crowns = _crowns([(25.0, 25.0, 6.0), (200.0, 200.0, 6.0)])
    assert len(crowns_in_stand(crowns, stand)) == 1


def test_crowns_in_stand_handles_an_empty_frame():
    empty = _crowns([]).iloc[0:0]
    assert crowns_in_stand(empty, box(0, 0, 10, 10)).empty


def test_diameter_statistics_are_exact():
    crowns = _crowns([(i * 20.0, 0.0, d) for i, d in enumerate([2.0, 4.0, 6.0, 8.0])])
    stats = crown_diameter_stats(crowns)
    assert stats.n_crowns == 4
    assert stats.mean == pytest.approx(5.0)
    assert stats.median == pytest.approx(5.0)
    assert stats.std == pytest.approx(np.std([2.0, 4.0, 6.0, 8.0]))
    assert stats.cv == pytest.approx(np.std([2.0, 4.0, 6.0, 8.0]) / 5.0)
    assert stats.p90 == pytest.approx(np.percentile([2.0, 4.0, 6.0, 8.0], 90))
    assert stats.reason == ""


def test_uniform_crowns_have_zero_variability():
    crowns = _crowns([(i * 20.0, 0.0, 5.0) for i in range(6)])
    assert crown_diameter_stats(crowns).cv == pytest.approx(0.0)


def test_below_the_minimum_crown_count_nothing_is_reported():
    crowns = _crowns([(0.0, 0.0, 4.0), (20.0, 0.0, 6.0)])
    stats = crown_diameter_stats(crowns, min_crowns=3)
    assert stats.n_crowns == 2
    assert np.isnan(stats.cv)
    assert np.isnan(stats.p90)
    assert stats.reason == "n_crowns=2 < min_crowns=3"


def test_empty_stand_reports_zero_crowns_with_a_reason():
    stats = crown_diameter_stats(_crowns([]).iloc[0:0])
    assert stats.n_crowns == 0
    assert "min_crowns" in stats.reason


def test_as_metrics_uses_registry_names():
    crowns = _crowns([(i * 20.0, 0.0, 5.0) for i in range(4)])
    metrics = crown_diameter_stats(crowns).as_metrics()
    assert set(metrics) >= {"crown_cv", "crown_p90", "crown_mean", "crown_count"}
    assert metrics["crown_count"] == pytest.approx(4.0)


def test_missing_diameter_column_raises():
    crowns = _crowns([(0.0, 0.0, 4.0)] * 4).drop(columns=["crown_diam_m"])
    with pytest.raises(KeyError, match="crown_diam_m"):
        crown_diameter_stats(crowns)


@pytest.fixture()
def cone_scene() -> tuple[np.ndarray, object]:
    """A 240 x 240 px CHM at 0.5 m with cone-shaped trees on a regular lattice."""
    size = 240
    chm = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.indices((size, size))
    for r in range(20, size, 40):
        for c in range(20, size, 40):
            dist = np.hypot(yy - r, xx - c)
            chm = np.maximum(chm, np.clip(18.0 - dist * 1.2, 0.0, None))
    return chm, from_origin(0.0, 120.0, 0.5, 0.5)


def test_scene_segmentation_finds_the_planted_lattice(cone_scene):
    chm, transform = cone_scene
    crowns = segment_scene_crowns(chm, transform, "EPSG:26916", block_px=240)
    # Six by six lattice of trees.
    assert 30 <= len(crowns) <= 42
    assert crowns["crown_diam_m"].min() > 0.0


def test_block_splitting_does_not_duplicate_or_lose_crowns(cone_scene):
    """Segmenting in blocks must give the same crown count as one pass, which
    is only true because the peak footprint is held to the scene mean."""
    chm, transform = cone_scene
    one_pass = segment_scene_crowns(chm, transform, "EPSG:26916", block_px=240)
    blocked = segment_scene_crowns(
        chm, transform, "EPSG:26916", block_px=80, overlap_px=32
    )
    assert len(blocked) == len(one_pass)


def test_scene_segmentation_returns_empty_below_the_height_threshold():
    chm = np.full((60, 60), 0.5, dtype=np.float32)
    crowns = segment_scene_crowns(chm, from_origin(0.0, 30.0, 0.5, 0.5), "EPSG:26916")
    assert crowns.empty
    assert "crown_diam_m" in crowns.columns
