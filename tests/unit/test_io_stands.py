"""Tests for reading and decoding AIS disturbance polygon deliveries."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon, box

from csdv_core.io.stands import (
    COVER_CLASS_MIDPOINT,
    cover_bounds,
    cover_midpoint,
    cover_series_frame,
    interpreter_cover_series,
    module_footprint,
    read_ais_stands,
    stands_by_id,
)

LAYER = "DisturbancePoly"


def _feature(**overrides: object) -> dict[str, object]:
    """One AIS record with protocol defaults, overridable per test."""
    record: dict[str, object] = {
        "DistType": 121,
        "DOQQ": "ElkinsvilleNE",
        "UID1": 1,
        "UID2": 0,
        "UID3": 0,
        "TreeType": 999,
        "BaseDate": 2018,
        "FollowUpDate": 2022,
        "LastImageryPreDist": 2016,
        "FirstImageryPostDist": 2017,
        "WithinMappingarea": 1,
        "PercentTreePreDist": 6,
        "PercentTreeFirstPost": 1,
        "PercentTreeBase": 2,
        "PercentWoodyRegenBase": 5,
        "PercentTreeFollowUp": 4,
        "PercentWoodyRegenFollowUp": 3,
        "AdditionalDisturbance": 0,
        "Acres": 1.0,
    }
    record.update(overrides)
    return record


@pytest.fixture()
def delivery(tmp_path: Path) -> Path:
    """A four-feature stand-in for an AIS delivery, written as a GeoPackage.

    Module is a 400 x 400 m square. Two harvest polygons, one wind polygon that
    shares a UID1 with the first harvest (a subdivided footprint), and the
    DistType=999 remainder with the disturbances punched out as holes.
    """
    module = box(0.0, 0.0, 400.0, 400.0)
    small = box(10.0, 10.0, 60.0, 110.0)  # 5,000 m2, bbox fill 1.0
    large = box(100.0, 100.0, 300.0, 300.0)  # 40,000 m2
    third = box(320.0, 20.0, 380.0, 80.0)  # 3,600 m2
    remainder = module.difference(small.union(large).union(third))

    records = [
        _feature(UID1=7, DistType=121),
        _feature(UID1=7, DistType=121, PercentTreeBase=0),
        _feature(UID1=9, DistType=131, TreeType=999),
        _feature(
            UID1=0,
            DistType=999,
            TreeType=999,
            LastImageryPreDist=9999,
            FirstImageryPostDist=9999,
            WithinMappingarea=9,
            PercentTreePreDist=9,
            PercentTreeFirstPost=9,
            PercentTreeBase=9,
            PercentWoodyRegenBase=9,
            PercentTreeFollowUp=9,
            PercentWoodyRegenFollowUp=9,
            AdditionalDisturbance=9,
        ),
    ]
    gdf = gpd.GeoDataFrame(
        records,
        geometry=[MultiPolygon([small]), MultiPolygon([large]), third, remainder],
        crs="EPSG:26916",
    )
    path = tmp_path / "delivery.gpkg"
    gdf.to_file(path, layer=LAYER, driver="GPKG")
    return path


def test_reads_and_decodes(delivery: Path):
    stands = read_ais_stands(delivery, layer=LAYER)
    assert len(stands) == 3  # the 999 remainder is dropped
    assert set(stands["dist_label"]) == {"Clearcut", "Wind"}
    assert set(stands["dist_group"]) == {"Harvest", "Weather"}
    assert stands["tree_type_label"].eq("Not applicable").all()
    assert stands.crs.to_epsg() == 26916


def test_keeps_non_disturbance_when_asked(delivery: Path):
    stands = read_ais_stands(delivery, layer=LAYER, drop_non_disturbance=False)
    assert len(stands) == 4
    assert "Non-disturbance area" in set(stands["dist_label"])


def test_stand_ids_are_unique_and_suffix_subdivided_footprints(delivery: Path):
    stands = read_ais_stands(delivery, layer=LAYER)
    assert stands["stand_id"].is_unique
    suffixed = sorted(s for s in stands["stand_id"] if s[-1].isalpha())
    assert suffixed == ["ELKNE-U7-0-0a", "ELKNE-U7-0-0b"]
    # 'a' is the larger of the two pieces.
    areas = stands.set_index("stand_id")["area_m2"]
    assert areas["ELKNE-U7-0-0a"] > areas["ELKNE-U7-0-0b"]
    # A footprint with a single impact polygon carries no suffix.
    assert "ELKNE-U9-0-0" in set(stands["stand_id"])


def test_geometry_measures(delivery: Path):
    stands = read_ais_stands(delivery, layer=LAYER).set_index("stand_id")
    assert stands.loc["ELKNE-U7-0-0a", "area_m2"] == pytest.approx(40_000.0)
    assert stands.loc["ELKNE-U7-0-0b", "area_m2"] == pytest.approx(5_000.0)
    # A rectangle exactly fills its own bounding box.
    assert stands["bbox_fill"].to_numpy() == pytest.approx(1.0)
    assert stands.loc["ELKNE-U9-0-0", "perimeter_m"] == pytest.approx(240.0)


def test_multipolygon_geometry_is_not_exploded(delivery: Path):
    stands = read_ais_stands(delivery, layer=LAYER)
    # An impact polygon is one stand even when delivered as a MultiPolygon.
    assert len(stands) == 3


def test_bbox_fill_reflects_a_non_rectangular_stand(tmp_path: Path):
    """An L-shaped stand fills only part of its bounding box."""
    ell = Polygon([(0, 0), (100, 0), (100, 50), (50, 50), (50, 100), (0, 100)])
    gdf = gpd.GeoDataFrame([_feature(UID1=1)], geometry=[ell], crs="EPSG:26916")
    path = tmp_path / "ell.gpkg"
    gdf.to_file(path, layer=LAYER, driver="GPKG")
    stands = read_ais_stands(path, layer=LAYER)
    assert stands.loc[0, "bbox_fill"] == pytest.approx(0.75)


def test_module_footprint_fills_holes(delivery: Path):
    footprint = module_footprint(delivery, layer=LAYER)
    assert footprint.geom_type == "Polygon"
    assert len(footprint.interiors) == 0
    assert footprint.area == pytest.approx(160_000.0)


def test_invalid_geometry_is_repaired(tmp_path: Path):
    bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
    assert not bowtie.is_valid
    gdf = gpd.GeoDataFrame([_feature()], geometry=[bowtie], crs="EPSG:26916")
    path = tmp_path / "bowtie.gpkg"
    gdf.to_file(path, layer=LAYER, driver="GPKG")
    stands = read_ais_stands(path, layer=LAYER)
    assert stands.geometry.is_valid.all()


def test_geographic_crs_is_rejected(tmp_path: Path):
    gdf = gpd.GeoDataFrame([_feature()], geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    path = tmp_path / "wgs84.gpkg"
    gdf.to_file(path, layer=LAYER, driver="GPKG")
    with pytest.raises(ValueError, match="projected CRS"):
        read_ais_stands(path, layer=LAYER)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (0, (0.00, 0.01)),
        (4, (0.20, 0.40)),
        (7, (0.80, 1.00)),
    ],
)
def test_cover_bounds(code: int, expected: tuple[float, float]):
    assert cover_bounds(code) == expected


def test_cover_class_bins_are_ordered_and_cover_zero_to_one():
    coded = [COVER_CLASS_MIDPOINT[c] for c in range(8)]
    assert coded == sorted(coded)
    assert cover_bounds(0)[0] == 0.0
    assert cover_bounds(7)[1] == 1.0


def test_not_assessed_cover_is_nan():
    assert np.isnan(cover_midpoint(9))
    assert np.isnan(cover_bounds(9)[0])
    assert np.isnan(cover_midpoint(None))


def test_interpreter_cover_series(delivery: Path):
    stands = read_ais_stands(delivery, layer=LAYER).set_index("stand_id")
    series = interpreter_cover_series(stands.loc["ELKNE-U9-0-0"])
    assert list(series["which"]) == [
        "pre-disturbance",
        "first post-disturbance",
        "base",
        "follow-up",
    ]
    assert list(series["year"]) == [2016.0, 2017.0, 2018.0, 2022.0]
    assert series.loc[0, "tree_mid"] == pytest.approx(0.70)  # class 6, 60-80%
    assert series.loc[1, "tree_mid"] == pytest.approx(0.03)  # class 1, 1-5%
    # Woody regeneration is only recorded on the base and follow-up images.
    assert np.isnan(series.loc[0, "regen_mid"])
    assert series.loc[2, "regen_mid"] == pytest.approx(0.50)


def test_cover_series_year_sentinel_becomes_nan(delivery: Path):
    stands = read_ais_stands(delivery, layer=LAYER, drop_non_disturbance=False)
    remainder = stands[stands["DistType"] == 999].iloc[0]
    series = interpreter_cover_series(remainder)
    assert np.isnan(series.loc[0, "year"])
    assert np.isnan(series.loc[0, "tree_mid"])


def test_cover_series_frame_stacks_every_stand(delivery: Path):
    stands = read_ais_stands(delivery, layer=LAYER)
    frame = cover_series_frame(stands)
    assert len(frame) == 4 * len(stands)
    assert set(frame["stand_id"]) == set(stands["stand_id"])


def test_stands_by_id_preserves_order(delivery: Path):
    stands = read_ais_stands(delivery, layer=LAYER)
    wanted = ["ELKNE-U9-0-0", "ELKNE-U7-0-0a"]
    assert list(stands_by_id(stands, wanted)["stand_id"]) == wanted


def test_stands_by_id_rejects_unknown(delivery: Path):
    stands = read_ais_stands(delivery, layer=LAYER)
    with pytest.raises(KeyError, match="ELKNE-U99-0-0"):
        stands_by_id(stands, ["ELKNE-U99-0-0"])
