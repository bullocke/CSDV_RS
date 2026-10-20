"""Tests for csdv_core.disturbance.sampling."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from csdv_core.disturbance.sampling import allocate_by_stratum, sample_objects


def test_allocation_takes_everything_when_small() -> None:
    """A sample larger than the population takes every unit."""
    counts = {"a": 2, "b": 3}
    allocation = allocate_by_stratum(counts, 10, 2)
    assert allocation == counts


def test_allocation_floor_and_proportion() -> None:
    """Every stratum gets its floor and large strata get the remainder."""
    counts = {"2012-2014": 5, "2014-2016": 9, "2016-2018": 16, "2018-2021": 2}
    allocation = allocate_by_stratum(counts, 12, 2)
    assert sum(allocation.values()) == 12
    assert all(value >= 2 for value in allocation.values())
    assert allocation["2016-2018"] >= allocation["2014-2016"]
    assert allocation["2018-2021"] == 2


def test_allocation_floor_capped_by_count() -> None:
    """A stratum with one unit gets one unit even with a floor of two."""
    allocation = allocate_by_stratum({"a": 1, "b": 20}, 6, 2)
    assert allocation == {"a": 1, "b": 5}


def test_allocation_floors_exceed_total() -> None:
    """When floors exceed the total, the largest strata are served first."""
    allocation = allocate_by_stratum({"a": 10, "b": 5, "c": 3}, 2, 2)
    assert sum(allocation.values()) == 2
    assert allocation["a"] >= 1


def test_sample_objects_is_reproducible() -> None:
    """The same seed draws the same objects, stratified by interval."""
    intervals = ["2012-2014"] * 6 + ["2014-2016"] * 6
    objects = gpd.GeoDataFrame(
        {
            "object_id": np.arange(1, 13),
            "onset_year": [2014] * 6 + [2016] * 6,
            "interval": intervals,
            "geometry": [box(index, 0, index + 1, 1) for index in range(12)],
        },
        crs="EPSG:32617",
    )
    first_rng = np.random.default_rng(5)
    second_rng = np.random.default_rng(5)
    first = sample_objects(objects, 4, 2, first_rng)
    second = sample_objects(objects, 4, 2, second_rng)
    assert list(first["object_id"]) == list(second["object_id"])
    interval_counts = first["interval"].value_counts().to_dict()
    assert interval_counts == {"2012-2014": 2, "2014-2016": 2}


def test_cells_in_polygon_falls_back_to_touched_cells() -> None:
    """Cell centres select cells, and a tiny polygon falls back to touched cells."""
    from affine import Affine
    from rasterio.crs import CRS

    from csdv_core.disturbance._chip_panels import cells_in_polygon, context_bounds
    from csdv_core.disturbance.inputs import CellGrid

    grid = CellGrid(
        crs=CRS.from_epsg(32617),
        pixel_transform=Affine(0.6, 0.0, 0.0, 0.0, -0.6, 96.0),
        pixel_m=0.6,
        block_px=32,
        cell_rows=5,
        cell_cols=5,
    )
    two_cells = box(0.0, 76.8, 38.4, 96.0)
    assert cells_in_polygon(two_cells, grid).sum() == 2
    tiny = box(1.0, 90.0, 2.0, 91.0)
    assert cells_in_polygon(tiny, grid).sum() == 1
    view = context_bounds((0.0, 0.0, 100.0, 100.0), 4.0, 600.0)
    assert view == (-250.0, -250.0, 350.0, 350.0)
