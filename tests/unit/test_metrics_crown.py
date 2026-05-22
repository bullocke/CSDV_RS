"""Tests for csdv_core.metrics.crown."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import Point

from csdv_core.io.grids import GridSpec
from csdv_core.metrics.crown import crown_count, crown_cv, crown_p90, crown_stats


@pytest.fixture()
def crown_gdf() -> tuple[gpd.GeoDataFrame, GridSpec, tuple[float, float, float, float]]:
    """A grid 100x100 m at 0.5 m pixels (200x200), windows of 50 m -> 2x2 output.
    Cell (0,0) (upper-left, x in [0,50], y in [50,100]): 4 crowns, diameters [4,6,8,10].
    Cell (0,1) (x in [50,100], y in [50,100]): 2 crowns -> NaN for non-count stats.
    Other cells: no crowns.
    """
    diam_left = [4.0, 6.0, 8.0, 10.0]
    diam_right = [5.0, 7.0]
    crowns = []
    for d in diam_left:
        crowns.append({"crown_diam_m": d, "geometry": Point(10.0, 80.0)})
    for d in diam_right:
        crowns.append({"crown_diam_m": d, "geometry": Point(75.0, 80.0)})
    gdf = gpd.GeoDataFrame(crowns, crs="EPSG:5070")
    transform = from_origin(0.0, 100.0, 0.5, 0.5)
    grid = GridSpec(transform=transform, crs="EPSG:5070", pixel_size_m=0.5)
    bounds = (0.0, 0.0, 100.0, 100.0)
    return gdf, grid, bounds


def test_count_includes_empty_cells(crown_gdf):
    gdf, grid, bounds = crown_gdf
    res = crown_count(gdf, grid, bounds, window_m=50.0)
    assert res.array.shape == (2, 2)
    assert res.array[0, 0] == 4
    assert res.array[0, 1] == 2
    assert res.array[1, 0] == 0
    assert res.array[1, 1] == 0


def test_cv_matches_hand_value(crown_gdf):
    gdf, grid, bounds = crown_gdf
    res = crown_cv(gdf, grid, bounds, window_m=50.0)
    # diameters [4,6,8,10]: mean=7, std=sqrt(((9+1+1+9)/4))=sqrt(5)
    expected = float(np.sqrt(5.0) / 7.0)
    assert res.array[0, 0] == pytest.approx(expected, rel=1e-4)
    # Right cell has only 2 crowns -> NaN (below min_crowns=3).
    assert np.isnan(res.array[0, 1])


def test_p90_matches_numpy(crown_gdf):
    gdf, grid, bounds = crown_gdf
    res = crown_p90(gdf, grid, bounds, window_m=50.0)
    expected = float(np.percentile([4.0, 6.0, 8.0, 10.0], 90))
    assert res.array[0, 0] == pytest.approx(expected)


def test_invalid_stat_raises(crown_gdf):
    gdf, grid, bounds = crown_gdf
    with pytest.raises(ValueError):
        crown_stats(gdf, grid, bounds, stat="bogus")  # type: ignore[arg-type]


def test_missing_column_raises(crown_gdf):
    gdf, grid, bounds = crown_gdf
    gdf2 = gdf.drop(columns=["crown_diam_m"])
    with pytest.raises(ValueError):
        crown_stats(gdf2, grid, bounds, stat="cv")


def test_empty_gdf_returns_nan(crown_gdf):
    _, grid, bounds = crown_gdf
    empty = gpd.GeoDataFrame({"crown_diam_m": [], "geometry": []}, crs="EPSG:5070")
    res = crown_cv(empty, grid, bounds, window_m=50.0)
    assert np.all(np.isnan(res.array))
