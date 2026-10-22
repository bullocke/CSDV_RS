"""Unit tests for csdv_core.zonal.crowns.clip_to_cell.

Adjacent NAIP-CHM tiles overlap by a few hundred metres. Clipping each tile's
crowns to its own nominal cell must keep every crown exactly once.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from shapely.geometry import Point

from csdv_core.zonal.crowns import clip_to_cell

UTM18 = "EPSG:26918"


def _crowns_at(lon: np.ndarray, lat: np.ndarray) -> gpd.GeoDataFrame:
    to_utm = Transformer.from_crs("EPSG:4269", UTM18, always_xy=True)
    x, y = to_utm.transform(lon, lat)
    return gpd.GeoDataFrame(
        {"seed_x": x, "seed_y": y},
        geometry=[Point(a, b).buffer(2.0) for a, b in zip(x, y, strict=True)],
        crs=UTM18,
    )


def test_adjacent_cells_keep_each_crown_once() -> None:
    rng = np.random.default_rng(1)
    # Tree tops scattered over two side-by-side quarter quads and their
    # overlap margins.
    lon = rng.uniform(-76.01, -75.865, 2000)
    lat = rng.uniform(38.94, 38.99, 2000)
    crowns = _crowns_at(lon, lat)
    west_cell = (-76.0, 38.9375, -75.9375, 39.0)
    east_cell = (-75.9375, 38.9375, -75.875, 39.0)
    a = clip_to_cell(crowns, west_cell)
    b = clip_to_cell(crowns, east_cell)
    assert set(a.index).isdisjoint(b.index)
    inside_either = (lon >= -76.0) & (lon < -75.875)
    assert len(a) + len(b) == int(inside_either.sum())


def test_tree_top_on_shared_edge_belongs_to_one_cell() -> None:
    crowns = _crowns_at(np.array([-75.9375]), np.array([38.95]))
    to_ll = Transformer.from_crs(UTM18, "EPSG:4269", always_xy=True)
    lon, _ = to_ll.transform(crowns.seed_x.iloc[0], crowns.seed_y.iloc[0])
    west = clip_to_cell(crowns, (-76.0, 38.9375, -75.9375, 39.0))
    east = clip_to_cell(crowns, (-75.9375, 38.9375, -75.875, 39.0))
    # Round-trip error decides the side. What matters is exactly one.
    assert len(west) + len(east) == 1, lon


def test_empty_frame_passes_through() -> None:
    empty = gpd.GeoDataFrame({"seed_x": [], "seed_y": []}, geometry=[], crs=UTM18)
    assert clip_to_cell(empty, (-76.0, 38.0, -75.0, 39.0)).empty
