"""Unit tests for csdv_core.national.inventory.

The quarter-quad arithmetic matters most. A crown is assigned to the tile whose
nominal cell holds its tree top, so a wrong cell either drops crowns or counts
them twice along every tile edge in CONUS.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from csdv_core.national.inventory import (
    QUARTER_QUAD_DEG,
    build_inventory,
    drop_duplicate_quarter_quads,
    quarter_quad_bounds,
    source_resolution_m,
)


def test_quarter_quad_bounds_first_and_last_cells() -> None:
    # Block 38N 75W spans 38-39N and 75-76W. Quad 01 is its north-west
    # corner and quad 64 its south-east corner.
    b = quarter_quad_bounds(pd.Series(["3807501_nw", "3807564_se"]))
    nw = b.iloc[0]
    assert nw.cell_north == pytest.approx(39.0)
    assert nw.cell_west == pytest.approx(-76.0)
    se = b.iloc[1]
    assert se.cell_south == pytest.approx(38.0)
    assert se.cell_east == pytest.approx(-75.0)


def test_quarter_quad_bounds_matches_index_footprint_example() -> None:
    # m_2408002_ne: footprint from the NTSG index spans lon -80.813 to -80.749
    # and lat 24.937 to 25.001, a 3.75' cell plus the raster margin.
    b = quarter_quad_bounds(pd.Series(["2408002_ne"])).iloc[0]
    assert (b.cell_west, b.cell_east) == pytest.approx((-80.8125, -80.75))
    assert (b.cell_south, b.cell_north) == pytest.approx((24.9375, 25.0))


def test_quarter_quads_of_one_quad_tile_it_without_gaps() -> None:
    names = pd.Series([f"4507130_{q}" for q in ("nw", "ne", "sw", "se")])
    b = quarter_quad_bounds(names)
    assert np.allclose(b.cell_east - b.cell_west, QUARTER_QUAD_DEG)
    assert np.allclose(b.cell_north - b.cell_south, QUARTER_QUAD_DEG)
    # nw shares its east edge with ne and its south edge with sw.
    assert b.iloc[0].cell_east == pytest.approx(b.iloc[1].cell_west)
    assert b.iloc[0].cell_south == pytest.approx(b.iloc[2].cell_north)


def test_source_resolution_tokens() -> None:
    names = pd.Series(
        [
            "m_2408002_ne_17_030_20230111_20230530",
            "m_4107912_ne_17_060_20220619",
            "m_3008618_se_16_1_20171110",
            "m_3008618_se_16_h_20171110",
        ]
    )
    out = source_resolution_m(names).tolist()
    assert out[:3] == pytest.approx([0.3, 0.6, 1.0])
    assert np.isnan(out[3])


def test_drop_duplicates_keeps_latest_acquisition() -> None:
    frame = pd.DataFrame(
        {
            "quarter_quad": ["2408124_sw", "2408124_sw", "2408225_ne"],
            "acquisition_date": pd.to_datetime(
                ["2023-01-10", "2021-12-07", "2023-01-16"]
            ),
            "tile_id": ["new", "old", "only"],
        }
    )
    kept, n = drop_duplicate_quarter_quads(frame)
    assert n == 1
    assert sorted(kept["tile_id"]) == ["new", "only"]


def _square(w: float, s: float, e: float, n: float) -> str:
    ring = [[w, s], [e, s], [e, n], [w, n], [w, s]]
    return json.dumps({"type": "Polygon", "coordinates": [ring]})


def test_build_inventory_from_small_index(tmp_path: Path) -> None:
    rows = [
        ("m_3807501_nw_18_060_20220601_chm", "2022-06-01", "3807501", "nw", 2022),
        ("m_3807501_nw_18_060_20200601_chm", "2020-06-01", "3807501", "nw", 2020),
        ("m_3807501_ne_18_060_20220601_chm", "2022-06-01", "3807501", "ne", 2022),
    ]
    frame = pd.DataFrame(
        {
            "system:index": [r[0] for r in rows],
            "acquisition_date": [r[1] for r in rows],
            "chm_url": [f"http://example.org/{r[0]}.tif" for r in rows],
            "quad_id": [int(r[2]) for r in rows],
            "quarter": [r[3] for r in rows],
            "scale_factor": 100.0,
            "source_doqq": [r[0].removesuffix("_chm") for r in rows],
            "utm_zone": 18.0,
            "year": [float(r[4]) for r in rows],
            ".geo": [
                _square(-76.003, 38.935, -75.935, 39.003),
                _square(-76.003, 38.935, -75.935, 39.003),
                _square(-75.94, 38.935, -75.872, 39.003),
            ],
        }
    )
    index = tmp_path / "index.csv"
    frame.to_csv(index, index=False)
    tiles, summary = build_inventory(index)
    assert summary.n_index_rows == 3
    assert summary.n_tiles == 2
    assert summary.n_duplicates_dropped == 1
    assert set(tiles["year"]) == {2022}
    assert tiles["scale_factor"].tolist() == pytest.approx([0.01, 0.01])
    assert all(u.startswith("https://") for u in tiles["chm_url"])
    nw = tiles.set_index("quarter_quad").loc["3807501_nw"]
    assert nw.cell_west == pytest.approx(-76.0)
    assert nw.cell_north == pytest.approx(39.0)
    assert nw.footprint_km2 > 0
