"""Blocks of grid cells inside a polygon."""

from __future__ import annotations

from pathlib import Path

from shapely.geometry import box

from csdv_core.grid.blocks import block_units, write_batches
from csdv_core.grid.units import read_batch

ROOT = Path(__file__).resolve().parents[2]


def test_block_reproduces_southern_indiana() -> None:
    names = ("indiana_block", "indiana_ring", "indiana_south")
    want = {
        u.unit_id for n in names for u in read_batch(ROOT / f"config/grid/{n}.csv")
    } | {"q3908654"}
    units = block_units((-87.0, 38.25, -86.0, 39.5))
    assert {u.unit_id for u in units} == want
    assert units[0].unit_id == "q3908633"  # north-west corner first


def test_polygon_margin_and_skip() -> None:
    state = box(-87.0, 38.0, -86.0, 39.0)
    inside = block_units((-88.0, 37.0, -85.0, 40.0), state)
    assert len(inside) == 64
    kept = block_units((-88.0, 37.0, -85.0, 40.0), state, margin_km=2.0)
    assert len(kept) == 36  # the outer ring of cells touches the margin
    fewer = block_units(
        (-88.0, 37.0, -85.0, 40.0), state, skip={inside[0].unit_id}, prefix="IN_"
    )
    assert len(fewer) == 63 and fewer[0].label.startswith("IN_")


def test_parts_are_read_back(tmp_path: Path) -> None:
    units = block_units((-87.0, 38.0, -86.0, 39.0), prefix="IN_")
    paths = write_batches(units, tmp_path, "demo", part_size=40, note="A block\n\nx")
    assert [p.name for p in paths] == ["demo_1.csv", "demo_2.csv"]
    back = [u.unit_id for p in paths for u in read_batch(p)]
    assert back == [u.unit_id for u in units]
