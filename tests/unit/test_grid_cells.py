"""Tests for the CONUS processing grid and unit records.

No network calls. Cell IDs are checked against NAIP quarter-quad names and,
when naipcrowns is installed, against its quarter-quad bounds.
"""

from __future__ import annotations

import math

import pytest

from csdv_core.grid.cells import (
    MAX_BUFFER_KM,
    QUAD_DEG,
    point_bounds,
    quad_bounds,
    quad_id_at,
)
from csdv_core.grid.units import GridPaths, Unit, load_unit, read_batch, save_unit


@pytest.mark.parametrize(
    "lon, lat, quad",
    [
        # NAIP m_4407159_* quarter quads (White Mountains, NH).
        (-71.68, 44.06, "q4407159"),
        # m_4506921_* covers the NorthMaine site centre.
        (-69.4457, 45.7070, "q4506921"),
        # Hoosier NF, m_3908654_* (the ElkinsvilleNE quarter quad).
        (-86.3, 39.2, "q3908654"),
    ],
)
def test_quad_id_matches_naip_quad_codes(lon, lat, quad):
    assert quad_id_at(lon, lat) == quad


@pytest.mark.parametrize(
    "lon, lat", [(-71.68, 44.06), (-124.2, 48.9), (-80.1, 25.2), (-100.0001, 40.9999)]
)
def test_cell_contains_its_point_and_is_one_quad(lon, lat):
    west, south, east, north = quad_bounds(quad_id_at(lon, lat))
    assert west <= lon < east and south <= lat < north
    assert east - west == pytest.approx(QUAD_DEG)
    assert north - south == pytest.approx(QUAD_DEG)


def test_quad_bounds_accepts_ids_without_prefix():
    assert quad_bounds("4407159") == quad_bounds("q4407159")


@pytest.mark.parametrize("bad", ["q440715", "q4407165", "q4407100", "x4407159"])
def test_quad_bounds_rejects_bad_ids(bad):
    with pytest.raises(ValueError):
        quad_bounds(bad)


def test_quad_id_rejects_points_outside_the_quadrant():
    with pytest.raises(ValueError):
        quad_id_at(10.0, 45.0)


def test_quad_is_the_union_of_its_naipcrowns_quarter_quads():
    pd = pytest.importorskip("pandas")
    inventory = pytest.importorskip("naipcrowns.national.inventory")
    quarters = pd.Series([f"4407159_{q}" for q in ("nw", "ne", "sw", "se")])
    b = inventory.quarter_quad_bounds(quarters)
    union = (
        b["cell_west"].min(),
        b["cell_south"].min(),
        b["cell_east"].max(),
        b["cell_north"].max(),
    )
    assert union == pytest.approx(quad_bounds("q4407159"))


def test_point_bounds_has_the_requested_half_width():
    lon, lat = -71.6, 44.1
    west, south, east, north = point_bounds(lon, lat, 5.0)
    assert (north - south) * 111.32 == pytest.approx(10.0, rel=1e-4)
    width_km = (east - west) * 111.32 * math.cos(math.radians(lat))
    assert width_km == pytest.approx(10.0, rel=1e-4)


@pytest.mark.parametrize("buffer_km", [0.0, -1.0, MAX_BUFFER_KM + 0.1])
def test_point_bounds_rejects_bad_buffers(buffer_km):
    with pytest.raises(ValueError, match="buffer_km"):
        point_bounds(-71.6, 44.1, buffer_km)


def _paths(tmp_path) -> GridPaths:
    return GridPaths(
        grid_dir=tmp_path / "grid",
        export_dir=tmp_path / "exports",
        data_root=tmp_path / "data",
        results_root=tmp_path / "results",
    )


def test_read_batch_builds_cells_and_points(tmp_path):
    batch = tmp_path / "batch.csv"
    batch.write_text(
        "# comment line\n"
        "label,lon,lat,buffer_km,cell,notes\n"
        "Lincoln,-71.68,44.06,,,cell from a point\n"
        "Elk,,,,q3908654,named cell\n"
        "Maine,-69.44,45.70,5,,point unit\n",
        encoding="utf-8",
    )
    units = read_batch(batch)
    assert [u.unit_id for u in units] == ["q4407159", "q3908654", "p_Maine"]
    assert [u.kind for u in units] == ["cell", "cell", "point"]
    assert units[0].label == "Lincoln" and units[0].notes == "cell from a point"
    assert units[2].buffer_km == 5.0


def test_read_batch_rejects_duplicate_units(tmp_path):
    batch = tmp_path / "batch.csv"
    batch.write_text(
        "label,lon,lat,buffer_km,cell,notes\n"
        "a,-71.68,44.06,,,\n"
        "b,-71.70,44.07,,,\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="both q4407159"):
        read_batch(batch)


def test_point_unit_needs_a_clean_label():
    with pytest.raises(ValueError, match="label"):
        Unit.point(-71.6, 44.1, 3.0, label="has space")


def test_unit_record_round_trip_and_fixed_box(tmp_path):
    paths = _paths(tmp_path)
    unit = Unit.cell("q4407159", label="Lincoln")
    unit.years = {2021: {"source": "pc", "coverage": 1.0}}
    unit.skipped_years = {2023: "no imagery in either catalogue"}
    save_unit(paths, unit)
    loaded = load_unit(paths, "q4407159")
    assert loaded is not None
    assert loaded.years == unit.years
    assert loaded.skipped_years == unit.skipped_years
    assert loaded.bbox_4326 == pytest.approx(unit.bbox_4326)

    moved = Unit.point(-71.6, 44.1, 3.0, label="x")
    moved.unit_id = "q4407159"
    with pytest.raises(ValueError, match="already exists"):
        save_unit(paths, moved)
