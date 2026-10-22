"""Unit tests for csdv_core.disturbance.outlines."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import Polygon, box

from csdv_core.config._models import DisturbanceConfig
from csdv_core.disturbance import outlines, windows
from csdv_core.disturbance.decide import onset_stack
from csdv_core.disturbance.inputs import CellGrid, SiteInputs, YearInput

#: UTM zone used by the synthetic grids.
TEST_CRS = CRS.from_epsg(32617)
#: Upper-left corner of the synthetic grids (UTM metres).
ORIGIN_X = 500_000.0
ORIGIN_Y = 4_000_000.0
#: CHM pixel size of the synthetic grids (m).
PIXEL_M = 0.6
#: Years of the synthetic stacks.
YEARS = (2013, 2015, 2017)


def _grid(cell_rows: int = 10, cell_cols: int = 10) -> CellGrid:
    """Build a 19.2 m cell grid over 0.6 m pixels."""
    pixel_transform = Affine(PIXEL_M, 0.0, ORIGIN_X, 0.0, -PIXEL_M, ORIGIN_Y)
    return CellGrid(
        crs=TEST_CRS,
        pixel_transform=pixel_transform,
        pixel_m=PIXEL_M,
        block_px=32,
        cell_rows=cell_rows,
        cell_cols=cell_cols,
    )


def _two_object_stack() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detections, probability and z with one object at t=1 and one at t=2."""
    onset_index = np.full((10, 10), -1, dtype="int8")
    onset_index[1:3, 1:4] = 1
    onset_index[6:9, 6:8] = 2
    probability = np.zeros((3, 10, 10), dtype="float32")
    probability[1, 1:3, 1:4] = 0.8
    probability[1, 1, 1] = 0.95
    probability[2, 6:9, 6:8] = 0.6
    z = np.zeros((3, 10, 10), dtype="float32")
    z[1, 1:3, 1:4] = -4.0
    z[2, 6:9, 6:8] = -3.0
    z[2, 6, 6] = np.nan
    return onset_stack(onset_index, len(YEARS)), probability, z


def test_detection_objects_two_years() -> None:
    """Objects at two dates get the right years, labels and sizes."""
    grid = _grid()
    detected, probability, z = _two_object_stack()
    objects = outlines.detection_objects(detected, probability, z, grid, YEARS)
    assert list(objects.columns) == list(outlines.OBJECT_COLUMNS)
    assert list(objects["object_id"]) == [1, 2]
    first = objects.iloc[0]
    second = objects.iloc[1]
    assert first["onset_year"] == 2015
    assert first["year_before"] == 2013
    assert first["interval"] == "2013-2015"
    assert first["n_cells"] == 6
    assert first["area_ha"] == pytest.approx(6 * grid.cell_ha)
    assert first["mean_prob"] == pytest.approx((5 * 0.8 + 0.95) / 6, rel=1e-5)
    assert first["max_prob"] == pytest.approx(0.95)
    assert first["mean_z"] == pytest.approx(-4.0)
    assert first.geometry.area / outlines.M2_PER_HA == pytest.approx(first["area_ha"])
    assert second["onset_year"] == 2017
    assert second["interval"] == "2015-2017"
    assert second["n_cells"] == 6
    assert second["mean_z"] == pytest.approx(-3.0)
    assert bool(first["follow_up"]) is True
    assert bool(second["follow_up"]) is False
    assert first["earlier_share"] == 0.0
    assert bool(first["growth"]) is False
    assert objects.crs == TEST_CRS


def test_detection_objects_overlap_across_intervals() -> None:
    """A cell detected twice belongs to one object per interval."""
    grid = _grid()
    detected = np.zeros((3, 10, 10), dtype=bool)
    detected[1, 2:5, 2:5] = True
    detected[2, 3:6, 3:6] = True
    z = np.zeros((3, 10, 10), dtype="float32")
    objects = outlines.detection_objects(detected, None, z, grid, YEARS, min_cells=11)
    assert list(objects["onset_index"]) == [1, 2]
    assert list(objects["n_cells"]) == [9, 9]
    second = objects.iloc[1]
    assert second["earlier_share"] == pytest.approx(4 / 9)
    assert bool(second["growth"]) is True
    assert objects.iloc[0].geometry.intersection(second.geometry).area == (
        pytest.approx(4 * grid.cell_m**2)
    )


def test_detection_objects_diagonal_merges() -> None:
    """Diagonally touching cells form one object."""
    grid = _grid()
    onset_index = np.full((10, 10), -1, dtype="int8")
    onset_index[2, 2] = 1
    onset_index[3, 3] = 1
    onset_index[4, 4] = 1
    z = np.zeros((3, 10, 10), dtype="float32")
    detected = onset_stack(onset_index, len(YEARS))
    objects = outlines.detection_objects(detected, None, z, grid, YEARS)
    assert len(objects) == 1
    assert objects.iloc[0]["n_cells"] == 3
    assert np.isnan(objects.iloc[0]["mean_prob"])


def test_detection_objects_empty() -> None:
    """No detection gives an empty frame with the object columns."""
    grid = _grid()
    detected = np.zeros((3, 10, 10), dtype=bool)
    z = np.zeros((3, 10, 10), dtype="float32")
    objects = outlines.detection_objects(detected, None, z, grid, YEARS)
    assert objects.empty
    assert list(objects.columns) == list(outlines.OBJECT_COLUMNS)


def test_objects_table_adds_centroids() -> None:
    """The flat table drops geometry and adds projected and lon/lat centroids."""
    grid = _grid()
    detected, probability, z = _two_object_stack()
    objects = outlines.detection_objects(detected, probability, z, grid, YEARS)
    table = outlines.objects_table(objects)
    assert "geometry" not in table.columns
    for column in outlines.CENTROID_COLUMNS:
        assert column in table.columns
    expected_x = ORIGIN_X + 2.5 * grid.cell_m
    assert table.loc[0, "centroid_x"] == pytest.approx(expected_x)
    assert -84.0 < table.loc[0, "centroid_lon"] < -80.0
    assert 35.0 < table.loc[0, "centroid_lat"] < 37.0


def _synthetic_chm_pair(
    size_px: int, block: slice
) -> tuple[np.ndarray, np.ndarray, Affine]:
    """Forest near 20 m at both dates, with a block cleared at the later date."""
    rng = np.random.default_rng(42)
    noise_before = rng.normal(0.0, 1.5, (size_px, size_px))
    noise_after = rng.normal(0.0, 1.5, (size_px, size_px))
    chm_before = (20.0 + noise_before).astype("float32")
    chm_after = (20.0 + noise_after).astype("float32")
    chm_after[block, block] = 0.3
    transform = Affine(PIXEL_M, 0.0, ORIGIN_X, 0.0, -PIXEL_M, ORIGIN_Y)
    return chm_before, chm_after, transform


def _block_polygon(block: slice) -> Polygon:
    """Polygon of a square pixel block in the synthetic window."""
    left = ORIGIN_X + block.start * PIXEL_M
    right = ORIGIN_X + block.stop * PIXEL_M
    top = ORIGIN_Y - block.start * PIXEL_M
    bottom = ORIGIN_Y - block.stop * PIXEL_M
    return box(left, bottom, right, top)


def test_refine_object_finds_clearing() -> None:
    """A cleared block in 20 m forest yields an envelope over the object."""
    params = DisturbanceConfig()
    block = slice(140, 260)
    chm_before, chm_after, transform = _synthetic_chm_pair(400, block)
    geometry = _block_polygon(block)
    outline = outlines.refine_object(chm_before, chm_after, transform, geometry, params)
    assert outline is not None
    assert outline.envelope.intersects(geometry)
    overlap = outline.envelope.intersection(geometry).area / geometry.area
    assert overlap > 0.9
    assert outline.envelope_ha >= params.mmu.mmu_ha
    assert outline.n_ring_px >= params.refine.min_ring_px
    assert 0.5 < outline.opening_share <= 1.0
    assert outline.n_openings >= 1


def test_refine_object_min_area_for_growth() -> None:
    """A 0.2 ha clearing fails the 1-acre envelope minimum but passes 0.11 ha."""
    params = DisturbanceConfig()
    block = slice(163, 237)
    chm_before, chm_after, transform = _synthetic_chm_pair(400, block)
    geometry = _block_polygon(block)
    default = outlines.refine_object(chm_before, chm_after, transform, geometry, params)
    assert default is None
    growth = outlines.refine_object(
        chm_before, chm_after, transform, geometry, params, min_area_ha=0.11
    )
    assert growth is not None
    assert 0.11 <= growth.envelope_ha < params.mmu.mmu_ha


def test_refine_object_ring_too_small() -> None:
    """An object filling the window leaves no ring and returns None."""
    params = DisturbanceConfig()
    block = slice(10, 90)
    chm_before, chm_after, transform = _synthetic_chm_pair(100, block)
    geometry = _block_polygon(slice(0, 100))
    outline = outlines.refine_object(chm_before, chm_after, transform, geometry, params)
    assert outline is None


def test_refine_object_shape_mismatch() -> None:
    """CHM windows of different shapes raise ValueError."""
    params = DisturbanceConfig()
    chm_before = np.zeros((10, 10), dtype="float32")
    chm_after = np.zeros((10, 11), dtype="float32")
    transform = Affine(PIXEL_M, 0.0, ORIGIN_X, 0.0, -PIXEL_M, ORIGIN_Y)
    with pytest.raises(ValueError, match="differ in shape"):
        outlines.refine_object(
            chm_before, chm_after, transform, box(0, 0, 1, 1), params
        )


def test_refine_objects_reads_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """refine_objects reads both years and records skipped objects."""
    params = DisturbanceConfig()
    block = slice(140, 260)
    chm_before, chm_after, transform = _synthetic_chm_pair(400, block)
    geometry = _block_polygon(block)
    tiny_geometry = box(0.0, 0.0, 1.0, 1.0)
    grid = _grid()
    year_inputs = tuple(
        YearInput(
            year=year,
            chm_paths=(),
            acquisition_dates=(),
            chm_dtype="float32",
            chm_scale=1.0,
            chm_nodata=None,
            crs=TEST_CRS,
            pixel_m=PIXEL_M,
        )
        for year in YEARS
    )
    site_inputs = SiteInputs(
        site="Synthetic",
        chm_root=Path("chm"),
        naip_root=Path("naip"),
        years=year_inputs,
        grid=grid,
        reference_year=YEARS[0],
    )
    chm_by_year = {2013: chm_before, 2015: chm_after}
    read_years = []

    def fake_read_bounds(
        paths: tuple,
        grid: CellGrid,
        bounds: tuple[float, float, float, float],
        *,
        band: int = 1,
        scale: float = 1.0,
        nodata: float | None = None,
    ) -> tuple[np.ndarray, Affine]:
        """Return the synthetic window for the year being read (before, after)."""
        year = 2013 if len(read_years) % 2 == 0 else 2015
        read_years.append(year)
        return chm_by_year[year], transform

    monkeypatch.setattr(windows, "read_bounds", fake_read_bounds)
    objects = gpd.GeoDataFrame(
        {
            "object_id": [1, 2],
            "onset_year": [2015, 2015],
            "year_before": [2013, 2013],
            "interval": ["2013-2015", "2013-2015"],
        },
        geometry=[geometry, tiny_geometry],
        crs=TEST_CRS,
    )
    result = outlines.refine_objects(objects, site_inputs, params)
    assert result.n_objects == 2
    assert result.n_refined == 1
    assert result.skipped_object_ids == [2]
    assert list(result.outlines.columns) == list(outlines.REFINED_COLUMNS)
    assert list(result.openings.columns) == list(outlines.OPENING_COLUMNS)
    assert read_years == [2013, 2015, 2013, 2015]
