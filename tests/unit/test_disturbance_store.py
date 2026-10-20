"""Tests for the disturbance run folder layout, raster and vector I/O, reload."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import box

from csdv_core.config._models import DisturbanceConfig
from csdv_core.disturbance.inputs import CellGrid
from csdv_core.disturbance.layers import CellLayers
from csdv_core.disturbance.outlines import (
    OBJECT_COLUMNS,
    OPENING_COLUMNS,
    REFINED_COLUMNS,
    RefinementResult,
)
from csdv_core.disturbance.store import (
    OBJECTS_LAYER,
    REFINED_OPENINGS_LAYER,
    REFINED_OUTLINES_LAYER,
    RunPaths,
    claim_output_dir,
    load_run,
    read_layers,
    read_run_params,
    read_year_stack,
    read_z,
    write_detection,
    write_layers,
    write_run_params,
    write_year_stack,
    write_z,
)

#: Years of the synthetic run.
YEARS = (2011, 2013, 2015)
#: Synthetic cell grid shape.
CELL_ROWS = 6
CELL_COLS = 5


@pytest.fixture
def grid() -> CellGrid:
    """A small UTM cell grid of 19.2 m cells over 0.6 m pixels."""
    crs = CRS.from_epsg(32617)
    transform = Affine(0.6, 0.0, 500_000.0, 0.0, -0.6, 4_000_000.0)
    return CellGrid(
        crs=crs,
        pixel_transform=transform,
        pixel_m=0.6,
        block_px=32,
        cell_rows=CELL_ROWS,
        cell_cols=CELL_COLS,
    )


@pytest.fixture
def layers() -> CellLayers:
    """Synthetic cell layers with a NaN cell and forest = cover >= 0.5."""
    rng = np.random.default_rng(0)
    shape = (len(YEARS), CELL_ROWS, CELL_COLS)
    height = rng.uniform(0.0, 30.0, shape).astype("float32")
    cover = rng.uniform(0.0, 1.0, shape).astype("float32")
    height[0, 0, 0] = np.nan
    cover[0, 0, 0] = np.nan
    valid_share = np.ones(shape, dtype="float32")
    cover_filled = np.nan_to_num(cover, nan=-1.0)
    forest = cover_filled >= 0.5
    return CellLayers(
        years=YEARS,
        chm_height=height,
        chm_cover=cover,
        valid_share=valid_share,
        forest=forest,
    )


def _synthetic_z() -> np.ndarray:
    """A z stack with NaN at index 0 and values beyond the int16 range."""
    rng = np.random.default_rng(1)
    shape = (len(YEARS), CELL_ROWS, CELL_COLS)
    z = rng.normal(0.0, 2.0, shape).astype("float32")
    z[0] = np.nan
    z[1, 0, 0] = 500.0
    return z


def test_claim_output_dir_creates_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "nested" / "run"
    run_paths = claim_output_dir(run_dir)
    assert isinstance(run_paths, RunPaths)
    assert run_paths.root == run_dir
    for folder in ("layers", "z", "detection", "figures"):
        assert (run_dir / folder).is_dir()


def test_claim_output_dir_refuses_existing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(FileExistsError):
        claim_output_dir(run_dir)


def test_float_year_stack_round_trip(tmp_path: Path, grid: CellGrid) -> None:
    stack = np.full((len(YEARS), CELL_ROWS, CELL_COLS), 2.5, dtype="float64")
    stack[1, 2, 3] = np.nan
    path = write_year_stack(tmp_path / "height.tif", stack, grid, YEARS)
    with rasterio.open(path) as dataset:
        assert dataset.dtypes[0] == "float32"
        assert dataset.descriptions == ("2011", "2013", "2015")
        assert dataset.compression.name == "deflate"
        assert dataset.transform == grid.cell_transform
    loaded = read_year_stack(path)
    assert loaded.years == YEARS
    assert loaded.values.dtype == np.float32
    np.testing.assert_array_equal(loaded.values, stack.astype("float32"))


def test_bool_year_stack_round_trip(tmp_path: Path, grid: CellGrid) -> None:
    rng = np.random.default_rng(2)
    stack = rng.random((len(YEARS), CELL_ROWS, CELL_COLS)) > 0.5
    path = write_year_stack(tmp_path / "forest.tif", stack, grid, YEARS)
    with rasterio.open(path) as dataset:
        assert dataset.dtypes[0] == "uint8"
        assert dataset.nodata is None
    loaded = read_year_stack(path)
    assert loaded.values.dtype == bool
    np.testing.assert_array_equal(loaded.values, stack)


def test_year_stack_rejects_bad_shape_and_dtype(tmp_path: Path, grid: CellGrid) -> None:
    wrong_shape = np.zeros((2, CELL_ROWS, CELL_COLS), dtype="float32")
    with pytest.raises(ValueError, match="shape"):
        write_year_stack(tmp_path / "bad.tif", wrong_shape, grid, YEARS)
    integer_stack = np.zeros((len(YEARS), CELL_ROWS, CELL_COLS), dtype="int32")
    with pytest.raises(ValueError, match="float or bool"):
        write_year_stack(tmp_path / "bad.tif", integer_stack, grid, YEARS)


def test_z_round_trip(tmp_path: Path, grid: CellGrid) -> None:
    z = _synthetic_z()
    path = write_z(tmp_path / "z.tif", z, grid, YEARS)
    with rasterio.open(path) as dataset:
        assert dataset.dtypes[0] == "int16"
        assert dataset.nodata == -32768
        assert dataset.descriptions == ("2011", "2013", "2015")
        stored = dataset.read()
    assert stored[1, 0, 0] == 32767
    assert np.all(stored[0] == -32768)
    loaded = read_z(path)
    assert loaded.years == YEARS
    assert loaded.values.dtype == np.float32
    assert np.all(np.isnan(loaded.values[0]))
    expected = np.round(z[2] * 100) / 100
    np.testing.assert_allclose(loaded.values[2], expected, atol=1e-6)


def test_read_z_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_z(tmp_path / "missing.tif")


def test_layers_round_trip(tmp_path: Path, grid: CellGrid, layers: CellLayers) -> None:
    run_paths = claim_output_dir(tmp_path / "run")
    written = write_layers(run_paths, layers, grid)
    assert set(written) == {"chm_height", "chm_cover", "valid_share", "forest"}
    loaded = read_layers(run_paths)
    assert loaded.years == YEARS
    assert loaded.ndvi is None
    np.testing.assert_array_equal(loaded.chm_height, layers.chm_height)
    np.testing.assert_array_equal(loaded.forest, layers.forest)


def test_read_layers_missing_file(tmp_path: Path) -> None:
    run_paths = claim_output_dir(tmp_path / "run")
    with pytest.raises(FileNotFoundError):
        read_layers(run_paths)


def test_run_params_round_trip(tmp_path: Path, grid: CellGrid) -> None:
    run_paths = claim_output_dir(tmp_path / "run")
    params = DisturbanceConfig()
    write_run_params(
        run_paths,
        params,
        site="TestSite",
        command_line="csdv detect-disturbance --site TestSite",
        config_path=None,
        from_run=None,
        grid=grid,
    )
    saved = read_run_params(run_paths.root)
    assert saved["site"] == "TestSite"
    assert saved["config_path"] is None
    assert saved["grid"]["block_px"] == 32
    rebuilt = DisturbanceConfig.model_validate(saved["params"])
    assert rebuilt == params


def _write_synthetic_run(
    run_dir: Path, grid: CellGrid, layers: CellLayers, *, with_grid: bool = True
) -> DisturbanceConfig:
    """Write params, layers and z of a small run and return its parameters."""
    params = DisturbanceConfig()
    run_paths = claim_output_dir(run_dir)
    saved_grid = grid if with_grid else None
    write_run_params(
        run_paths,
        params,
        site="TestSite",
        command_line=None,
        config_path=None,
        from_run=None,
        grid=saved_grid,
    )
    write_layers(run_paths, layers, grid)
    z_path = run_paths.z_path(params)
    z = _synthetic_z()
    write_z(z_path, z, grid, YEARS)
    return params


def test_load_run_round_trip(
    tmp_path: Path, grid: CellGrid, layers: CellLayers
) -> None:
    run_dir = tmp_path / "run"
    params = _write_synthetic_run(run_dir, grid, layers)
    saved = load_run(run_dir)
    assert saved.site == "TestSite"
    assert saved.params == params
    assert saved.grid == grid
    assert saved.years == YEARS
    assert saved.z.shape == (len(YEARS), CELL_ROWS, CELL_COLS)
    assert saved.run_paths.root == run_dir
    np.testing.assert_array_equal(saved.layers.forest, layers.forest)


def test_load_run_without_grid_fails(
    tmp_path: Path, grid: CellGrid, layers: CellLayers
) -> None:
    run_dir = tmp_path / "run"
    _write_synthetic_run(run_dir, grid, layers, with_grid=False)
    with pytest.raises(FileNotFoundError, match="grid"):
        load_run(run_dir)


def test_load_run_missing_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path / "absent")


def _objects_frame(grid: CellGrid) -> gpd.GeoDataFrame:
    """One detected object in the grid's top-left corner."""
    left, _, _, top = grid.bounds
    polygon = box(left, top - 38.4, left + 38.4, top)
    record = {
        "object_id": [1],
        "onset_year": [2013],
        "year_before": [2011],
        "interval": ["2011-2013"],
        "onset_index": [1],
        "n_cells": [4],
        "area_ha": [0.147456],
        "mean_prob": [0.8],
        "max_prob": [0.9],
        "mean_z": [-4.0],
        "follow_up": [True],
    }
    frame = pd.DataFrame(record)
    objects = gpd.GeoDataFrame(frame, geometry=[polygon], crs=grid.crs)
    return objects[list(OBJECT_COLUMNS)]


def _refinement(grid: CellGrid, objects: gpd.GeoDataFrame) -> RefinementResult:
    """A refinement result with one envelope and its openings."""
    envelope = objects.geometry.iloc[0].buffer(-2.0)
    outlines = gpd.GeoDataFrame(
        {
            "object_id": [1],
            "onset_year": [2013],
            "year_before": [2011],
            "interval": ["2011-2013"],
            "envelope_ha": [0.12],
            "openings_ha": [0.05],
            "opening_share": [0.4],
            "n_openings": [2],
            "n_ring_px": [5000],
        },
        geometry=[envelope],
        crs=grid.crs,
    )
    openings = gpd.GeoDataFrame(
        {"object_id": [1], "onset_year": [2013], "openings_ha": [0.05]},
        geometry=[envelope.buffer(-5.0)],
        crs=grid.crs,
    )
    return RefinementResult(
        outlines=outlines[list(REFINED_COLUMNS)],
        openings=openings[list(OPENING_COLUMNS)],
        n_objects=1,
        n_refined=1,
        skipped_object_ids=[],
    )


def test_write_detection(tmp_path: Path, grid: CellGrid) -> None:
    run_paths = claim_output_dir(tmp_path / "run")
    onset = np.zeros((CELL_ROWS, CELL_COLS), dtype="int16")
    onset[:2, :2] = 2013
    onset[-1, -1] = -1
    probability = np.zeros((len(YEARS), CELL_ROWS, CELL_COLS), dtype="float32")
    objects = _objects_frame(grid)
    refinement = _refinement(grid, objects)
    written = write_detection(
        run_paths,
        onset_year=onset,
        onset_year_nommu=onset,
        probability=probability,
        grid=grid,
        years=YEARS,
        objects=objects,
        refinement=refinement,
    )
    assert set(written) == {
        "onset_year",
        "onset_year_nommu",
        "probability",
        "disturbances_gpkg",
        "objects_csv",
    }
    with rasterio.open(run_paths.onset_year) as dataset:
        assert dataset.dtypes[0] == "int16"
        assert dataset.nodata is None
        stored_onset = dataset.read(1)
    np.testing.assert_array_equal(stored_onset, onset)
    stored_probability = read_year_stack(run_paths.probability)
    assert stored_probability.years == YEARS
    for layer_name in (OBJECTS_LAYER, REFINED_OUTLINES_LAYER, REFINED_OPENINGS_LAYER):
        layer = gpd.read_file(run_paths.disturbances_gpkg, layer=layer_name)
        assert len(layer) == 1
    stored_objects = gpd.read_file(run_paths.disturbances_gpkg, layer=OBJECTS_LAYER)
    assert stored_objects["follow_up"].tolist() == [True]
    table = pd.read_csv(run_paths.objects_csv)
    assert "centroid_lon" in table.columns
    assert "follow_up" in table.columns
    assert "geometry" not in table.columns


def test_write_detection_d1_without_probability(tmp_path: Path, grid: CellGrid) -> None:
    run_paths = claim_output_dir(tmp_path / "run")
    onset = np.zeros((CELL_ROWS, CELL_COLS), dtype="int16")
    objects = _objects_frame(grid)
    written = write_detection(
        run_paths,
        onset_year=onset,
        onset_year_nommu=onset,
        probability=None,
        grid=grid,
        years=YEARS,
        objects=objects,
        refinement=None,
    )
    assert "probability" not in written
    assert not run_paths.probability.exists()
