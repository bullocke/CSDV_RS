"""Tests for csdv_core.disturbance.layers (cell aggregation and masks)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from csdv_core.config._models import DisturbanceConfig
from csdv_core.disturbance.inputs import SiteInputs, build_common_grid, discover_years
from csdv_core.disturbance.layers import (
    CellLayers,
    aggregate_block,
    aggregate_ndvi,
    build_cell_layers,
    change_layer,
    forest_mask,
    monitored_mask,
    population_mask,
)
from csdv_core.disturbance.windows import read_window

#: Pixels per cell side in the synthetic layer tests.
BLOCK_PX = 4
#: Pixel size of the synthetic rasters (m).
PIXEL_M = 0.6


def write_tif(
    path: Path, values: np.ndarray, *, nodata: float | None = -9999.0
) -> None:
    """Write a tiny UTM GeoTIFF with 0.6 m pixels.

    Args:
        path: Output path (parents are created).
        values: ``(rows, cols)`` or ``(bands, rows, cols)`` array.
        nodata: Nodata value written to the metadata.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    stack = values[np.newaxis] if values.ndim == 2 else values
    band_count, rows, cols = stack.shape
    transform = Affine(PIXEL_M, 0.0, 500_000.0, 0.0, -PIXEL_M, 4_000_000.0)
    profile = {
        "driver": "GTiff",
        "width": cols,
        "height": rows,
        "count": band_count,
        "dtype": stack.dtype.name,
        "crs": "EPSG:26916",
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(stack)


def make_site_inputs(tmp_path: Path, *, with_naip: bool = False) -> SiteInputs:
    """Build a three-year synthetic site of 24 x 16 pixels with random CHMs.

    Args:
        tmp_path: Folder for the GeoTIFFs.
        with_naip: Also write 4-band NAIP rasters and list them.

    Returns:
        Site inputs on a 4-pixel cell grid.
    """
    generator = np.random.default_rng(0)
    chm_root = tmp_path / "naip_chm" / "Site"
    naip_root = tmp_path / "naip" / "Site"
    for year in (2012, 2014, 2016):
        heights = generator.uniform(0.0, 20.0, size=(24, 16)).astype("float32")
        gaps = generator.random((24, 16)) < 0.2
        heights[gaps] = -9999.0
        heights[:4, :4] = -9999.0
        write_tif(chm_root / str(year) / f"CHM_Site_{year}0612_0p6m.tif", heights)
        if with_naip:
            bands = generator.integers(0, 256, size=(4, 24, 16)).astype("uint8")
            bands[:, 4:8, 4:8] = 0
            naip_path = naip_root / str(year) / f"NAIP_Site_{year}0612_0p6m.tif"
            write_tif(naip_path, bands, nodata=None)
    year_inputs = discover_years(chm_root, naip_root=naip_root if with_naip else None)
    grid = build_common_grid(year_inputs, BLOCK_PX * PIXEL_M)
    return SiteInputs(
        site="Site",
        chm_root=chm_root,
        naip_root=naip_root,
        years=tuple(year_inputs),
        grid=grid,
        reference_year=2012,
    )


def test_aggregate_block_known_values() -> None:
    """Mean height, strict cover and valid share for hand-made cells."""
    chm = np.array(
        [
            [2.0, 6.0, 5.0, np.nan],
            [8.0, 4.0, np.nan, np.nan],
        ],
        dtype="float32",
    )
    aggregate = aggregate_block(chm, 2, height_m=5.0, min_valid_share=0.5)
    assert aggregate.height[0, 0] == pytest.approx(5.0)
    assert aggregate.cover[0, 0] == pytest.approx(0.5)
    assert aggregate.valid_share[0, 0] == pytest.approx(1.0)
    # One valid pixel of four: share 0.25 < 0.5, so height and cover are NaN.
    assert aggregate.valid_share[0, 1] == pytest.approx(0.25)
    assert np.isnan(aggregate.height[0, 1])
    assert np.isnan(aggregate.cover[0, 1])
    assert aggregate.height.dtype == np.float32


def test_aggregate_block_half_valid_is_defined() -> None:
    """A valid share of exactly min_valid_share keeps the cell."""
    chm = np.array([[6.0, 6.0], [np.nan, np.nan]], dtype="float32")
    aggregate = aggregate_block(chm, 2, height_m=5.0, min_valid_share=0.5)
    assert aggregate.height[0, 0] == pytest.approx(6.0)
    assert aggregate.cover[0, 0] == pytest.approx(1.0)


def test_aggregate_block_rejects_partial_blocks() -> None:
    """Shapes that are not whole blocks raise ValueError."""
    chm = np.zeros((3, 4), dtype="float32")
    with pytest.raises(ValueError, match="multiple"):
        aggregate_block(chm, 2, height_m=5.0, min_valid_share=0.5)


def test_aggregate_ndvi_formula_and_validity() -> None:
    """NDVI uses (nir - red) / max(nir + red, 1) over pixels with red + nir > 0."""
    red = np.array([[10.0, 0.0], [30.0, 0.0]], dtype="float32")
    nir = np.array([[30.0, 0.0], [10.0, 0.0]], dtype="float32")
    cell_ndvi = aggregate_ndvi(red, nir, 2, min_valid_share=0.5)
    expected = (0.5 + -0.5) / 2
    assert cell_ndvi[0, 0] == pytest.approx(expected)
    sparse = aggregate_ndvi(red, nir, 2, min_valid_share=0.75)
    assert np.isnan(sparse[0, 0])
    with pytest.raises(ValueError, match="differ"):
        aggregate_ndvi(red, nir[:1], 2, min_valid_share=0.5)


def test_forest_mask_treats_nan_as_non_forest() -> None:
    """Cover at the threshold is forest, NaN is not."""
    cover = np.array([0.49, 0.5, np.nan, 0.9], dtype="float32")
    is_forest = forest_mask(cover, min_cover=0.5)
    np.testing.assert_array_equal(is_forest, [False, True, False, True])


def test_monitored_and_population_masks() -> None:
    """Monitored is forest at t - 1, population adds forest[0] at t = 0."""
    forest = np.array([[[True, False]], [[False, True]], [[True, True]]])
    monitored = monitored_mask(forest)
    population = population_mask(forest)
    np.testing.assert_array_equal(monitored[0], [[False, False]])
    np.testing.assert_array_equal(monitored[1:], forest[:-1])
    np.testing.assert_array_equal(population[0], forest[0])
    np.testing.assert_array_equal(population[1:], forest[:-1])
    assert not monitored[0].any()


def test_change_layer_differences_and_nan_at_start() -> None:
    """Change is NaN at t = 0 and state[t] - state[t - 1] afterwards."""
    height = np.array([[[10.0]], [[7.0]], [[np.nan]]], dtype="float32")
    layers = CellLayers(
        years=(2012, 2014, 2016),
        chm_height=height,
        chm_cover=height,
        valid_share=np.ones_like(height),
        forest=np.ones(height.shape, dtype=bool),
    )
    change = change_layer(layers, "d_chm_h")
    assert np.isnan(change[0, 0, 0])
    assert change[1, 0, 0] == pytest.approx(-3.0)
    assert np.isnan(change[2, 0, 0])
    assert change.dtype == np.float32
    with pytest.raises(ValueError, match="ndvi"):
        change_layer(layers, "d_ndvi")
    with pytest.raises(ValueError, match="Unknown"):
        change_layer(layers, "d_other")


@pytest.mark.parametrize("strip_cells", [1, 2, 4])
def test_strip_layers_equal_whole_array(tmp_path: Path, strip_cells: int) -> None:
    """Strip-built layers equal aggregation of each year in one pass."""
    site_inputs = make_site_inputs(tmp_path)
    params = DisturbanceConfig()
    layers = build_cell_layers(site_inputs, params, strip_cells=strip_cells)
    grid = site_inputs.grid
    pixel_rows = grid.pixel_shape[0]
    for year_index, year_input in enumerate(site_inputs.years):
        whole = read_window(
            year_input.chm_paths, grid, 0, pixel_rows, nodata=year_input.chm_nodata
        )
        expected = aggregate_block(
            whole, grid.block_px, height_m=5.0, min_valid_share=0.5
        )
        np.testing.assert_array_equal(layers.chm_height[year_index], expected.height)
        np.testing.assert_array_equal(layers.chm_cover[year_index], expected.cover)
        np.testing.assert_array_equal(
            layers.valid_share[year_index], expected.valid_share
        )
    expected_forest = forest_mask(layers.chm_cover, min_cover=0.5)
    np.testing.assert_array_equal(layers.forest, expected_forest)
    assert layers.years == (2012, 2014, 2016)
    assert layers.cell_shape == (6, 4)
    assert np.isnan(layers.chm_height[0, 0, 0])
    assert layers.ndvi is None


def test_build_cell_layers_ndvi(tmp_path: Path) -> None:
    """NDVI layers come from NAIP red and NIR, NaN where NAIP is fill."""
    site_inputs = make_site_inputs(tmp_path, with_naip=True)
    params = DisturbanceConfig()
    layers = build_cell_layers(site_inputs, params, include_ndvi=True)
    assert layers.ndvi is not None
    assert layers.ndvi.shape == layers.chm_height.shape
    assert np.isnan(layers.ndvi[:, 1, 1]).all()
    grid = site_inputs.grid
    first_year = site_inputs.years[0]
    pixel_rows = grid.pixel_shape[0]
    red = read_window(first_year.naip_paths, grid, 0, pixel_rows, band=1)
    nir = read_window(first_year.naip_paths, grid, 0, pixel_rows, band=4)
    expected = aggregate_ndvi(red, nir, grid.block_px, min_valid_share=0.5)
    np.testing.assert_array_equal(layers.ndvi[0], expected)
    change = change_layer(layers, "d_ndvi")
    assert np.isnan(change[0]).all()


def test_build_cell_layers_ndvi_needs_naip(tmp_path: Path) -> None:
    """Asking for NDVI without NAIP files raises ValueError."""
    site_inputs = make_site_inputs(tmp_path)
    params = DisturbanceConfig()
    with pytest.raises(ValueError, match="NAIP"):
        build_cell_layers(site_inputs, params, include_ndvi=True)
    with pytest.raises(ValueError, match="strip_cells"):
        build_cell_layers(site_inputs, params, strip_cells=0)
