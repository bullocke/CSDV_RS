"""Tests for csdv_core.disturbance.inputs (discovery, common grid, window reads)."""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from csdv_core.config._models import DisturbanceConfig
from csdv_core.disturbance.inputs import (
    build_common_grid,
    discover_site_inputs,
    discover_years,
    grid_from_dict,
    grid_to_dict,
    parse_acquisition_date,
)
from csdv_core.disturbance.windows import read_bounds, read_window
from csdv_core.io.paths import ProjectPaths

#: Pixel size of the synthetic rasters (m).
PIXEL_M = 0.6
#: Top-left corner shared by the unshifted synthetic rasters.
ORIGIN_X = 500_000.0
#: Top edge shared by the unshifted synthetic rasters.
ORIGIN_Y = 4_000_000.0
#: CRS of the synthetic rasters.
UTM_CRS = "EPSG:26916"


def write_tif(
    path: Path,
    values: np.ndarray,
    *,
    left: float = ORIGIN_X,
    top: float = ORIGIN_Y,
    crs: str = UTM_CRS,
    nodata: float | None = -9999.0,
) -> Path:
    """Write a tiny GeoTIFF (single or multi band) with 0.6 m pixels.

    Args:
        path: Output path (parents are created).
        values: ``(rows, cols)`` or ``(bands, rows, cols)`` array.
        left: Left edge (m).
        top: Top edge (m).
        crs: CRS string.
        nodata: Nodata value written to the metadata.

    Returns:
        ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    stack = values[np.newaxis] if values.ndim == 2 else values
    band_count, rows, cols = stack.shape
    transform = Affine(PIXEL_M, 0.0, left, 0.0, -PIXEL_M, top)
    profile = {
        "driver": "GTiff",
        "width": cols,
        "height": rows,
        "count": band_count,
        "dtype": stack.dtype.name,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(stack)
    return path


def ramp(rows: int, cols: int) -> np.ndarray:
    """Return a float32 ramp with distinct values per pixel.

    Args:
        rows: Number of rows.
        cols: Number of columns.

    Returns:
        float32 ``(rows, cols)``.
    """
    flat = np.arange(rows * cols, dtype="float32")
    return flat.reshape(rows, cols)


def make_site(root: Path, years: list[int], shape: tuple[int, int] = (8, 8)) -> Path:
    """Write one float32 CHM per year under ``root/<year>/``.

    Args:
        root: Site folder.
        years: Years to write.
        shape: Raster shape.

    Returns:
        ``root``.
    """
    for year in years:
        values = ramp(*shape) + year
        write_tif(root / str(year) / f"CHM_Site_{year}0612_0p6m.tif", values)
    return root


def test_parse_acquisition_date_both_naming_styles() -> None:
    """Local and CHPC file names both yield the acquisition date."""
    local_date = parse_acquisition_date(Path("CHM_Site_20180812_0p6m.tif"))
    chpc_date = parse_acquisition_date(Path("NAIP_Site_20210920_0p6m_chm.tif"))
    no_date = parse_acquisition_date(Path("CHM_Site_0p6m.tif"))
    assert local_date == datetime.date(2018, 8, 12)
    assert chpc_date == datetime.date(2021, 9, 20)
    assert no_date is None


def test_discover_years_filters_and_sorts(tmp_path: Path) -> None:
    """A years filter keeps only those years, ascending, with metadata."""
    site_root = make_site(tmp_path / "site", [2016, 2012, 2014])
    (site_root / "notes").mkdir()
    year_inputs = discover_years(site_root, years=[2016, 2012])
    found_years = [year_input.year for year_input in year_inputs]
    assert found_years == [2012, 2016]
    first = year_inputs[0]
    assert first.chm_dtype == "float32"
    assert first.chm_scale == 1.0
    assert first.chm_nodata == -9999.0
    assert first.pixel_m == pytest.approx(PIXEL_M)
    assert first.acquisition_dates == (datetime.date(2012, 6, 12),)


def test_discover_years_missing_year_raises(tmp_path: Path) -> None:
    """Requesting a year without CHM files raises FileNotFoundError."""
    site_root = make_site(tmp_path / "site", [2012, 2014])
    with pytest.raises(FileNotFoundError, match="2020"):
        discover_years(site_root, years=[2012, 2020])


def test_discover_years_needs_two_years(tmp_path: Path) -> None:
    """A single year cannot be differenced."""
    site_root = make_site(tmp_path / "site", [2012])
    with pytest.raises(ValueError, match="at least two"):
        discover_years(site_root)


def test_discover_years_lists_naip(tmp_path: Path) -> None:
    """NAIP files are listed only when a NAIP root is given, and must exist."""
    site_root = make_site(tmp_path / "chm", [2012, 2014])
    naip_root = tmp_path / "naip"
    naip_values = np.ones((4, 8, 8), dtype="uint8")
    write_tif(
        naip_root / "2012" / "NAIP_Site_20120612_0p6m.tif", naip_values, nodata=None
    )
    with pytest.raises(FileNotFoundError, match="2014"):
        discover_years(site_root, naip_root=naip_root)
    write_tif(
        naip_root / "2014" / "NAIP_Site_20140612_0p6m.tif", naip_values, nodata=None
    )
    year_inputs = discover_years(site_root, naip_root=naip_root)
    naip_counts = [len(year_input.naip_paths) for year_input in year_inputs]
    assert naip_counts == [1, 1]


def test_uint16_scaled_to_metres(tmp_path: Path) -> None:
    """uint16 centimetre CHMs resolve to scale 0.01 and read as metres."""
    site_root = tmp_path / "site"
    centimetres = np.full((4, 4), 1234, dtype="uint16")
    centimetres[0, 0] = 65535
    write_tif(site_root / "2012" / "a.tif", centimetres, nodata=65535)
    write_tif(site_root / "2014" / "a.tif", centimetres, nodata=65535)
    year_inputs = discover_years(site_root)
    year_input = year_inputs[0]
    assert year_input.chm_scale == 0.01
    grid = build_common_grid(year_inputs, 1.2)
    metres = read_window(
        year_input.chm_paths,
        grid,
        0,
        4,
        scale=year_input.chm_scale,
        nodata=year_input.chm_nodata,
    )
    assert np.isnan(metres[0, 0])
    assert metres[1, 1] == pytest.approx(12.34, abs=1e-5)


def test_missing_nodata_logs_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Files without nodata and no override log a warning."""
    site_root = tmp_path / "site"
    values = ramp(4, 4)
    write_tif(site_root / "2012" / "a.tif", values, nodata=None)
    write_tif(site_root / "2014" / "a.tif", values, nodata=None)
    year_inputs = discover_years(site_root)
    assert year_inputs[0].chm_nodata is None
    assert "no nodata" in caplog.text
    overridden = discover_years(site_root, chm_nodata=0.0)
    assert overridden[0].chm_nodata == 0.0


def test_block_px_rounding(tmp_path: Path) -> None:
    """block_px rounds grain_m / pixel_m, and a sub-pixel grain fails."""
    site_root = make_site(tmp_path / "site", [2012, 2014], shape=(40, 40))
    year_inputs = discover_years(site_root)
    grid = build_common_grid(year_inputs, 2.5)
    assert grid.block_px == 4
    assert grid.cell_m == pytest.approx(2.4)
    assert grid.cell_shape == (10, 10)
    with pytest.raises(ValueError, match="smaller than one pixel"):
        build_common_grid(year_inputs, 0.3)


def test_intersection_grid_with_shifted_extents(tmp_path: Path) -> None:
    """The grid is the intersection, snapped to reference pixels and trimmed."""
    site_root = tmp_path / "site"
    write_tif(site_root / "2012" / "a.tif", ramp(10, 10))
    shifted_left = ORIGIN_X + 3 * PIXEL_M
    shifted_top = ORIGIN_Y - 2 * PIXEL_M
    write_tif(
        site_root / "2014" / "a.tif",
        ramp(10, 10),
        left=shifted_left,
        top=shifted_top,
    )
    year_inputs = discover_years(site_root)
    grid = build_common_grid(year_inputs, 2 * PIXEL_M)
    # Overlap is 7 cols x 8 rows, trimmed to 6 x 8 for 2 px cells.
    assert grid.block_px == 2
    assert grid.pixel_shape == (8, 6)
    assert grid.pixel_transform.c == pytest.approx(shifted_left)
    assert grid.pixel_transform.f == pytest.approx(shifted_top)
    first_year = year_inputs[0]
    strip = read_window(first_year.chm_paths, grid, 0, 8, nodata=-9999.0)
    expected = ramp(10, 10)[2:10, 3:9]
    np.testing.assert_array_equal(strip, expected)


def test_reference_year_sets_alignment(tmp_path: Path) -> None:
    """A half-pixel offset year is snapped inward onto the reference pixels."""
    site_root = tmp_path / "site"
    write_tif(site_root / "2012" / "a.tif", ramp(10, 10))
    half_left = ORIGIN_X + 0.5 * PIXEL_M
    write_tif(site_root / "2014" / "a.tif", ramp(10, 10), left=half_left)
    year_inputs = discover_years(site_root)
    grid = build_common_grid(year_inputs, PIXEL_M, reference_year=2012)
    assert grid.pixel_transform.c == pytest.approx(ORIGIN_X + PIXEL_M)
    assert grid.pixel_shape == (10, 9)
    with pytest.raises(ValueError, match="reference_year"):
        build_common_grid(year_inputs, PIXEL_M, reference_year=2030)
    shifted_year = year_inputs[1]
    warped = read_window(shifted_year.chm_paths, grid, 0, 10)
    assert np.isfinite(warped).all()


def test_crs_mismatch_raises(tmp_path: Path) -> None:
    """Years on different CRSs raise a ValueError naming the year."""
    site_root = tmp_path / "site"
    write_tif(site_root / "2012" / "a.tif", ramp(4, 4))
    write_tif(site_root / "2014" / "a.tif", ramp(4, 4), crs="EPSG:26917")
    year_inputs = discover_years(site_root)
    with pytest.raises(ValueError, match="2014"):
        build_common_grid(year_inputs, PIXEL_M)


def test_two_tif_mosaic_first_file_wins(tmp_path: Path) -> None:
    """Two tiles of one year mosaic, the first file wins where both are valid."""
    site_root = tmp_path / "site"
    write_tif(site_root / "2012" / "a.tif", ramp(4, 8))
    left_tile = np.full((4, 5), 1.0, dtype="float32")
    left_tile[0, 0] = -9999.0
    right_tile = np.full((4, 5), 2.0, dtype="float32")
    right_left = ORIGIN_X + 3 * PIXEL_M
    write_tif(site_root / "2014" / "a_left.tif", left_tile)
    write_tif(site_root / "2014" / "b_right.tif", right_tile, left=right_left)
    year_inputs = discover_years(site_root)
    mosaic_year = year_inputs[1]
    assert len(mosaic_year.chm_paths) == 2
    grid = build_common_grid(year_inputs, PIXEL_M)
    assert grid.pixel_shape == (4, 8)
    mosaic = read_window(mosaic_year.chm_paths, grid, 0, 4)
    assert np.isnan(mosaic[0, 0])
    np.testing.assert_array_equal(mosaic[:, 1:5], 1.0)
    np.testing.assert_array_equal(mosaic[:, 5:], 2.0)


def test_read_window_rejects_bad_strip(tmp_path: Path) -> None:
    """Strips outside the grid and empty path lists raise ValueError."""
    site_root = make_site(tmp_path / "site", [2012, 2014])
    year_inputs = discover_years(site_root)
    grid = build_common_grid(year_inputs, PIXEL_M)
    first_paths = year_inputs[0].chm_paths
    with pytest.raises(ValueError, match="outside"):
        read_window(first_paths, grid, 4, 8)
    with pytest.raises(ValueError, match="No raster"):
        read_window([], grid, 0, 2)


def test_read_bounds_clips_to_grid(tmp_path: Path) -> None:
    """read_bounds snaps the bounds to pixels and clips them to the grid."""
    site_root = make_site(tmp_path / "site", [2012, 2014])
    year_inputs = discover_years(site_root)
    grid = build_common_grid(year_inputs, PIXEL_M)
    first_paths = year_inputs[0].chm_paths
    bounds = (
        ORIGIN_X - 6.0,
        ORIGIN_Y - 2 * PIXEL_M,
        ORIGIN_X + 3 * PIXEL_M,
        ORIGIN_Y + 6.0,
    )
    values, transform = read_bounds(first_paths, grid, bounds)
    expected = ramp(8, 8)[0:2, 0:3] + 2012
    np.testing.assert_array_equal(values, expected)
    assert transform.c == pytest.approx(ORIGIN_X)
    assert transform.f == pytest.approx(ORIGIN_Y)
    far_bounds = (0.0, 0.0, 10.0, 10.0)
    with pytest.raises(ValueError, match="overlap"):
        read_bounds(first_paths, grid, far_bounds)


def test_discover_site_inputs_and_grid_round_trip(tmp_path: Path) -> None:
    """Site discovery uses the project roots, and the grid serializes."""
    data_root = tmp_path / "data"
    make_site(data_root / "naip_chm" / "Site", [2012, 2014, 2016], shape=(8, 8))
    paths = ProjectPaths(
        data_root=data_root,
        results_root=tmp_path / "results",
        cache_root=tmp_path / "cache",
    )
    params = DisturbanceConfig.model_validate(
        {"grid": {"grain_m": 2.4, "reference_year": 2014}}
    )
    site_inputs = discover_site_inputs("Site", params, paths=paths)
    assert site_inputs.year_list == (2012, 2014, 2016)
    assert site_inputs.reference_year == 2014
    assert site_inputs.chm_root == data_root / "naip_chm" / "Site"
    assert site_inputs.grid.cell_shape == (2, 2)
    restored = grid_from_dict(grid_to_dict(site_inputs.grid))
    assert restored == site_inputs.grid
