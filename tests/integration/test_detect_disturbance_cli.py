"""End-to-end test of ``csdv detect-disturbance`` and ``disturbance-figures``.

Four synthetic NAIP-CHM years (240 x 240 pixels at 0.6 m) are written under a
temporary ``CSDV_DATA_ROOT``. The canopy is a spatially correlated height
field that changes a little every year, so the background behaves like real
noise rather than independent pixels. A clearcut appears in the third year
and stays open in the fourth. The 2015 year is stored as uint16 centimetres
with a nodata block and an extent shifted by a few pixels. The run uses
2.4 m cells and a 30 m window so the test stays small.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from click.testing import CliRunner
from rasterio.transform import from_origin
from scipy import ndimage

from csdv_core.cli import main

pytestmark = pytest.mark.slow

#: Site name of the synthetic data.
SITE = "SYNTH"
#: NAIP years of the synthetic data.
YEARS: tuple[int, ...] = (2013, 2015, 2017, 2019)
#: Year in which the clearcut appears.
CLEARCUT_YEAR = 2017
#: Year stored as uint16 centimetres with a shifted extent.
UINT16_YEAR = 2015
#: Pixels per side of every synthetic CHM.
SIZE_PX = 240
#: Pixel size (m).
PIXEL_M = 0.6
#: CRS of the synthetic data (UTM 17N).
CRS = "EPSG:32617"
#: Top-left corner of the reference year (m).
ORIGIN_X = 500_000.0
ORIGIN_Y = 4_000_000.0
#: Shift of the uint16 year, in pixels east and south.
SHIFT_PX = 6
#: Clearcut rows and columns in reference pixels.
CLEARCUT_ROWS = slice(100, 150)
CLEARCUT_COLS = slice(90, 140)
#: Mean canopy height (m).
CANOPY_M = 20.0
#: SD of the spatially correlated canopy field (m).
FIELD_SD_M = 2.5
#: Gaussian sigma of the correlated field (pixels).
FIELD_SIGMA_PX = 6.0
#: SD of the year-to-year change of the correlated field (m).
YEAR_CHANGE_SD_M = 0.8
#: SD of independent pixel noise (m).
PIXEL_NOISE_M = 0.5
#: Command-line options shared by every detection run.
GRID_OPTIONS: tuple[str, ...] = ("--grain-m", "2.4", "--window-m", "30")
#: Parameter overrides for the small scene.
OVERLAY_YAML = """\
decide:
  min_fit_cells: 200
mmu:
  mmu_ha: 0.03
refine:
  min_ring_px: 500
figures:
  dpi: 60
"""


def _correlated_field(rng: np.random.Generator, sd_m: float) -> np.ndarray:
    """Spatially correlated Gaussian field with standard deviation ``sd_m``.

    Args:
        rng: Random generator.
        sd_m: Target standard deviation (m).

    Returns:
        float64 ``(SIZE_PX, SIZE_PX)`` with mean 0.
    """
    white = rng.normal(size=(SIZE_PX, SIZE_PX))
    smooth = ndimage.gaussian_filter(white, FIELD_SIGMA_PX)
    scale = sd_m / smooth.std()
    return smooth * scale


def _canopy_years(seed: int = 3) -> dict[int, np.ndarray]:
    """Canopy heights (m) per year on the reference pixel grid.

    Args:
        seed: Random seed.

    Returns:
        Year to float32 ``(SIZE_PX, SIZE_PX)`` heights.
    """
    rng = np.random.default_rng(seed)
    base_field = _correlated_field(rng, FIELD_SD_M)
    canopy = CANOPY_M + base_field
    heights: dict[int, np.ndarray] = {}
    for year in YEARS:
        year_change = _correlated_field(rng, YEAR_CHANGE_SD_M)
        pixel_noise = rng.normal(scale=PIXEL_NOISE_M, size=canopy.shape)
        year_heights = canopy + year_change + pixel_noise
        if year >= CLEARCUT_YEAR:
            stump_noise = rng.uniform(0.0, 1.0, size=canopy.shape)
            year_heights[CLEARCUT_ROWS, CLEARCUT_COLS] = stump_noise[
                CLEARCUT_ROWS, CLEARCUT_COLS
            ]
        clipped = np.clip(year_heights, 0.0, None)
        heights[year] = clipped.astype("float32")
    return heights


def _write_chm(
    path: Path, values: np.ndarray, *, left: float, top: float, nodata: float
) -> None:
    """Write one CHM GeoTIFF with 0.6 m pixels.

    Args:
        path: Output file (parents are created).
        values: ``(rows, cols)`` stored values (float32 or uint16).
        left: Left edge (m).
        top: Top edge (m).
        nodata: Nodata value in the metadata.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(left, top, PIXEL_M, PIXEL_M)
    rows, cols = values.shape
    profile = {
        "driver": "GTiff",
        "width": cols,
        "height": rows,
        "count": 1,
        "dtype": values.dtype.name,
        "crs": CRS,
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(values, 1)


def _write_site(data_root: Path) -> None:
    """Write the four synthetic CHM years under ``data_root/naip_chm/SITE``.

    Args:
        data_root: Value of ``CSDV_DATA_ROOT``.
    """
    heights = _canopy_years()
    site_dir = data_root / "naip_chm" / SITE
    for year, year_heights in heights.items():
        file_name = f"CHM_{SITE}_{year}0612_0p6m.tif"
        path = site_dir / str(year) / file_name
        if year != UINT16_YEAR:
            _write_chm(path, year_heights, left=ORIGIN_X, top=ORIGIN_Y, nodata=-9999.0)
            continue
        # Shift the extent by SHIFT_PX pixels east and south and store cm.
        shifted = np.full_like(year_heights, np.nan)
        shifted[:-SHIFT_PX, :-SHIFT_PX] = year_heights[SHIFT_PX:, SHIFT_PX:]
        centimetres = np.round(shifted * 100.0)
        stored = np.nan_to_num(centimetres, nan=65535.0).astype("uint16")
        stored[200:220, 10:30] = 65535
        shift_m = SHIFT_PX * PIXEL_M
        _write_chm(
            path,
            stored,
            left=ORIGIN_X + shift_m,
            top=ORIGIN_Y - shift_m,
            nodata=65535.0,
        )


@pytest.fixture()
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Synthetic site, project roots in ``tmp_path`` and an overlay config.

    Returns:
        Paths: ``data``, ``results``, ``config``.
    """
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    monkeypatch.setenv("CSDV_DATA_ROOT", str(data_root))
    monkeypatch.setenv("CSDV_RESULTS_ROOT", str(results_root))
    _write_site(data_root)
    config_path = tmp_path / "overlay.yaml"
    config_path.write_text(OVERLAY_YAML, encoding="utf-8")
    return {"data": data_root, "results": results_root, "config": config_path}


def _detect(config_path: Path, *extra: str) -> object:
    """Invoke ``csdv detect-disturbance`` on the synthetic site.

    Args:
        config_path: Overlay YAML.
        *extra: Further command-line options.

    Returns:
        The Click result.
    """
    arguments = [
        "detect-disturbance",
        "--site",
        SITE,
        "--config",
        str(config_path),
        *GRID_OPTIONS,
        *extra,
    ]
    runner = CliRunner()
    return runner.invoke(main, arguments, catch_exceptions=False)


def _read_band(path: Path) -> tuple[np.ndarray, rasterio.Affine]:
    """Read band 1 of a raster and its transform.

    Args:
        path: GeoTIFF.

    Returns:
        ``(array, transform)``.
    """
    with rasterio.open(path) as dataset:
        values = dataset.read(1)
        transform = dataset.transform
    return values, transform


def _clearcut_cells(onset_year: np.ndarray, transform: rasterio.Affine) -> np.ndarray:
    """Onset values of the cells that lie fully inside the clearcut.

    Args:
        onset_year: ``onset_year`` raster.
        transform: Its cell transform.

    Returns:
        1-D onset values of the inner clearcut cells.
    """
    cell_m = transform.a
    left = ORIGIN_X + CLEARCUT_COLS.start * PIXEL_M
    right = ORIGIN_X + CLEARCUT_COLS.stop * PIXEL_M
    top = ORIGIN_Y - CLEARCUT_ROWS.start * PIXEL_M
    bottom = ORIGIN_Y - CLEARCUT_ROWS.stop * PIXEL_M
    first_col = int(np.ceil((left - transform.c) / cell_m))
    last_col = int(np.floor((right - transform.c) / cell_m))
    first_row = int(np.ceil((transform.f - top) / cell_m))
    last_row = int(np.floor((transform.f - bottom) / cell_m))
    return onset_year[first_row:last_row, first_col:last_col].ravel()


def test_detect_disturbance_end_to_end(project: dict[str, Path]) -> None:
    """Full run, refusal to overwrite, --from-run and disturbance-figures."""
    config_path = project["config"]
    result = _detect(config_path)
    assert result.exit_code == 0, result.output
    run_dir = project["results"] / "disturbance" / SITE
    expected_files = [
        "run_params.json",
        "detect_disturbance.log",
        "summary.json",
        "report.md",
        "layers/chm_height.tif",
        "layers/chm_cover.tif",
        "layers/valid_share.tif",
        "layers/forest.tif",
        "z/z_d_chm_h_local_twopass_30m.tif",
        "detection/onset_year.tif",
        "detection/onset_year_nommu.tif",
        "detection/probability.tif",
        "detection/disturbances.gpkg",
        "detection/objects.csv",
        "figures/disturbance_onset.png",
        "figures/null_fits.png",
        "figures/refined_examples.png",
    ]
    for relative in expected_files:
        assert (run_dir / relative).is_file(), relative

    onset_year, transform = _read_band(run_dir / "detection" / "onset_year.tif")
    clearcut_onsets = _clearcut_cells(onset_year, transform)
    clearcut_share = float(np.mean(clearcut_onsets == CLEARCUT_YEAR))
    assert clearcut_share > 0.8
    detected = onset_year > 0
    detected_share = float(detected.mean())
    assert detected_share < 0.2

    with rasterio.open(run_dir / "detection" / "probability.tif") as dataset:
        first_band = dataset.read(1)
    assert np.isnan(first_band).all()

    gpkg_path = run_dir / "detection" / "disturbances.gpkg"
    refined = gpd.read_file(gpkg_path, layer="refined_outlines")
    assert len(refined) > 0
    assert (refined["onset_year"] == CLEARCUT_YEAR).any()

    summary_text = (run_dir / "summary.json").read_text(encoding="utf-8")
    summary = json.loads(summary_text)
    for key in ("site", "params", "grid", "intervals", "stage_seconds", "software"):
        assert key in summary
    assert summary["site"] == SITE
    assert "figures" in summary["stage_seconds"]
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "figures/disturbance_onset.png" in report_text

    second = _detect(config_path)
    assert second.exit_code != 0
    assert "already exists" in second.output
    assert "--output-dir" in second.output

    rerun_dir = project["results"] / "rerun"
    rerun_arguments = [
        "detect-disturbance",
        "--site",
        SITE,
        "--config",
        str(config_path),
        "--from-run",
        str(run_dir),
        "--output-dir",
        str(rerun_dir),
    ]
    runner = CliRunner()
    rerun = runner.invoke(main, rerun_arguments, catch_exceptions=False)
    assert rerun.exit_code == 0, rerun.output
    rerun_onset, _ = _read_band(rerun_dir / "detection" / "onset_year.tif")
    np.testing.assert_array_equal(rerun_onset, onset_year)
    rerun_params_text = (rerun_dir / "run_params.json").read_text(encoding="utf-8")
    rerun_params = json.loads(rerun_params_text)
    assert rerun_params["from_run"] == str(run_dir)
    assert rerun_params["params"]["grid"]["grain_m"] == 2.4
    assert rerun_params["params"]["standardize"]["window_m"] == 30.0

    onset_png = run_dir / "figures" / "disturbance_onset.png"
    onset_png.unlink()
    figures = runner.invoke(
        main, ["disturbance-figures", "--run-dir", str(run_dir)], catch_exceptions=False
    )
    assert figures.exit_code == 0, figures.output
    assert onset_png.is_file()
    assert (run_dir / "figures" / "refined_examples.png").is_file()


def test_from_run_rejects_changed_window(project: dict[str, Path]) -> None:
    """--from-run takes the saved window and refuses an explicit different one."""
    config_path = project["config"]
    first_dir = project["results"] / "first"
    first = _detect(config_path, "--no-figures", "--output-dir", str(first_dir))
    assert first.exit_code == 0, first.output
    second_dir = project["results"] / "second"
    arguments = [
        "detect-disturbance",
        "--site",
        SITE,
        "--config",
        str(config_path),
        "--grain-m",
        "2.4",
        "--window-m",
        "40",
        "--from-run",
        str(first_dir),
        "--output-dir",
        str(second_dir),
    ]
    runner = CliRunner()
    result = runner.invoke(main, arguments, catch_exceptions=False)
    assert result.exit_code != 0
    assert "standardize.window_m" in result.output
    assert "grid.grain_m" not in result.output
    assert not second_dir.exists()


def test_from_run_rule_d7(project: dict[str, Path]) -> None:
    """A D6 run re-decided as D7 without refinement keeps the persistent clearcut."""
    config_path = project["config"]
    first_dir = project["results"] / "d6"
    first = _detect(config_path, "--no-figures", "--output-dir", str(first_dir))
    assert first.exit_code == 0, first.output
    d7_dir = project["results"] / "d7"
    arguments = [
        "detect-disturbance",
        "--site",
        SITE,
        "--config",
        str(config_path),
        "--from-run",
        str(first_dir),
        "--rule",
        "D7",
        "--no-refine",
        "--no-figures",
        "--output-dir",
        str(d7_dir),
    ]
    runner = CliRunner()
    result = runner.invoke(main, arguments, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    for suffix in ("back2", "fwd2"):
        bracket_path = d7_dir / "z" / f"z_d_chm_h_local_twopass_30m_{suffix}.tif"
        assert bracket_path.is_file(), suffix
    onset_year, transform = _read_band(d7_dir / "detection" / "onset_year.tif")
    clearcut_onsets = _clearcut_cells(onset_year, transform)
    clearcut_share = float(np.mean(clearcut_onsets == CLEARCUT_YEAR))
    assert clearcut_share > 0.8
    gpkg_path = d7_dir / "detection" / "disturbances.gpkg"
    objects = gpd.read_file(gpkg_path, layer="objects")
    assert "follow_up" in objects.columns
    clearcut_objects = objects[objects["onset_year"] == CLEARCUT_YEAR]
    assert clearcut_objects["follow_up"].all()
    summary_text = (d7_dir / "summary.json").read_text(encoding="utf-8")
    summary = json.loads(summary_text)
    assert summary["rule"] == "D7"


def _write_naip(data_root: Path) -> None:
    """Write a 4-band uint8 NAIP image per year, shaded by the canopy height.

    Args:
        data_root: Value of ``CSDV_DATA_ROOT``.
    """
    heights = _canopy_years()
    transform = from_origin(ORIGIN_X, ORIGIN_Y, PIXEL_M, PIXEL_M)
    for year, year_heights in heights.items():
        path = data_root / "naip" / SITE / str(year) / f"NAIP_{SITE}_{year}0612.tif"
        path.parent.mkdir(parents=True, exist_ok=True)
        brightness = np.clip(220.0 - 8.0 * year_heights, 0.0, 255.0)
        band = brightness.astype("uint8")
        bands = np.stack([band, band, band, band])
        profile = {
            "driver": "GTiff",
            "width": SIZE_PX,
            "height": SIZE_PX,
            "count": 4,
            "dtype": "uint8",
            "crs": CRS,
            "transform": transform,
        }
        with rasterio.open(path, "w", **profile) as dataset:
            dataset.write(bands)


def test_disturbance_chips(project: dict[str, Path]) -> None:
    """Sample a run and draw its chips, then refuse an existing folder."""
    _write_naip(project["data"])
    detect_result = _detect(project["config"])
    assert detect_result.exit_code == 0, detect_result.output
    run_dir = project["results"] / "disturbance" / SITE
    arguments = [
        "disturbance-chips",
        "--run-dir",
        str(run_dir),
        "--n-detected",
        "3",
        "--n-below-mmu",
        "1",
        "--n-no-detection",
        "2",
        "--min-chip-m",
        "60",
    ]
    runner = CliRunner()
    result = runner.invoke(main, arguments, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    inspection_dir = run_dir / "inspection"
    samples = gpd.read_file(inspection_dir / "samples.gpkg")
    assert "detected" in set(samples["stratum"])
    assert (samples["stratum"] == "no_detection").sum() == 2
    for name in ("samples.csv", "contact_sheet.png", "index.md"):
        assert (inspection_dir / name).stat().st_size > 0
    chip_paths = sorted((inspection_dir / "chips").glob("*.png"))
    assert len(chip_paths) == len(samples)
    index_text = (inspection_dir / "index.md").read_text()
    assert "contact_sheet.png" in index_text
    repeat = runner.invoke(main, arguments, catch_exceptions=False)
    assert repeat.exit_code != 0
    assert "already exists" in repeat.output
