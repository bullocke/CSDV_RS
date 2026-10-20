"""Tests for the disturbance run summary, summary.json and report.md."""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import box

from csdv_core.config._models import DisturbanceConfig
from csdv_core.disturbance.decide import DateFit, DecisionResult
from csdv_core.disturbance.inputs import CellGrid
from csdv_core.disturbance.layers import CellLayers
from csdv_core.disturbance.report import (
    REPORTED_PACKAGES,
    InputFileInfo,
    RunSummary,
    build_summary,
    grid_summary,
    read_summary_json,
    software_info,
    summary_from_dict,
    summary_to_dict,
    write_report_markdown,
    write_summary_json,
)
from csdv_core.disturbance.store import RunPaths, claim_output_dir

#: Years of the synthetic run.
YEARS = (2011, 2013, 2015)
#: Synthetic cell grid shape.
CELL_ROWS = 6
CELL_COLS = 5
#: Section headings every report must hold.
EXPECTED_SECTIONS = (
    "## Run summary",
    "## Parameters",
    "## Inputs",
    "## Grid",
    "## Forest area per year",
    "## Detections by interval",
    "## Empirical null fits per date",
    "## Figures",
    "## Outputs",
    "## Runtime per stage",
    "## Software",
    "## Caveats",
)


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


def _summary(tmp_path: Path, grid: CellGrid) -> tuple[RunSummary, RunPaths]:
    """Build a summary of a synthetic run with one detected object."""
    run_paths = claim_output_dir(tmp_path / "run")
    run_paths.run_params.write_text("{}", encoding="utf-8")
    shape = (len(YEARS), CELL_ROWS, CELL_COLS)
    forest = np.ones(shape, dtype=bool)
    forest[:, 0, :] = False
    filler = np.zeros(shape, dtype="float32")
    layers = CellLayers(
        years=YEARS,
        chm_height=filler,
        chm_cover=filler,
        valid_share=filler,
        forest=forest,
    )
    monitored = np.zeros(shape, dtype=bool)
    monitored[1:] = forest[:-1]
    onset_index = np.full((CELL_ROWS, CELL_COLS), -1, dtype="int8")
    onset_index[2:4, 1:3] = 1
    onset_index[5, 4] = 2
    onset_index_mmu = onset_index.copy()
    onset_index_mmu[5, 4] = -1
    fits = [
        DateFit(2011, 0, math.nan, math.nan, math.nan, "skipped", True),
        DateFit(2013, 24, -0.02, 1.01, 0.97, "central", False),
        DateFit(2015, 24, math.nan, math.nan, math.nan, "skipped", True),
    ]
    decision = DecisionResult(
        rule="D6",
        onset_index=onset_index,
        onset_index_mmu=onset_index_mmu,
        probability=None,
        fits=fits,
    )
    left, _, _, top = grid.bounds
    polygon = box(left + 19.2, top - 76.8, left + 57.6, top - 38.4)
    objects = gpd.GeoDataFrame(
        pd.DataFrame({"object_id": [1], "onset_index": [1], "onset_year": [2013]}),
        geometry=[polygon],
        crs=grid.crs,
    )
    inputs = [
        InputFileInfo(
            year=year,
            kind="chm",
            path=f"/data/naip_chm/TestSite/{year}/CHM_TestSite_{year}0612_0p6m.tif",
            acquisition_date=f"{year}-06-12",
            dtype="float32",
            scale=1.0,
            nodata=None,
        )
        for year in YEARS
    ]
    summary = build_summary(
        site="TestSite",
        params=DisturbanceConfig(),
        inputs=inputs,
        grid=grid,
        layers=layers,
        monitored=monitored,
        decision=decision,
        objects=objects,
        refinement=None,
        stage_seconds={"inputs": 1.5, "layers": 3.25},
        run_paths=run_paths,
        command_line="csdv detect-disturbance --site TestSite",
        config_path=None,
        from_run=None,
    )
    return summary, run_paths


def test_software_info_keys() -> None:
    info = software_info()
    for key in ("csdv_core", "git_commit", "git_dirty", "python"):
        assert key in info
    for package_name in REPORTED_PACKAGES:
        assert package_name in info
    assert info["git_dirty"] in {"true", "false", "unknown"}


def test_software_info_without_git_repo(tmp_path: Path) -> None:
    info = software_info(repo_root=tmp_path)
    assert info["git_commit"] == "unknown"
    assert info["git_dirty"] == "unknown"


def test_grid_summary_window_and_sigmas(grid: CellGrid) -> None:
    summary = grid_summary(grid, DisturbanceConfig())
    assert summary.crs == "EPSG:32617"
    assert summary.cell_m == pytest.approx(19.2)
    assert summary.window_cells == 9
    assert summary.window_m_effective == pytest.approx(172.8)
    assert summary.sigma_cells == pytest.approx((10 / 19.2, 25 / 19.2))
    scene_params = DisturbanceConfig.model_validate(
        {"standardize": {"method": "scene"}}
    )
    scene_summary = grid_summary(grid, scene_params)
    assert scene_summary.window_cells is None
    assert scene_summary.window_m_effective is None


def test_build_summary_areas(tmp_path: Path, grid: CellGrid) -> None:
    summary, _ = _summary(tmp_path, grid)
    cell_ha = grid.cell_ha
    assert summary.years == list(YEARS)
    assert summary.forest_ha_by_year[2011] == pytest.approx(25 * cell_ha)
    assert type(summary.forest_ha_by_year[2011]) is float
    assert len(summary.intervals) == 2
    first, second = summary.intervals
    assert first.label == "2011-2013"
    assert first.monitored_ha == pytest.approx(25 * cell_ha)
    assert first.detected_ha == pytest.approx(4 * cell_ha)
    assert first.n_objects == 1
    assert first.share_of_monitored == pytest.approx(0.16)
    assert second.detected_ha == 0.0
    assert second.detected_ha_nommu == pytest.approx(cell_ha)
    assert summary.skipped_years == [2015]
    assert summary.detected_ha == pytest.approx(4 * cell_ha)
    assert summary.outputs["run_params"] == "run_params.json"


def test_summary_json_round_trip(tmp_path: Path, grid: CellGrid) -> None:
    summary, run_paths = _summary(tmp_path, grid)
    data = summary_to_dict(summary)
    assert data["intervals"][1]["fit"]["mu0"] is None
    assert "2011" in data["forest_ha_by_year"]
    path = write_summary_json(summary, run_paths.summary_json)
    loaded = read_summary_json(path)
    assert loaded.years == summary.years
    assert loaded.forest_ha_by_year == summary.forest_ha_by_year
    assert loaded.inputs == summary.inputs
    assert loaded.grid == summary.grid
    assert math.isnan(loaded.intervals[1].fit.mu0)
    assert loaded.intervals[0] == summary.intervals[0]
    assert loaded.params == summary.params


def test_summary_from_dict_rejects_newer_schema(tmp_path: Path, grid: CellGrid) -> None:
    summary, _ = _summary(tmp_path, grid)
    data = summary_to_dict(summary)
    data["schema_version"] = 99
    with pytest.raises(ValueError, match="newer"):
        summary_from_dict(data)


def test_report_markdown_sections_and_links(tmp_path: Path, grid: CellGrid) -> None:
    summary, run_paths = _summary(tmp_path, grid)
    present_figure = run_paths.figures_dir / "forest_initial.png"
    present_figure.write_bytes(b"png")
    absent_figure = run_paths.figures_dir / "z_maps.png"
    figure_paths = {"forest_initial": present_figure, "z_maps": absent_figure}
    path = write_report_markdown(
        summary, run_paths.report_md, figure_paths=figure_paths
    )
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Disturbance detection report: TestSite")
    for heading in EXPECTED_SECTIONS:
        assert heading in text
    assert "](figures/forest_initial.png)" in text
    assert "z_maps.png" not in text
    assert "`decide.rule` | D6" in text
    assert "9 cells (172.8 m)" in text
    assert "2011-2013" in text
    assert "`summary.json`" in text
    assert "—" not in text


def test_report_markdown_finds_figures_folder(tmp_path: Path, grid: CellGrid) -> None:
    summary, run_paths = _summary(tmp_path, grid)
    figure = run_paths.figures_dir / "null_fits.png"
    figure.write_bytes(b"png")
    path = write_report_markdown(summary, run_paths.report_md)
    text = path.read_text(encoding="utf-8")
    assert "](figures/null_fits.png)" in text
