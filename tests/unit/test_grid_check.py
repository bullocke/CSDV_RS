"""Tests for the checks of an exported grid run (``csdv grid check``).

A small export tree whose files agree with each other is written to a
temporary folder. The clean tree must pass, and copies broken in one way each
must fail (or warn) on the right check.
"""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box

from csdv_core.grid.check import FAIL, PASS, WARN, check_run, find_runs, write_check
from csdv_core.grid.export import NODATA_U8, RUN_DIRS, write_cog

CRS = "EPSG:26919"
UNIT, TAG = "q4407159", "default"
CELL = 19.2
LEFT, TOP = 290000.0, 4890000.0
YEARS = [2018, 2021]
CELL_HA = CELL * CELL / 1e4


def _write_export(root):
    run = root / "runs" / TAG / UNIT
    looks = root / "quicklooks" / UNIT
    transform = from_origin(LEFT, TOP, CELL, CELL)

    onset = np.zeros((20, 20), dtype=np.uint8)
    onset[2:4, 2:4] = 21
    onset[-1] = NODATA_U8
    write_cog(run / "onset_year_u8.tif", onset, crs=CRS, transform=transform)
    prob = np.full((20, 20), 5, dtype=np.uint8)
    prob[-1] = NODATA_U8
    write_cog(run / "max_probability_u8.tif", prob, crs=CRS, transform=transform)

    area = 4 * CELL_HA
    square = box(LEFT + 2 * CELL, TOP - 4 * CELL, LEFT + 4 * CELL, TOP - 2 * CELL)
    objects = gpd.GeoDataFrame(
        {"object_id": [1], "interval": ["2018-2021"], "area_ha": [area]},
        geometry=[square],
        crs=CRS,
    )
    objects.to_file(run / "disturbances.gpkg", layer="objects")
    for layer in ("refined_outlines", "refined_openings"):
        objects.to_file(run / "disturbances.gpkg", layer=layer)
    objects.drop(columns="geometry").to_csv(run / "objects.csv", index=False)

    summary = {
        "years": YEARS,
        "inputs": [
            {
                "year": 2018,
                "path": "/x/NAIP_q_20180901_0p6m_chm.tif",
                "acquisition_date": "2018-09-01",
            },
            {
                "year": 2021,
                "path": "/x/NAIP_q_20210910_0p6m_chm.tif",
                "acquisition_date": "2021-09-10",
            },
        ],
        "grid": {
            "crs": CRS,
            "bounds": [LEFT, TOP - 20 * CELL, LEFT + 20 * CELL, TOP],
            "cell_m": CELL,
            "cell_rows": 20,
            "cell_cols": 20,
            "area_ha": 400 * CELL_HA,
        },
        "forest_ha_by_year": {"2018": 380 * CELL_HA, "2021": 379 * CELL_HA},
        "intervals": [
            {
                "label": "2018-2021",
                "year": 2021,
                "detected_ha": area,
                "repeat_ha": 0.0,
                "skipped": False,
                "fit": {"sigma0": 0.98, "p0": 1.0, "skipped": False},
            }
        ],
        "n_objects": 1,
        "n_refined": 1,
        "detected_ha": area,
        "ever_detected_ha": area,
        "stage_seconds": {"layers": 10.0, "decide": 1.0},
        "software": {"git_commit": "abc1234", "git_dirty": "false"},
    }
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    params = {"params": {"inputs": {"years": YEARS}}}
    (run / "run_params.json").write_text(json.dumps(params), encoding="utf-8")
    unit = {
        "unit_id": UNIT,
        "label": "Test",
        "years": {str(y): {"source": "pc", "coverage": 1.0} for y in YEARS},
        "skipped_years": {},
    }
    (run / "unit.json").write_text(json.dumps(unit), encoding="utf-8")
    export = {"unit": UNIT, "run_tag": TAG, "run_dir": "/chpc/run"}
    (run / "export.json").write_text(json.dumps(export), encoding="utf-8")
    (run / "report.md").write_text("# report\n", encoding="utf-8")
    (run / "detect_disturbance.log").write_text(
        "2026-10-29 14:28:45,694 INFO csdv: Run finished in 11.0 s: /out\n",
        encoding="utf-8",
    )
    for name in RUN_DIRS:
        (run / name).mkdir(exist_ok=True)
    (run / "inspection" / "chips").mkdir()
    (run / "inspection" / "chips" / "S01.png").write_bytes(b"png")
    (run / "inspection" / "contact_sheet.png").write_bytes(b"png")
    (run / "inspection" / "samples.csv").write_text(
        "sample_id,chip,interp_label\nS01,chips/S01.png,\n", encoding="utf-8"
    )

    fine = from_origin(LEFT, TOP, 3.0, 3.0)
    for year in YEARS:
        chm = np.full((10, 10), 44, dtype=np.uint8)
        write_cog(looks / f"chm_{year}_3m.tif", chm, crs=CRS, transform=fine)
        rgb = np.full((3, 10, 10), 100, dtype=np.uint8)
        write_cog(
            looks / f"naip_{year}_3m_rgb.tif",
            rgb,
            crs=CRS,
            transform=fine,
            nodata=None,
        )
    (root / "status.csv").write_text(
        "unit,label,years,naip,chm,run_tag,detect,chips,export,state,run_dir\n"
        f"{UNIT},Test,2,2/2,2/2,{TAG},done,done,done,done,/chpc/run\n",
        encoding="utf-8",
    )
    return run


@pytest.fixture()
def export(tmp_path):
    root = tmp_path / "exports"
    run = _write_export(root)
    return root, run


def _status(check, name):
    return {r.name: r.status for r in check.results}[name]


def test_clean_export_passes(export):
    root, run = export
    assert find_runs(root) == [(UNIT, TAG)]
    check = check_run(root, UNIT, TAG)
    assert check.status == PASS, check.to_markdown()
    write_check(check)
    saved = json.loads((run / "check.json").read_text())
    assert saved["status"] == PASS


def test_missing_file_fails_completeness(export):
    root, run = export
    (run / "objects.csv").unlink()
    assert _status(check_run(root, UNIT, TAG), "completeness") == FAIL


def test_object_count_mismatch_fails_vectors(export):
    root, run = export
    summary = json.loads((run / "summary.json").read_text())
    summary["n_objects"] = 2
    (run / "summary.json").write_text(json.dumps(summary))
    assert _status(check_run(root, UNIT, TAG), "vectors") == FAIL


def test_onset_area_mismatch_fails(export):
    root, run = export
    summary = json.loads((run / "summary.json").read_text())
    summary["intervals"][0]["detected_ha"] *= 3
    (run / "summary.json").write_text(json.dumps(summary))
    assert _status(check_run(root, UNIT, TAG), "onset raster") == FAIL


def test_wrong_crs_fails_raster_grid(export):
    root, run = export
    onset = np.zeros((20, 20), dtype=np.uint8)
    write_cog(
        run / "onset_year_u8.tif",
        onset,
        crs="EPSG:26918",
        transform=from_origin(LEFT, TOP, CELL, CELL),
    )
    assert _status(check_run(root, UNIT, TAG), "raster grid") == FAIL


def test_invalid_polygon_warns(export):
    root, run = export
    bowtie = Polygon(
        [(LEFT, TOP), (LEFT + 40, TOP - 40), (LEFT + 40, TOP), (LEFT, TOP - 40)]
    )
    objects = gpd.read_file(run / "disturbances.gpkg", layer="objects")
    objects["geometry"] = [bowtie]
    objects.to_file(run / "disturbances.gpkg", layer="objects", mode="w")
    assert _status(check_run(root, UNIT, TAG), "geometry validity") == WARN


def test_season_gap_and_dirty_tree(export):
    root, run = export
    summary = json.loads((run / "summary.json").read_text())
    summary["inputs"][0]["acquisition_date"] = "2018-06-01"
    summary["software"]["git_dirty"] = "true"
    (run / "summary.json").write_text(json.dumps(summary))
    check = check_run(root, UNIT, TAG)
    assert _status(check, "season gaps") == WARN
    assert _status(check, "metadata") == FAIL
