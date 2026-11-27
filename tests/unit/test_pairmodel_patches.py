"""The pair model's maps as patches: growth, ownership, the join, the seam."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from click.testing import CliRunner
from pyproj import Transformer
from rasterio.transform import from_origin

from csdv_core.pairmodel import patches as P

CELL = 9.6
NAMES = ["2012-06-07_2014-07-05"]
NORTH, SOUTH = "q3908654", "q3908662"  # adjoining quads, one above the other
ROWS, COLS = 40, 30
#: Patch across the shared edge: 13 rows by 7 columns at 95 percent.
PATCH = (slice(14, 27), slice(10, 17))


def _origin() -> tuple[float, float]:
    """Top left of a map whose middle row lies on the edge of the two quads."""
    x, y = Transformer.from_crs(4326, 26916, always_xy=True).transform(-86.30, 39.125)
    return (round(x / CELL) * CELL, (round(y / CELL) + ROWS // 2) * CELL)


def _write(path: Path, data: np.ndarray, transform, names: list[str], **kw) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "height": data.shape[1], "width": data.shape[2],
        "count": data.shape[0], "dtype": data.dtype, "crs": "EPSG:26916",
        "transform": transform, **kw,
    }  # fmt: skip
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for k, name in enumerate(names):
            dst.set_band_description(k + 1, name)


def _export(root: Path, unit: str, prob: np.ndarray, agent: np.ndarray, top=None):
    """One unit in the layout of the exports, with cover that halves in PATCH."""
    x0, y0 = _origin()
    transform = from_origin(x0, top if top is not None else y0, CELL, CELL)
    base = root / "run_layers" / unit / "default"
    _write(
        base / "model" / f"prob_forest_{unit}.tif", prob, transform, NAMES, nodata=255
    )
    _write(base / "model" / f"agent_{unit}.tif", agent, transform, NAMES, nodata=255)
    (base / "model" / "checkpoint.txt").write_text("run_x\n")
    coarse = from_origin(transform.c, transform.f, 2 * CELL, 2 * CELL)
    cover = np.full((2, ROWS // 2, COLS // 2), 0.8, dtype=np.float32)
    cover[1, 7:14, 5:9] = 0.4
    for name, scale in (("chm_cover", 1.0), ("chm_height", 20.0)):
        _write(
            base / "layers" / f"{name}.tif", cover * scale, coarse, ["2012", "2014"],
            nodata=float("nan"),
        )  # fmt: skip


@pytest.fixture()
def block(tmp_path: Path) -> Path:
    """Two adjoining quads whose maps cover the same cells, one patch across."""
    prob = np.full((1, ROWS, COLS), 5, dtype=np.uint8)
    prob[0][PATCH] = 95
    agent = np.full((1, ROWS, COLS), 1, dtype=np.uint8)
    agent[0][PATCH] = 3
    for unit in (NORTH, SOUTH):
        _export(tmp_path, unit, prob, agent)
    return tmp_path


def _box(unit: str):
    from csdv_core.grid.cells import quad_bounds

    return quad_bounds(unit)


def test_grow_from_core_and_smallest_patch() -> None:
    band = np.full((40, 40), 10, dtype=np.uint8)
    band[4:12, 4:12] = 60  # ring around a core
    band[5:11, 5:11] = 95
    band[20:26, 20:26] = 60  # no core: never a patch
    band[30:32, 30:32] = 95  # core, under 0.3 ha
    band[0, 0] = 255
    cell_ha = CELL * CELL / 1e4
    labels, n = P.grow_patches(band, P.PatchSettings(90, 50), cell_ha)
    assert n == 1 and int((labels == 1).sum()) == 64
    labels, n = P.grow_patches(band, P.PatchSettings(90, 90), cell_ha)
    assert n == 1 and int((labels == 1).sum()) == 36
    labels, n = P.grow_patches(band, P.PatchSettings(50, 50), cell_ha)
    assert n == 2


def test_small_holes_are_filled_and_large_ones_kept() -> None:
    band = np.full((60, 60), 95, dtype=np.uint8)
    band[[0, -1], :] = 5
    band[:, [0, -1]] = 5
    band[10, 10] = 0  # one cell that was not forest
    band[11, 11] = 255  # one without a value, touching it at a corner only
    band[30:40, 30:40] = 5  # 100 cells, 0.92 ha: stays a hole
    labels, n = P.grow_patches(band, P.PatchSettings(), CELL * CELL / 1e4)
    assert n == 1
    assert labels[10, 10] == 1 and labels[11, 11] == 1
    assert not labels[30:40, 30:40].any()
    assert not labels[0].any()  # ground that reaches the map's edge is no hole


def test_settings_reject_growth_above_threshold() -> None:
    with pytest.raises(ValueError):
        P.PatchSettings(threshold=50, grow=90)


def test_every_cell_has_one_owner(block: Path) -> None:
    owned = []
    for unit in (NORTH, SOUTH):
        folder = block / "run_layers" / unit / "default" / "model"
        owned.append(P.owned_cells(P.read_unit_map(folder, unit, _box(unit))))
    assert owned[0].any() and owned[1].any()
    assert (owned[0] ^ owned[1]).all()
    # The northern quad owns the rows above the edge.
    assert owned[0][0].all() and owned[1][-1].all()


def test_patch_is_joined_across_the_edge(block: Path) -> None:
    settings = P.PatchSettings()
    frames = []
    for unit in (NORTH, SOUTH):
        base = block / "run_layers" / unit / "default"
        unit_map = P.read_unit_map(base / "model", unit, _box(unit))
        frame = P.unit_patches(unit_map, settings, base / "layers")
        assert len(frame) == 1 and bool(frame["on_edge"].iloc[0])
        frames.append(frame)
    cells = sum(int(f["cells"].iloc[0]) for f in frames)
    assert cells == 13 * 7
    layer = P.block_patches(frames, settings)
    assert len(layer) == 1
    row = layer.iloc[0]
    assert row["parts"] == 2 and row["units"] == f"{NORTH}+{SOUTH}"
    assert row["area_ha"] == pytest.approx(91 * CELL * CELL / 1e4, abs=0.002)
    assert row["cause"] == "fire" and row["cause_share"] == 1.0
    assert row["mean_percent"] == 95 and row["core_share"] == 1.0
    assert row["cover_before"] == pytest.approx(0.8, abs=1e-3)
    assert row["cover_loss"] == pytest.approx(0.5, abs=1e-3)
    assert row["height_loss"] == pytest.approx(0.5, abs=1e-3)
    assert abs(row["bg_cover_loss"]) < 0.05
    assert layer.crs.to_epsg() == 5070
    assert layer.geometry.iloc[0].area == pytest.approx(91 * CELL * CELL, rel=0.01)


def test_part_on_the_edge_without_a_neighbour_is_dropped(tmp_path: Path) -> None:
    # Three rows of the patch lie in the northern quad: under the smallest
    # size, kept for the join, dropped when no neighbour brings the rest.
    prob = np.full((1, ROWS, COLS), 5, dtype=np.uint8)
    prob[0, 17:30, 10:17] = 95
    _export(tmp_path, NORTH, prob, np.ones_like(prob))
    base = tmp_path / "run_layers" / NORTH / "default"
    unit_map = P.read_unit_map(base / "model", NORTH, _box(NORTH))
    frame = P.unit_patches(unit_map, P.PatchSettings(), base / "layers")
    assert len(frame) == 1 and frame["area_ha"].iloc[0] < 0.3
    assert len(P.block_patches([frame], P.PatchSettings())) == 0


def test_seam_table_reads_the_shared_cells_from_disk(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    prob = rng.integers(0, 101, size=(1, ROWS, COLS)).astype(np.uint8)
    _, top = _origin()
    _export(tmp_path, NORTH, prob, np.ones_like(prob))
    # The southern map starts 30 rows lower: 10 rows are shared.
    lower = np.full_like(prob, 5)
    lower[0, :10] = prob[0, 30:]
    lower[0, 0, :15] = 0
    _export(tmp_path, SOUTH, lower, np.ones_like(prob), top=top - 30 * CELL)
    maps = []
    for unit in (NORTH, SOUTH):
        folder = tmp_path / "run_layers" / unit / "default" / "model"
        unit_map = P.read_unit_map(folder, unit, _box(unit))
        unit_map.release()
        assert unit_map.prob is None
        maps.append(unit_map)
    rows = P.seam_table(maps, P.PatchSettings())
    assert len(rows) == 1
    row = rows[0]
    assert (row["unit_a"], row["unit_b"]) == (NORTH, SOUTH)
    assert row["cells"] == 10 * COLS and row["same_dates"]
    changed = prob[0, 30, :15]
    assert row["mean_abs_diff"] == pytest.approx(changed.sum() / (10 * COLS), abs=0.01)
    assert row["flagged_a_pct"] >= row["flagged_both_pct"] == row["flagged_b_pct"]
    assert 0 < row["agreement"] <= 1


def test_command_writes_the_block(block: Path, tmp_path: Path) -> None:
    from csdv_core.pairmodel.cli import pair

    out = tmp_path / "out"
    args = ["patches", "--export-dir", str(block), "--out", str(out)]
    for unit in (NORTH, SOUTH, "q3908663"):
        args += ["--unit", unit]
    result = CliRunner().invoke(pair, args)
    assert result.exit_code == 0, result.output
    assert "1 patches" in result.output and "1 units have no model map" in result.output
    table = pd.read_csv(out / "patches.csv")
    assert list(table.columns[:4]) == ["patch_id", "unit", "units", "interval"]
    assert table["interval"].tolist() == ["2012_2014"]
    assert table["model"].tolist() == ["run_x"]
    assert -86.31 < table["lon"].iloc[0] < -86.29
    summary = json.loads((out / "summary.json").read_text())
    assert summary["patches"] == 1 and summary["joined"] == 1
    assert summary["without_map"] == ["q3908663"]
    assert summary["by_interval_cause"][0]["cause"] == "fire"
    seams = pd.read_csv(out / "seams.csv")
    assert len(seams) == 1 and seams["agreement"].iloc[0] == 1.0
    assert (out / "patches.gpkg").is_file()
    background = pd.read_csv(out / "background.csv")
    assert background["unit"].tolist() == [NORTH, SOUTH]
    assert (background["bg_cover_before"] == 0.8).all()
    assert (background["bg_cover_loss"].abs() < 0.05).all()
    assert (background["flagged_pct"] > 0).all()
    assert summary["cover_shifts"] == 0
