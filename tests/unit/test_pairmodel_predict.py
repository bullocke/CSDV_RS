"""Mapping a synthetic unit, and the whole chain from chips to comparison."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine
from click.testing import CliRunner
from rasterio.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry import box
from test_pairmodel_data import DATES, write_chip

from csdv_core.disturbance.inputs import CellGrid, grid_to_dict
from csdv_core.disturbance.store import write_year_stack
from csdv_core.pairmodel import predict as P

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

UNIT = "qtest"
YEARS = [2012, 2014, 2016]
# The unit's grid starts 14 and 12 pixels off the 9.6 m lattice, like q4412328.
LEFT_PX = 16000 + 14
TOP_PX = 80000 - 12
GRID = CellGrid(
    crs=CRS.from_epsg(26910),
    pixel_transform=Affine(0.6, 0.0, LEFT_PX * 0.6, 0.0, -0.6, TOP_PX * 0.6),
    pixel_m=0.6,
    block_px=32,
    cell_rows=12,
    cell_cols=12,
)
# A clearing of 96 by 96 pixels that appears in 2016, on the 9.6 m lattice.
CLEAR_PX = (16128, 79872, 16224, 79776)  # left, top, right, bottom


@pytest.fixture()
def unit(tmp_path: Path) -> dict[str, Path]:
    """A unit with three mosaics, a run folder and two label chips."""
    data_root = tmp_path / "data"
    pad = 100
    width = height = 384 + 2 * pad
    left, top = LEFT_PX - pad, TOP_PX + pad
    for i, (year, date) in enumerate(zip(YEARS, DATES, strict=True)):
        folder = data_root / "naip" / UNIT / str(year)
        folder.mkdir(parents=True)
        rng = np.random.default_rng(year)
        data = rng.integers(40, 90, (4, height, width), dtype=np.uint8)
        if i == 2:
            rows = slice(top - CLEAR_PX[1], top - CLEAR_PX[3])
            cols = slice(CLEAR_PX[0] - left, CLEAR_PX[2] - left)
            data[:, rows, cols] = rng.integers(170, 230, (4, 96, 96), dtype=np.uint8)
        with rasterio.open(
            folder / f"NAIP_{UNIT}_{date.replace('-', '')}_0p6m.tif",
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=4,
            dtype="uint8",
            crs="EPSG:26910",
            transform=from_origin(left * 0.6, top * 0.6, 0.6, 0.6),
        ) as dst:
            dst.write(data)
    run_dir = tmp_path / "runs" / UNIT
    run_dir.mkdir(parents=True)
    (run_dir / "run_params.json").write_text(
        json.dumps({"site": UNIT, "grid": grid_to_dict(GRID)})
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "years": YEARS,
                "inputs": [
                    {"year": y, "acquisition_date": d}
                    for y, d in zip(YEARS, DATES, strict=True)
                ],
            }
        )
    )
    clearing = box(
        CLEAR_PX[0] * 0.6, CLEAR_PX[3] * 0.6, CLEAR_PX[2] * 0.6, CLEAR_PX[1] * 0.6
    )
    gpd.GeoDataFrame(
        {
            "object_id": [1],
            "interval": ["2014-2016"],
            "area_ha": [clearing.area / 10_000],
            "mean_prob": [0.9],
        },
        geometry=[clearing],
        crs="EPSG:26910",
    ).to_file(run_dir / "disturbances.gpkg", layer="objects", driver="GPKG")
    forest = np.ones((3, 12, 12), dtype=bool)
    forest[2, 3:6, 3:6] = False
    prob = np.zeros((3, 12, 12), dtype="float32")
    prob[0] = np.nan
    prob[2, 3:6, 3:6] = 0.9
    write_year_stack(run_dir / "layers" / "forest.tif", forest, GRID, YEARS)
    write_year_stack(run_dir / "detection" / "probability.tif", prob, GRID, YEARS)
    for name in ("chm_cover", "chm_height"):
        write_year_stack(run_dir / "layers" / f"{name}.tif", prob, GRID, YEARS)

    chips = tmp_path / "chips"
    chips.mkdir()
    rows = []
    for u, name in enumerate(["qa", "qb", "qc"]):
        for c in range(2):
            rows.append(
                write_chip(
                    chips,
                    f"{name}_c{c}",
                    loss_interval=1 if c == 0 else None,
                    seed=10 * u + c,
                )
            )
    # Two chips of the unit itself, on the lattice. The first holds the clearing.
    rows.append(
        write_chip(
            chips,
            f"{UNIT}_c0",
            loss_interval=1,
            seed=90,
            x0=CLEAR_PX[0] * 0.6,
            y1=CLEAR_PX[1] * 0.6,
        )
    )
    rows.append(
        write_chip(
            chips,
            f"{UNIT}_c1",
            loss_interval=None,
            seed=91,
            x0=(CLEAR_PX[0] + 128) * 0.6,
            y1=CLEAR_PX[1] * 0.6,
        )
    )
    pd.DataFrame(rows).to_parquet(chips / "chips.parquet", index=False)
    return {
        "data_root": data_root,
        "runs": tmp_path / "runs",
        "run_dir": run_dir,
        "chips": chips,
        "out": tmp_path / "out",
    }


def test_map_grid_snaps_to_the_cell_lattice() -> None:
    grid = P.unit_pred_grid(GRID.bounds, 26910, 16)
    assert (grid.col0, grid.row0) == (16000, 80000)
    assert (grid.cols, grid.rows) == (25, 25)
    assert grid.col0 % 16 == 0 and grid.row0 % 16 == 0
    left, bottom, right, top = grid.bounds
    unit = GRID.bounds
    assert left <= unit[0] and bottom <= unit[1] and right >= unit[2] and top >= unit[3]
    # The real origin of q4412328: 14 and 12 pixels off the lattice.
    real = P.unit_pred_grid((450162.0, 4930000.2, 460000.2, 4941636.0), 26910, 16)
    assert round(450162.0 / 0.6) - real.col0 == 14
    assert real.row0 - round(4941636.0 / 0.6) == 4
    rows, cols = grid.cell_window(
        (CLEAR_PX[0] * 0.6, CLEAR_PX[3] * 0.6, CLEAR_PX[2] * 0.6, CLEAR_PX[1] * 0.6)
    )
    assert (rows.start, rows.stop, cols.start, cols.stop) == (8, 14, 8, 14)


def test_blocks_tile_the_grid_with_a_halo() -> None:
    grid = P.unit_pred_grid(GRID.bounds, 26910, 16)
    blocks = P.blocks_of(grid, 8, 2)
    assert len(blocks) == 16
    covered = np.zeros((grid.rows, grid.cols), dtype=int)
    for block in blocks:
        covered[
            block.row : block.row + block.rows, block.col : block.col + block.cols
        ] += 1
        width = round((block.bounds[2] - block.bounds[0]) / 0.6)
        assert width == (block.cols + 4) * 16
    assert (covered == 1).all()
    assert len(P.central_blocks(blocks, grid, 3)) == 3


def test_unit_cell_layer_is_sampled_at_cell_centres() -> None:
    grid = P.unit_pred_grid(GRID.bounds, 26910, 16)
    layer = np.arange(144, dtype="float32").reshape(1, 12, 12)
    values, inside = P.cells_at_centres(layer, GRID.cell_transform, grid)
    assert values.shape == (1, 25, 25)
    # The centre of map cell (0, 0) lies 8 px inside the map, which is 4 px
    # north and 6 px west of the unit: outside. Cell (1, 1) is in unit cell 0.
    assert not inside[0, 0]
    assert inside[1, 1] and values[0, 1, 1] == 0
    assert values[0, 3, 3] == layer[0, 1, 1]
    assert inside.sum() == 24 * 24


class _LocalModel(torch.nn.Module):
    """A stand-in whose output depends on the cell alone, so blocks cannot matter."""

    cell_px = 16

    def encode(self, x):
        return [torch.nn.functional.avg_pool2d(x, 16)]

    def decode(self, fa, fb):
        return (fb[0] - fa[0]).mean(dim=1, keepdim=True) * 3.0


def test_blocked_map_equals_a_single_pass(unit, monkeypatch) -> None:
    from csdv_core.pairmodel import train as T

    ckpt = {"mean": [0.3] * 4, "std": [0.2] * 4}
    monkeypatch.setattr(
        T, "load_checkpoint", lambda path, device="cpu": (_LocalModel().eval(), ckpt)
    )
    maps = []
    for name, block_cells, halo in (("blocked", 8, 2), ("single", 64, 0)):
        result = P.predict_unit(
            Path("unused.pt"),
            unit["run_dir"],
            unit["data_root"],
            unit["out"] / name,
            block_cells=block_cells,
            halo_cells=halo,
            workers=0,
            device="cpu",
        )
        with rasterio.open(result.prob_path) as src:
            maps.append(src.read())
            assert src.count == 2
            assert src.descriptions == (
                "2012-07-01_2014-07-01",
                "2014-07-01_2016-07-01",
            )
            assert src.nodata == 255
            assert src.transform.a == pytest.approx(9.6)
        assert result.forest_path is not None and result.forest_path.is_file()
    assert np.array_equal(maps[0], maps[1])
    second = maps[0][1]
    assert (second[8:14, 8:14] > 90).all()  # the clearing is brighter in 2016
    assert (maps[0][0][8:14, 8:14] < 70).all()
    assert (second <= 100).all()  # the mosaics cover the whole map


class _LocalAgentModel(_LocalModel):
    """The same with an agent head: fire where the later date is brighter."""

    n_agents = 6

    def decode_agents(self, fa, fb):
        logits = self.decode(fa, fb)
        agents = torch.zeros(logits.shape[0], 6, *logits.shape[2:])
        agents[:, 2] = logits[:, 0]  # harvest (class 0) wins where it is negative
        return logits, agents


def test_flagged_agents_keeps_codes_at_or_above_the_threshold() -> None:
    codes = np.array([[1, 2, 3, 255, 5, 6]], dtype="uint8")
    prob = np.array([[44, 45, 255, 90, 100, 0]], dtype="uint8")
    flagged = P.flagged_agents(codes, prob, 0.45)
    assert flagged.dtype == np.uint8 and flagged.shape == codes.shape
    # 44 is below, 45 is exactly at the threshold, 255 in either array is no value.
    assert flagged.tolist() == [[0, 2, 0, 0, 5, 0]]
    # The threshold is rounded to a whole percent, as the map is.
    assert np.array_equal(P.flagged_agents(codes, prob, 0.449), flagged)
    assert P.flagged_agents(codes, prob, 0.46).tolist() == [[0, 0, 0, 0, 5, 0]]
    # At 0 every cell with imagery and a code counts, and no other.
    assert P.flagged_agents(codes, prob, 0.0).tolist() == [[1, 2, 0, 0, 5, 6]]
    assert P.flagged_agents(codes, prob, 1.0).tolist() == [[0, 0, 0, 0, 5, 0]]


def test_type_rasters_hold_the_agent_of_flagged_cells(unit, monkeypatch) -> None:
    from csdv_core.pairmodel import train as T
    from csdv_core.pairmodel.cli import pair

    ckpt = {"mean": [0.3] * 4, "std": [0.2] * 4}
    monkeypatch.setattr(
        T,
        "load_checkpoint",
        lambda path, device="cpu": (_LocalAgentModel().eval(), ckpt),
    )
    result = P.predict_unit(
        Path("unused.pt"),
        unit["run_dir"],
        unit["data_root"],
        unit["out"] / "default",
        block_cells=64,
        halo_cells=0,
        workers=0,
        device="cpu",
    )
    assert result.threshold == 0.5  # this checkpoint holds no threshold
    names = [
        f"type_{UNIT}_{a}_{b}.tif" for a, b in zip(DATES[:-1], DATES[1:], strict=True)
    ]
    assert [path.name for path in result.type_paths] == names
    assert result.agent_path is not None
    with rasterio.open(result.agent_path) as src:
        codes = src.read()
        assert src.count == 2 and src.nodata == 255
        with pytest.raises(ValueError):  # several bands, so no colour table
            src.colormap(1)
    with rasterio.open(result.prob_path) as src:
        prob = src.read()
    for k, path in enumerate(result.type_paths):
        with rasterio.open(path) as src:
            assert src.count == 1 and src.nodata == 0 and src.dtypes[0] == "uint8"
            table = src.colormap(1)
            kind = src.read(1)
        assert all(table[code] == colour for code, colour in P.AGENT_COLOURS.items())
        assert table[0][3] == 0
        assert np.array_equal(kind, P.flagged_agents(codes[k], prob[k], 0.5))

    # A stricter threshold from the command line leaves the clearing alone.
    checkpoint = unit["out"] / "stand_in.pt"
    checkpoint.write_bytes(b"")
    mapped = CliRunner().invoke(
        pair,
        [
            "predict",
            "--checkpoint",
            str(checkpoint),
            "--unit",
            UNIT,
            "--runs-dir",
            str(unit["runs"]),
            "--data-root",
            str(unit["data_root"]),
            "--out-dir",
            str(unit["out"] / "strict"),
            "--block-cells",
            "64",
            "--halo-cells",
            "0",
            "--workers",
            "0",
            "--device",
            "cpu",
            "--threshold",
            "0.9",
        ],
    )
    assert mapped.exit_code == 0, mapped.output
    assert "at or above 0.90: 2 type rasters" in mapped.output
    kinds = []
    for name in names:
        with rasterio.open(unit["out"] / "strict" / name) as src:
            kinds.append(src.read(1))
    assert not kinds[0].any()
    assert (kinds[1][8:14, 8:14] == 3).all()  # fire, where 2016 is brighter
    assert np.count_nonzero(kinds[1]) == 36


def test_label_image_is_transparent_where_no_chip_lies() -> None:
    from csdv_core.pairmodel import compare as C

    target = np.array([[0, 1], [C.IGNORE, C.NO_CHIP]], dtype="uint8")
    image = C._label_image(target)
    assert image.shape == (2, 2, 4) and image.dtype == np.uint8
    assert image[..., 3].tolist() == [[255, 255], [255, 0]]
    assert tuple(image[0, 1]) == (227, 73, 72, 255)
    assert len({tuple(pixel) for pixel in image.reshape(-1, 4)}) == 4


@pytest.fixture()
def layers():
    """Layers of a map of 25 by 25 cells with one chip and a loss of two agents."""
    from csdv_core.pairmodel import compare as C

    grid = P.unit_pred_grid(GRID.bounds, 26910, 16)
    rng = np.random.default_rng(0)
    prob = rng.integers(0, 40, (2, grid.rows, grid.cols), dtype=np.uint8)
    prob[1, 8:14, 8:14] = 95
    prob[:, 0] = 255
    codes = np.ones(prob.shape, dtype="uint8")
    codes[1, 8:14, 11:14] = 3
    codes[prob == 255] = 255
    target = np.full(prob.shape, C.NO_CHIP, dtype="uint8")
    target[:, 8:16, 8:16] = 0
    target[1, 8:14, 8:12] = 1
    left, _, _, top = grid.bounds
    chip = (left + 8 * 9.6, top - 16 * 9.6, left + 16 * 9.6, top - 8 * 9.6)
    return C._PanelLayers(
        names=[f"{a}_{b}" for a, b in zip(DATES[:-1], DATES[1:], strict=True)],
        prob=prob,
        di=prob.copy(),
        target=target,
        grid=grid,
        footprints=[chip],
        agent_codes=codes,
        threshold=0.5,
    )


def _draw(layers, k: int = 1):
    from csdv_core.pairmodel import compare as C

    rng = np.random.default_rng(1)
    images = [rng.integers(0, 255, (40, 40, 3), dtype=np.uint8) for _ in range(2)]
    return C._panel_figure("An event", layers.grid.bounds, k, images, layers, None)


def test_close_up_shows_the_agent_of_flagged_cells(layers) -> None:
    import dataclasses

    import matplotlib.pyplot as plt

    fig = _draw(layers)
    try:
        assert len(fig.axes) == 6
        assert fig.get_size_inches()[0] == pytest.approx(6 * 3.4)
        assert [ax.get_title() for ax in fig.axes[:4]] == [
            "2014-07-01",
            "2016-07-01",
            "Model (%)",
            "DI (%)",
        ]
        label_ax, type_ax = fig.axes[4], fig.axes[5]
        assert "hatched no label" in label_ax.get_title()
        hatched = [patch for patch in label_ax.patches if patch.get_hatch()]
        outlines = [patch for patch in label_ax.patches if not patch.get_fill()]
        assert len(hatched) == 1 and len(outlines) == 1
        left, bottom, right, top = layers.grid.bounds
        assert hatched[0].get_bbox().extents == pytest.approx(
            (left, bottom, right, top)
        )
        assert outlines[0].get_bbox().extents == pytest.approx(layers.footprints[0])
        # The label image lies over the hatching and is clear where no chip is.
        label_image = label_ax.images[0]
        assert label_image.get_zorder() > hatched[0].get_zorder()
        assert label_image.get_array()[0, 0, 3] == 0
        assert label_image.get_array()[8, 8, 3] == 255

        assert type_ax.get_title() == "Model type (loss at or above 50%)"
        shown = np.asarray(type_ax.images[0].get_array())
        assert tuple(shown[8, 8]) == P.AGENT_COLOURS[1]
        assert tuple(shown[8, 12]) == P.AGENT_COLOURS[3]
        assert shown[0, 0, 3] == 0  # no imagery
        assert shown[3, 3, 3] == 255 and tuple(shown[3, 3]) not in set(
            P.AGENT_COLOURS.values()
        )
        legend = type_ax.get_legend()
        assert [text.get_text() for text in legend.get_texts()] == ["harvest", "fire"]
        assert all(ax.get_legend() is None for ax in fig.axes[:5])
    finally:
        plt.close(fig)

    # Nothing is flagged in the first interval: the legend names every agent.
    fig = _draw(layers, k=0)
    try:
        texts = [t.get_text() for t in fig.axes[5].get_legend().get_texts()]
        assert texts == [
            "harvest",
            "wind",
            "fire",
            "biotic mortality",
            "conversion",
            "other",
        ]
    finally:
        plt.close(fig)

    # The threshold of the comparison decides what is flagged.
    fig = _draw(dataclasses.replace(layers, threshold=0.96))
    try:
        assert len(fig.axes[5].get_legend().get_texts()) == 6
        assert "96%" in fig.axes[5].get_title()
    finally:
        plt.close(fig)


def test_close_up_without_an_agent_raster_keeps_five_axes(layers) -> None:
    import dataclasses

    import matplotlib.pyplot as plt

    fig = _draw(dataclasses.replace(layers, agent_codes=None))
    try:
        assert len(fig.axes) == 5
        assert tuple(fig.get_size_inches()) == pytest.approx((17.0, 3.9))
        assert fig.axes[-1].get_title().startswith("Label")
        assert all(ax.get_legend() is None for ax in fig.axes)
        assert len(fig.axes[4].patches) == 2  # the hatching and one chip
    finally:
        plt.close(fig)


def test_a_quick_map_of_a_few_blocks_can_still_be_compared(unit, monkeypatch) -> None:
    from csdv_core.pairmodel import train as T
    from csdv_core.pairmodel.compare import compare_unit

    ckpt = {"mean": [0.3] * 4, "std": [0.2] * 4}
    monkeypatch.setattr(
        T, "load_checkpoint", lambda path, device="cpu": (_LocalModel().eval(), ckpt)
    )
    result = P.predict_unit(
        Path("unused.pt"),
        unit["run_dir"],
        unit["data_root"],
        unit["out"] / "quick",
        block_cells=4,
        halo_cells=1,
        workers=0,
        limit_blocks=1,
        device="cpu",
    )
    assert result.blocks == 1 and "quick run" in result.notes[0]
    with rasterio.open(result.prob_path) as src:
        prob = src.read()
    assert (prob <= 100).sum() == 2 * 16  # one block of 4 by 4 cells, two intervals
    # Most of the map is empty, so few or no label cells can be scored. The
    # tables are written all the same.
    compared = compare_unit(
        result.prob_path,
        unit["chips"],
        unit["run_dir"],
        unit["out"] / "quick_compare",
        data_root=unit["data_root"],
        n_objects=1,
        n_model_only=1,
        n_di_only=1,
    )
    assert (compared.out_dir / "chips_agreement.csv").is_file()
    assert (compared.out_dir / "overview.png").is_file()
    assert compared.summary["cells_scored"] <= 2 * 16


def test_train_predict_compare_from_the_command_line(unit) -> None:
    from csdv_core.pairmodel.cli import pair

    runner = CliRunner()
    run = unit["out"] / "run"
    common = ["--chips-dir", str(unit["chips"]), "--holdout-units", UNIT]
    dry = runner.invoke(pair, ["train", *common, "--val-units", "qb", "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "train: 2 units, 4 chips, 8 samples" in dry.output
    assert "Every chip file is there." in dry.output
    trained = runner.invoke(
        pair,
        [
            "train",
            *common,
            "--val-units",
            "qb",
            "--out-dir",
            str(run),
            "--encoder",
            "resnet18",
            "--encoder-weights",
            "none",
            "--crop-px",
            "96",
            "--batch-size",
            "4",
            "--epochs",
            "2",
            "--samples-per-epoch",
            "8",
            "--workers",
            "0",
            "--device",
            "cpu",
        ],
    )
    assert trained.exit_code == 0, trained.output
    for name in (
        "best.pt",
        "last.pt",
        "metrics.csv",
        "config.json",
        "curves.png",
        "agent_confusion_val.csv",
    ):
        assert (run / name).is_file(), name
    history = pd.read_csv(run / "metrics.csv")
    assert list(history["epoch"]) == [1, 2]
    assert "ap_qb" in history.columns
    assert history["val_agent_cells"].iloc[-1] > 0
    confusion = pd.read_csv(run / "agent_confusion_val.csv")
    assert list(confusion["label"])[0] == "harvest"
    assert confusion["cells"].sum() == history["val_agent_cells"].iloc[-1]
    record = json.loads((run / "config.json").read_text())
    assert record["split"]["test"] == [UNIT]
    assert UNIT not in record["split"]["train"]
    # A second call finds the run finished and trains nothing more.
    again = runner.invoke(
        pair,
        [
            "train",
            *common,
            "--val-units",
            "qb",
            "--out-dir",
            str(run),
            "--encoder",
            "resnet18",
            "--encoder-weights",
            "none",
            "--crop-px",
            "96",
            "--batch-size",
            "4",
            "--epochs",
            "2",
            "--samples-per-epoch",
            "8",
            "--workers",
            "0",
            "--device",
            "cpu",
        ],
    )
    assert again.exit_code == 0, again.output
    assert list(pd.read_csv(run / "metrics.csv")["epoch"]) == [1, 2]

    where = ["--unit", UNIT, "--runs-dir", str(unit["runs"])]
    mapped = runner.invoke(
        pair,
        [
            "predict",
            "--checkpoint",
            str(run / "best.pt"),
            *where,
            "--data-root",
            str(unit["data_root"]),
            "--out-dir",
            str(run / "map"),
            "--block-cells",
            "16",
            "--halo-cells",
            "2",
            "--workers",
            "0",
            "--device",
            "cpu",
        ],
    )
    assert mapped.exit_code == 0, mapped.output
    prob = run / "map" / f"prob_{UNIT}.tif"
    assert prob.is_file()
    with rasterio.open(run / "map" / f"agent_{UNIT}.tif") as src:
        codes = src.read()
        assert src.count == 2 and src.nodata == 255
    assert set(np.unique(codes)) <= {1, 2, 3, 4, 5, 6, 255}
    # One type raster per interval: the agent where the loss probability
    # reaches the checkpoint's threshold, 0 elsewhere, with a colour table.
    best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    cut = round(best["best_threshold"] * 100)
    assert f"at or above {best['best_threshold']:.2f}" in mapped.output
    with rasterio.open(prob) as src:
        percent = src.read()
        intervals = src.descriptions
    assert len(list((run / "map").glob("type_*.tif"))) == len(intervals) == 2
    for k, interval in enumerate(intervals):
        with rasterio.open(run / "map" / f"type_{UNIT}_{interval}.tif") as src:
            assert src.count == 1 and src.nodata == 0
            assert src.colormap(1)[3] == (227, 73, 72, 255)
            kind = src.read(1)
        flagged = (percent[k] != 255) & (percent[k] >= cut)
        assert np.array_equal(kind > 0, flagged)
        assert np.array_equal(kind[flagged], codes[k][flagged])

    compared = runner.invoke(
        pair,
        [
            "compare",
            "--prob",
            str(prob),
            "--chips-dir",
            str(unit["chips"]),
            *where,
            "--checkpoint",
            str(run / "best.pt"),
            "--data-root",
            str(unit["data_root"]),
            "--out-dir",
            str(run / "compare"),
            "--panels",
            "2",
        ],
    )
    assert compared.exit_code == 0, compared.output
    out = run / "compare"
    for name in (
        "chips_agreement.csv",
        "di_objects.csv",
        "model_only.csv",
        "overview.png",
        "compare.json",
    ):
        assert (out / name).is_file(), name
    scores = pd.read_csv(out / "chips_agreement.csv")
    assert set(scores["source"]) == {"model", "di"}
    assert set(scores["interval"]) == {
        "2012-07-01_2014-07-01",
        "2014-07-01_2016-07-01",
        "all",
    }
    # Two chips of 8 by 8 cells, two intervals. The DI is right on its own cells.
    total = scores[(scores["interval"] == "all") & (scores["source"] == "di")].iloc[0]
    assert 0 < total["loss_cells"] <= 32
    assert total["cells"] <= 2 * 2 * 64
    assert total["average_precision"] == pytest.approx(1.0)
    objects = pd.read_csv(out / "di_objects.csv")
    assert len(objects) == 1 and objects.iloc[0]["cells"] >= 36
    summary = json.loads((out / "compare.json").read_text())
    assert summary["chips"] == 2
    # The unit's only loss is a harvest: every agent cell of the label is in
    # the harvest row, whatever the barely trained model names.
    agents = pd.read_csv(out / "agent_confusion.csv")
    assert agents["cells"].sum() == summary["agents"]["cells"] > 0
    assert agents.loc[agents["label"] == "harvest", "cells"].iloc[0] > 0
    assert agents.loc[agents["label"] != "harvest", "cells"].sum() == 0
    assert list((out / "events").glob("*.png"))
