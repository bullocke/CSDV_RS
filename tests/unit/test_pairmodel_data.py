"""Pair samples from synthetic chips: index, splits, weights, targets."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from csdv_core.config import load_label
from csdv_core.pairmodel import data as D

CHIP_PX = 128
DATES = ["2012-07-01", "2014-07-01", "2016-07-01"]


def write_chip(
    folder: Path,
    chip_id: str,
    *,
    loss_interval: int | None = 1,
    seed: int = 0,
    chip_px: int = CHIP_PX,
    dates: list[str] = DATES,
    x0: float = 0.0,
    y1: float = 0.0,
    agent_code: int = 1,
) -> dict:
    """Write the four files a loader reads of one chip, and return its table row.

    The chip is forest (dark, textured) on every date. In ``loss_interval``
    the left half is cleared (bright) and labelled loss of agent
    ``agent_code``, with an ignore band two pixels wide along its edge.
    """
    rng = np.random.default_rng(seed)
    n = len(dates)
    half = chip_px // 2
    naip = rng.integers(40, 90, (n, 4, chip_px, chip_px), dtype=np.uint8)
    change = np.zeros((n - 1, chip_px, chip_px), dtype=np.uint8)
    forest = np.ones((n, chip_px, chip_px), dtype=np.uint8)
    if loss_interval is not None:
        naip[loss_interval + 1 :, :, :, :half] = rng.integers(
            170, 230, (n - loss_interval - 1, 4, chip_px, half), dtype=np.uint8
        )
        change[loss_interval, :, :half] = 1
        change[loss_interval, :, half - 2 : half + 2] = 255
        forest[loss_interval + 1 :, :, :half] = 0
    transform = from_origin(x0, y1, 0.6, 0.6)
    profile = {
        "driver": "GTiff",
        "height": chip_px,
        "width": chip_px,
        "dtype": "uint8",
        "crs": "EPSG:26910",
        "transform": transform,
    }
    with rasterio.open(
        folder / f"{chip_id}_naip.tif", "w", count=4 * n, **profile
    ) as dst:
        dst.write(naip.reshape(-1, chip_px, chip_px))
    with rasterio.open(
        folder / f"{chip_id}_change.tif", "w", count=n - 1, nodata=255, **profile
    ) as dst:
        dst.write(change)
    agent = np.where(change == 1, agent_code, 255).astype(np.uint8)
    with rasterio.open(
        folder / f"{chip_id}_agent.tif", "w", count=n - 1, nodata=255, **profile
    ) as dst:
        dst.write(agent)
    aux = np.concatenate([np.zeros_like(change), forest])
    if loss_interval is not None:
        aux[loss_interval, :, :half] = 90
    with rasterio.open(
        folder / f"{chip_id}_aux.tif", "w", count=2 * n - 1, **profile
    ) as dst:
        dst.write(aux)
    return {
        "chip_id": chip_id,
        "window_id": f"w_{chip_id}",
        "unit": chip_id.split("_")[0],
        "epsg": 26910,
        "bounds": json.dumps([x0, y1 - chip_px * 0.6, x0 + chip_px * 0.6, y1]),
        "dates": json.dumps(dates),
        "native_m": json.dumps([1.0, 0.6, 0.3][:n]),
        "background": json.dumps((change == 0).sum(axis=(1, 2)).tolist()),
        "partial": json.dumps((change == 1).sum(axis=(1, 2)).tolist()),
        "full": json.dumps([0] * (n - 1)),
        "ignore_share": float((change == 255).mean()),
        "pixels": chip_px * chip_px,
    }


@pytest.fixture()
def chips(tmp_path: Path) -> Path:
    """Eight chips of four units, half of them with a loss."""
    folder = tmp_path / "chips"
    folder.mkdir()
    rows = []
    for u, unit in enumerate(["qa", "qb", "qc", "qd"]):
        for c in range(2):
            rows.append(
                write_chip(
                    folder,
                    f"{unit}_c{c}",
                    loss_interval=1 if c == 0 else None,
                    seed=10 * u + c,
                )
            )
    pd.DataFrame(rows).to_parquet(folder / "chips.parquet", index=False)
    return folder


def test_index_has_one_row_per_chip_and_interval(chips: Path) -> None:
    index = D.load_index(chips)
    assert len(index) == 8 * 2
    loss = index[index["loss_share"] > 0]
    assert set(loss["k"]) == {1}
    assert loss["loss_share"].between(0.45, 0.5).all()
    first = index.iloc[0]
    assert (first.date_a, first.date_b) == ("2012-07-01", "2014-07-01")
    assert (first.native_a, first.native_b) == (1.0, 0.6)
    assert D.missing_files(chips, index["chip_id"].unique()) == []


def test_index_rejects_a_chip_listed_twice(chips: Path) -> None:
    table = pd.read_parquet(chips / "chips.parquet")
    pd.concat([table, table.iloc[:1]]).to_parquet(chips / "chips.parquet")
    with pytest.raises(ValueError, match="twice"):
        D.load_index(chips)


def test_split_parts_are_disjoint() -> None:
    units = ["qa", "qb", "qc", "qd", "qe"]
    split = D.make_split(units, holdout=["qe"], val_units=["qb"])
    assert split.test == ("qe",)
    assert split.val == ("qb",)
    assert set(split.train) == {"qa", "qc", "qd"}
    folds = D.unit_folds(units[:4], 2, seed=1)
    assert sorted(sum(folds, [])) == units[:4]
    by_fold = D.make_split(units, holdout=["qe"], fold=0, n_folds=2, seed=1)
    assert set(by_fold.val) == set(folds[0])
    assert not set(by_fold.train) & (set(by_fold.val) | set(by_fold.test))
    with pytest.raises(ValueError, match="not in the chips"):
        D.make_split(units, holdout=["qz"], val_units=["qb"])


def test_sample_weights_balance_loss_and_damp_large_units() -> None:
    index = pd.DataFrame(
        {
            "unit": ["big"] * 16 + ["small"] * 4 + ["big"] * 30 + ["small"] * 10,
            "loss_share": [0.2] * 20 + [0.0] * 40,
        }
    )
    weights = D.sample_weights(index, loss_draws=0.5, unit_power=0.5)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[:20].sum() == pytest.approx(0.5)
    # With a power of one half, 16 samples weigh twice as much as 4 in all.
    assert weights[:16].sum() / weights[16:20].sum() == pytest.approx(2.0)


def test_forward_and_reversed_targets() -> None:
    change = np.array([[0, 1, 2, 255]], dtype=np.uint8)
    arrays = D.PairArrays(
        a=np.ones((4, 1, 4), np.uint8),
        b=np.ones((4, 1, 4), np.uint8),
        change=change,
        forest_a=np.array([[False, True, True, True]]),
        forest_b=np.array([[True, False, False, True]]),
        agent=np.array([[255, 3, 3, 255]], dtype=np.uint8),
    )
    assert D.pair_target(arrays).tolist() == [[0, 1, 1, 255]]
    # Reversed: nothing is loss, and the pixel that gained forest is ignore.
    assert D.pair_target(arrays, reverse=True).tolist() == [[255, 0, 0, 255]]


def test_validation_sample_is_the_whole_chip(chips: Path) -> None:
    index = D.load_index(chips)
    row = index[(index["chip_id"] == "qa_c0") & (index["k"] == 1)].iloc[0]
    mean, std = D.band_stats(chips, index["chip_id"].unique())
    assert mean.shape == (4,) and (std > 0).all()
    a, b, target, agent = D.build_sample(
        chips, row, cell_px=16, pooling=load_label().pooling, mean=mean, std=std
    )
    # The synthetic loss is a harvest (code 1, class 0), on loss cells only.
    assert (agent[target == 1] == 0).all()
    assert (agent[target != 1] == 255).all()
    assert a.shape == b.shape == (4, CHIP_PX, CHIP_PX)
    assert a.dtype == np.float32
    assert target.shape == (CHIP_PX // 16, CHIP_PX // 16)
    # The left half is loss. The cells on its edge mix loss, ignore and
    # background, which the pooling rule leaves as loss or ignore, never
    # background.
    assert (target[:, :3] == 1).all()
    assert (target[:, 5:] == 0).all()
    assert set(np.unique(target[:, 3:5])) <= {0, 1, 255}
    assert b[:, :, : CHIP_PX // 2].mean() > a[:, :, : CHIP_PX // 2].mean() + 1


def test_cells_without_imagery_are_ignore(chips: Path) -> None:
    index = D.load_index(chips)
    row = index[(index["chip_id"] == "qa_c1") & (index["k"] == 0)].iloc[0]
    with rasterio.open(chips / "qa_c1_naip.tif", "r+") as dst:
        data = dst.read()
        data[4:8, :32, :] = 0  # the top two cell rows of the later date
        dst.write(data)
    _, b, target, _ = D.build_sample(
        chips,
        row,
        cell_px=16,
        pooling=load_label().pooling,
        mean=np.full(4, 0.3, np.float32),
        std=np.full(4, 0.2, np.float32),
    )
    assert (target[:2] == 255).all()
    assert (target[2:] == 0).all()
    assert (b[:, :32] == 0).all()


def test_augmented_sample_is_cropped_and_seeded(chips: Path) -> None:
    index = D.load_index(chips)
    row = index[(index["chip_id"] == "qa_c0") & (index["k"] == 1)].iloc[0]
    args = {
        "cell_px": 16,
        "pooling": load_label().pooling,
        "mean": np.full(4, 0.3, np.float32),
        "std": np.full(4, 0.2, np.float32),
        "aug": D.Augment(crop_px=96, reverse_prob=0.0),
    }
    one = D.build_sample(chips, row, rng=np.random.default_rng(3), **args)
    two = D.build_sample(chips, row, rng=np.random.default_rng(3), **args)
    assert one[0].shape == (4, 96, 96)
    assert one[2].shape == (6, 6)
    for x, y in zip(one, two, strict=True):
        assert np.array_equal(x, y)
    reversed_args = {**args, "aug": D.Augment(crop_px=96, reverse_prob=1.0)}
    _, _, target, agent = D.build_sample(
        chips, row, rng=np.random.default_rng(3), **reversed_args
    )
    assert not (target == 1).any()
    assert (agent == 255).all()


def _sample_args() -> dict:
    return {
        "cell_px": 16,
        "pooling": load_label().pooling,
        "mean": np.full(4, 0.3, np.float32),
        "std": np.full(4, 0.2, np.float32),
    }


def test_cached_samples_equal_the_ones_read_from_the_rasters(
    chips: Path, tmp_path: Path
) -> None:
    from csdv_core.pairmodel.cache import build_cache

    index = D.load_index(chips)
    cache = tmp_path / "cache"
    report = build_cache(chips, cache, index[["chip_id", "n_dates"]])
    assert (report.written, report.kept) == (8, 0)
    for window_a, window_b in ((None, None), ((8, 16, 96), (10, 13, 96))):
        plain = D.read_pair(chips, "qa_c0", 1, 3, window_a=window_a, window_b=window_b)
        cached = D.read_pair(
            chips, "qa_c0", 1, 3, window_a=window_a, window_b=window_b, cache_dir=cache
        )
        for name in ("a", "b", "change", "forest_a", "forest_b", "agent"):
            assert np.array_equal(getattr(plain, name), getattr(cached, name)), name
    row = index[(index["chip_id"] == "qa_c0") & (index["k"] == 1)].iloc[0]
    for aug in (None, D.Augment(crop_px=96)):
        one = D.build_sample(
            chips, row, aug=aug, rng=np.random.default_rng(5), **_sample_args()
        )
        two = D.build_sample(
            chips,
            row,
            aug=aug,
            rng=np.random.default_rng(5),
            cache_dir=cache,
            **_sample_args(),
        )
        for x, y in zip(one, two, strict=True):
            assert np.array_equal(x, y)
    with pytest.raises(ValueError, match="bands"):
        D.read_pair(chips, "qa_c0", 1, 4, cache_dir=cache)


def test_cache_is_kept_until_a_chip_changes(chips: Path, tmp_path: Path) -> None:
    from csdv_core.pairmodel.cache import build_cache, cache_path

    table = D.load_index(chips)[["chip_id", "n_dates"]]
    cache = tmp_path / "cache"
    build_cache(chips, cache, table)
    again = build_cache(chips, cache, table)
    assert (again.written, again.kept) == (0, 8)
    # A chip written again, a cached array cut short and one removed.
    write_chip(chips, "qa_c1", loss_interval=0, seed=77)
    short = cache_path(cache, "qb_c0")
    short.write_bytes(short.read_bytes()[:-10])
    cache_path(cache, "qc_c0").unlink()
    third = build_cache(chips, cache, table, workers=2)
    assert (third.written, third.kept) == (3, 5)
    fresh = D.read_pair(chips, "qa_c1", 0, 3, cache_dir=cache)
    assert (fresh.change[:, : CHIP_PX // 2 - 2] == 1).all()
    # A build for a few chips leaves the entries of the others valid.
    build_cache(chips, cache, table.head(2))
    assert build_cache(chips, cache, table).written == 0


def test_agent_cells_count_the_loss_of_each_agent(chips: Path, tmp_path: Path) -> None:
    write_chip(chips, "qe_c0", loss_interval=0, seed=5, agent_code=2)
    cells = D.agent_cells(chips, ["qa_c0", "qa_c1", "qe_c0"], cell_px=16)
    loss_px = CHIP_PX * (CHIP_PX // 2 - 2)
    assert list(cells.columns) == list(D.AGENT_NAMES)
    assert cells.loc["qa_c0", "harvest"] == pytest.approx(loss_px / 256)
    assert cells.loc["qa_c1"].sum() == 0
    assert cells.loc["qe_c0", "wind"] == pytest.approx(loss_px / 256)
    assert cells.loc["qe_c0", "harvest"] == 0
    with pytest.raises(FileNotFoundError):
        D.agent_cells(chips, ["nowhere"], cell_px=16)


def _lattice() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two units of 12 by 12 chips in windows of two, with three agents."""
    rng = np.random.default_rng(4)
    rows, cells = [], []
    for u, unit in enumerate(["qa", "qb"]):
        for ix in range(0, 12, 2):
            for iy in range(12):
                agent = D.AGENT_NAMES[int(rng.integers(0, 3))]
                for dx in (0, 1):
                    chip = f"{unit}_x{ix + dx}_y{iy}"
                    rows.append(
                        {
                            "chip_id": chip,
                            "window_id": f"{unit}_w{ix}_{iy}",
                            "unit": unit,
                            "epsg": 26910 + u,
                            "ix": 100 + ix + dx,
                            "iy": 500 + iy,
                        }
                    )
                    count = {name: 0.0 for name in D.AGENT_NAMES}
                    count[agent] = float(rng.integers(1, 50))
                    cells.append({"chip_id": chip, **count})
    return pd.DataFrame(rows), pd.DataFrame(cells).set_index("chip_id")


def test_blocks_keep_windows_whole_and_sides_apart() -> None:
    chips, cells = _lattice()
    sides = D.assign_blocks(chips, cells, block_chips=3, val_share=0.25, seed=0)
    assert sides.equals(
        D.assign_blocks(chips, cells, block_chips=3, val_share=0.25, seed=0)
    )
    assert set(sides) == {"train", "val", "dropped"}
    table = chips.set_index("chip_id").assign(side=sides)
    kept = table[table["side"] != "dropped"]
    assert (kept.groupby("window_id")["side"].nunique() == 1).all()
    val = {(r.epsg, r.ix, r.iy) for r in table[table["side"] == "val"].itertuples()}
    for r in table[table["side"] == "train"].itertuples():
        near = {
            (r.epsg, r.ix + dx, r.iy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
        }
        assert not near & val
    # Every agent that occurs, and every unit, is on both sides.
    by_side = cells.join(sides.rename("side")).groupby("side").sum()
    for agent in D.AGENT_NAMES[:3]:
        share = by_side.loc["val", agent] / cells[agent].sum()
        assert 0.1 < share < 0.45, (agent, share)
        assert by_side.loc["train", agent] > 0
    assert (
        table.groupby("unit")["side"].agg(lambda s: {"train", "val"} <= set(s))
    ).all()


@pytest.fixture()
def lattice_chips(tmp_path: Path) -> Path:
    """Three units of 4 by 4 chips on the lattice, in windows of two chips."""
    folder = tmp_path / "lattice"
    folder.mkdir()
    side = CHIP_PX * 0.6
    rows = []
    for u, unit in enumerate(["qa", "qb", "qz"]):
        for ix in range(4):
            for iy in range(4):
                row = write_chip(
                    folder,
                    f"{unit}_x{ix}y{iy}",
                    loss_interval=(ix + iy) % 2,
                    seed=100 * u + 4 * ix + iy,
                    x0=(10 * u + ix) * side,
                    y1=(iy + 1) * side,
                    agent_code=1 + (ix // 2 + iy) % 3,
                )
                row["window_id"] = f"{unit}_w{ix // 2}_{iy}"
                rows.append(row)
    pd.DataFrame(rows).to_parquet(folder / "chips.parquet", index=False)
    return folder


def test_blocks_split_plans_both_sides_inside_every_unit(lattice_chips: Path) -> None:
    from csdv_core.pairmodel import train as T

    cfg = T.TrainConfig(
        chips_dir=lattice_chips,
        out_dir=lattice_chips,
        holdout_units=("qz",),
        split="blocks",
        block_chips=2,
        val_share=0.25,
    )
    planned = T.plan(cfg)
    assert planned.split.train == planned.split.val == ("qa", "qb")
    assert planned.split.test == ("qz",)
    listed = T.split_chips(planned)
    chips = set(listed["train"]) | set(listed["val"]) | set(listed["dropped"])
    assert len(chips) == 32 and not any(c.startswith("qz") for c in chips)
    assert not set(listed["train"]) & set(listed["val"])
    assert set(planned.train_index["chip_id"]) == set(listed["train"])
    assert set(planned.val_index["chip_id"]) == set(listed["val"])
    for part in (planned.train_index, planned.val_index):
        assert set(part["unit"]) == {"qa", "qb"}
    text = T.describe(planned) + T.describe_agents(planned, lattice_chips, 16)
    assert "training and validation share the units" in text
    assert "val share" in text and "wind" in text
    # The unit split is unchanged and lists no chips.
    by_unit = T.plan(dataclasses.replace(cfg, split="units", val_units=("qb",)))
    assert T.split_chips(by_unit) is None
    assert set(by_unit.val_index["unit"]) == {"qb"}
    with pytest.raises(ValueError, match="blocks split"):
        T.plan(dataclasses.replace(cfg, val_units=("qb",)))


def test_blocks_and_cache_from_the_command_line(
    lattice_chips: Path, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from click.testing import CliRunner

    from csdv_core.pairmodel.cli import pair

    runner = CliRunner()
    common = ["train", "--chips-dir", str(lattice_chips), "--holdout-units", "qz"]
    blocks = ["--split", "blocks", "--block-chips", "2", "--val-share", "0.25"]
    dry = runner.invoke(pair, [*common, *blocks, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "Loss cells per agent" in dry.output
    assert "training and validation share the units" in dry.output
    clash = runner.invoke(pair, [*common, *blocks, "--val-units", "qb", "--dry-run"])
    assert clash.exit_code != 0
    assert "--val-units" in clash.output
    run, cache = tmp_path / "run", tmp_path / "cache"
    args = [
        *common,
        *blocks,
        "--out-dir",
        str(run),
        "--cache-dir",
        str(cache),
        "--encoder",
        "resnet18",
        "--encoder-weights",
        "none",
        "--crop-px",
        "96",
        "--batch-size",
        "4",
        "--epochs",
        "1",
        "--samples-per-epoch",
        "8",
        "--workers",
        "0",
        "--device",
        "cpu",
    ]
    trained = runner.invoke(pair, args)
    assert trained.exit_code == 0, trained.output
    record = json.loads((run / "config.json").read_text())
    assert record["split_mode"] == "blocks"
    listed = record["split_chips"]
    assert listed["train"] and listed["val"]
    assert not set(listed["train"]) & set(listed["val"])
    assert record["split"]["train"] == record["split"]["val"] == ["qa", "qb"]
    used = set(listed["train"]) | set(listed["val"])
    assert {p.stem for p in cache.glob("*.npy")} == used
    history = pd.read_csv(run / "metrics.csv")
    assert "val_seconds" in history.columns
    # Another split in the same run folder is refused, not mixed in.
    other = runner.invoke(pair, [*args[:-14], "--val-share", "0.5", *args[-14:]])
    assert other.exit_code != 0
    assert "split differs" in other.output
