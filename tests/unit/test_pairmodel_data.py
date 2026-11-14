"""Pair samples from synthetic chips: index, splits, weights, targets."""

from __future__ import annotations

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
) -> dict:
    """Write the three files a loader reads of one chip, and return its table row.

    The chip is forest (dark, textured) on every date. In ``loss_interval``
    the left half is cleared (bright) and labelled loss, with an ignore band
    two pixels wide along its edge.
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
    )
    assert D.pair_target(arrays).tolist() == [[0, 1, 1, 255]]
    # Reversed: nothing is loss, and the pixel that gained forest is ignore.
    assert D.pair_target(arrays, reverse=True).tolist() == [[255, 0, 0, 255]]


def test_validation_sample_is_the_whole_chip(chips: Path) -> None:
    index = D.load_index(chips)
    row = index[(index["chip_id"] == "qa_c0") & (index["k"] == 1)].iloc[0]
    mean, std = D.band_stats(chips, index["chip_id"].unique())
    assert mean.shape == (4,) and (std > 0).all()
    a, b, target = D.build_sample(
        chips, row, cell_px=16, pooling=load_label().pooling, mean=mean, std=std
    )
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
    _, b, target = D.build_sample(
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
    _, _, target = D.build_sample(
        chips, row, rng=np.random.default_rng(3), **reversed_args
    )
    assert not (target == 1).any()
