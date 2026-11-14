"""csdv_core.pairmodel.data: pair samples from labelled chips.

Inputs: a chip folder of a label run (``chips.parquet`` and, per chip,
``_naip.tif``, ``_change.tif`` and ``_aux.tif``).
Outputs: an index with one row per chip and interval, splits by unit,
sampling weights, and samples as arrays (date A, date B, target on cells).

Everything up to :class:`PairDataset` needs NumPy and rasterio only, so the
samples can be drawn and looked at on a machine without PyTorch.

A sample is one chip in one interval: the image at the earlier date, the
image at the later date, and the change label of that interval pooled to
the model's output cells (0 background, 1 loss, 255 ignore).
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from csdv_core.config._models import LabelPooling
from csdv_core.label.pooling import pool_agent, pool_change
from csdv_core.label.verdicts import AGENT_CODES

logger = logging.getLogger(__name__)

IGNORE = 255
#: Pixel size of the chips, metres.
GSD = 0.6
#: A cell is ignore when at least this share of it has no imagery on a date.
NO_IMAGERY_SHARE = 0.5
#: Agent groups in the order of the model's classes. Class ``i`` is code
#: ``i + 1`` of the agent raster.
AGENT_NAMES: tuple[str, ...] = tuple(sorted(AGENT_CODES, key=AGENT_CODES.__getitem__))

__all__ = [
    "AGENT_NAMES",
    "IGNORE",
    "Augment",
    "PairArrays",
    "PairDataset",
    "Split",
    "band_stats",
    "build_sample",
    "load_index",
    "make_split",
    "missing_files",
    "pair_target",
    "read_pair",
    "sample_weights",
    "unit_folds",
]


@dataclass(frozen=True)
class Split:
    """Units of each part of a split. A unit is never in two parts."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]


@dataclass(frozen=True)
class Augment:
    """Random changes applied to a training sample.

    Attributes:
        crop_px: Side of the random crop. A multiple of the cell size.
        shift_px: Largest offset of date B against date A, pixels. Dates of
            NAIP are misregistered by a few metres.
        shift_prob: Probability of an offset.
        flip: Random flips and quarter turns.
        blur_prob: Probability of softening a date whose native resolution
            is 0.6 m or finer, so that sharpness does not tell the dates
            apart (older dates are 1 m).
        blur_sigma: Range of the Gaussian sigma of that blur, pixels.
        jitter: Random gain, gamma and offset per band, drawn separately for
            each date (NAIP radiometry is not calibrated between years).
        reverse_prob: Probability of feeding the pair in reverse order with
            a target of background.
    """

    crop_px: int = 384
    shift_px: int = 4
    shift_prob: float = 0.5
    flip: bool = True
    blur_prob: float = 0.5
    blur_sigma: tuple[float, float] = (0.3, 0.9)
    jitter: bool = True
    reverse_prob: float = 0.2


@dataclass
class PairArrays:
    """One chip in one interval, as read from disk.

    Attributes:
        a: Earlier date, ``(4, rows, cols)`` uint8.
        b: Later date, the same shape.
        change: Change label of the interval at 0.6 m (0, 1, 2 or 255).
        forest_a: Forest mask at the earlier date, bool.
        forest_b: Forest mask at the later date, bool.
        agent: Agent code of the interval at 0.6 m (1 to 6, 255 where there
            is no agreed loss or the two models named different agents).
    """

    a: np.ndarray
    b: np.ndarray
    change: np.ndarray
    forest_a: np.ndarray
    forest_b: np.ndarray
    agent: np.ndarray


def _json_list(value: Any) -> list:
    if isinstance(value, str):
        return json.loads(value)
    return list(value) if value is not None else []


def load_index(chips_dir: Path, *, min_valid: float = 0.05) -> pd.DataFrame:
    """One row per chip and interval, from ``chips.parquet``.

    Args:
        chips_dir: Chip folder of a label run.
        min_valid: Samples with a smaller labelled share are left out.

    Returns:
        Columns ``chip_id``, ``unit``, ``window_id``, ``k`` (interval index),
        ``n_dates``, ``date_a``, ``date_b``, ``native_a``, ``native_b``
        (metres, NaN when unknown), ``chip_px`` (side of the chip),
        ``loss_share`` and ``valid_share`` (of all pixels of the chip).

    Raises:
        ValueError: If a chip is listed twice.
    """
    table = pd.read_parquet(Path(chips_dir) / "chips.parquet")
    if table["chip_id"].duplicated().any():
        raise ValueError(f"{chips_dir}/chips.parquet lists a chip twice")
    rows = []
    for chip in table.itertuples():
        dates = _json_list(chip.dates)
        native = _json_list(chip.native_m)
        background = _json_list(chip.background)
        # With one loss class every loss pixel is counted under "partial".
        loss = [
            p + f
            for p, f in zip(
                _json_list(chip.partial), _json_list(chip.full), strict=True
            )
        ]
        for k in range(len(dates) - 1):
            valid = background[k] + loss[k]
            rows.append(
                {
                    "chip_id": chip.chip_id,
                    "unit": chip.unit,
                    "window_id": chip.window_id,
                    "k": k,
                    "n_dates": len(dates),
                    "date_a": dates[k],
                    "date_b": dates[k + 1],
                    "native_a": _native(native, k),
                    "native_b": _native(native, k + 1),
                    "chip_px": math.isqrt(int(chip.pixels)),
                    "loss_share": loss[k] / chip.pixels,
                    "valid_share": valid / chip.pixels,
                }
            )
    index = pd.DataFrame(rows)
    kept = index[index["valid_share"] >= min_valid].reset_index(drop=True)
    logger.info(
        "%d samples in %d chips of %d units (%d under %.0f%% labelled, left out)",
        len(kept),
        kept["chip_id"].nunique(),
        kept["unit"].nunique(),
        len(index) - len(kept),
        100 * min_valid,
    )
    return kept


def _native(native: list, k: int) -> float:
    if k < len(native) and native[k] is not None:
        return float(native[k])
    return float("nan")


def missing_files(chips_dir: Path, chip_ids: Sequence[str]) -> list[str]:
    """Chip files a loader needs and cannot find."""
    chips_dir = Path(chips_dir)
    missing = []
    for chip_id in chip_ids:
        for part in ("naip", "change", "aux", "agent"):
            if not (chips_dir / f"{chip_id}_{part}.tif").is_file():
                missing.append(f"{chip_id}_{part}.tif")
    return missing


def unit_folds(units: Sequence[str], k: int, seed: int = 0) -> list[list[str]]:
    """Units dealt into ``k`` folds in a seeded random order."""
    order = sorted(set(units))
    np.random.default_rng(seed).shuffle(order)
    return [sorted(order[i::k]) for i in range(k)]


def make_split(
    units: Sequence[str],
    *,
    holdout: Sequence[str] = (),
    val_units: Sequence[str] | None = None,
    fold: int | None = None,
    n_folds: int = 5,
    seed: int = 0,
) -> Split:
    """Split units into training, validation and test.

    Args:
        units: Every unit of the index.
        holdout: Test units, never trained or validated on.
        val_units: Validation units. Ignored when ``fold`` is given.
        fold: Use fold ``fold`` of ``n_folds`` grouped folds of the units
            outside ``holdout`` as validation.

    Raises:
        ValueError: If a named unit is unknown, or a part would be empty.
    """
    known = sorted(set(units))
    unknown = sorted((set(holdout) | set(val_units or ())) - set(known))
    if unknown:
        raise ValueError(f"Units not in the chips: {', '.join(unknown)}")
    rest = [u for u in known if u not in set(holdout)]
    if fold is not None:
        val = unit_folds(rest, n_folds, seed)[fold]
    else:
        val = sorted(set(val_units or ()) - set(holdout))
    train = [u for u in rest if u not in set(val)]
    if not train or not val:
        raise ValueError("The split needs at least one training and validation unit")
    return Split(tuple(train), tuple(val), tuple(sorted(holdout)))


def sample_weights(
    index: pd.DataFrame,
    *,
    loss_draws: float = 0.5,
    unit_power: float = 0.5,
    min_loss: float = 0.01,
) -> np.ndarray:
    """Sampling weight of every row of ``index``.

    Samples with loss and samples without are two groups, and ``loss_draws``
    of all draws go to the first. Inside a group a unit with ``n`` samples
    gets a total weight of ``n ** unit_power``, so a unit with very many
    samples (a burn) is drawn more often than a small one, and less than
    its count.

    Args:
        index: From :func:`load_index`.
        loss_draws: Share of draws that are samples with loss.
        unit_power: 1 draws units by their sample count, 0 draws them equally.
        min_loss: Smallest loss share of a chip that counts as "with loss".
    """
    has_loss = (index["loss_share"] >= min_loss).to_numpy()
    weights = np.zeros(len(index), dtype="float64")
    shares = {True: loss_draws, False: 1.0 - loss_draws}
    if not has_loss.any() or has_loss.all():
        shares = {True: 1.0, False: 1.0}
    for group, share in shares.items():
        rows = has_loss == group
        if not rows.any():
            continue
        counts = index.loc[rows, "unit"].map(index.loc[rows, "unit"].value_counts())
        raw = counts.to_numpy(dtype="float64") ** (unit_power - 1.0)
        weights[rows] = share * raw / raw.sum()
    return weights


def read_pair(
    chips_dir: Path,
    chip_id: str,
    k: int,
    n_dates: int,
    *,
    window_a: tuple[int, int, int] | None = None,
    window_b: tuple[int, int, int] | None = None,
) -> PairArrays:
    """Read interval ``k`` of a chip.

    Args:
        n_dates: Number of dates of the chip.
        window_a: ``(row, col, size)`` of a square crop of date A and of the
            labels. Default: the whole chip.
        window_b: The same for date B. Default: ``window_a``.

    Raises:
        ValueError: If the files do not hold ``n_dates`` dates.
    """
    import rasterio
    from rasterio.windows import Window

    def win(spec: tuple[int, int, int] | None) -> Window | None:
        return None if spec is None else Window(spec[1], spec[0], spec[2], spec[2])

    stem = Path(chips_dir) / chip_id
    with rasterio.open(f"{stem}_naip.tif") as src:
        if src.count != 4 * n_dates:
            raise ValueError(
                f"{chip_id}_naip.tif has {src.count} bands, expected {4 * n_dates}"
            )
        a = src.read([4 * k + i for i in (1, 2, 3, 4)], window=win(window_a))
        b = src.read(
            [4 * (k + 1) + i for i in (1, 2, 3, 4)], window=win(window_b or window_a)
        )
    with rasterio.open(f"{stem}_change.tif") as src:
        if src.count != n_dates - 1:
            raise ValueError(
                f"{chip_id}_change.tif has {src.count} bands, expected {n_dates - 1}"
            )
        change = src.read(k + 1, window=win(window_a))
    with rasterio.open(f"{stem}_aux.tif") as src:
        first = n_dates - 1
        forest = src.read([first + k + 1, first + k + 2], window=win(window_a))
    with rasterio.open(f"{stem}_agent.tif") as src:
        agent = src.read(k + 1, window=win(window_a))
    return PairArrays(a, b, change, forest[0] == 1, forest[1] == 1, agent)


def pair_target(arrays: PairArrays, *, reverse: bool = False) -> np.ndarray:
    """The 0.6 m target of a pair: 0 background, 1 loss, 255 ignore.

    Forward, it is the change label with both loss codes as 1. Reversed (the
    later date fed first), nothing is loss: labelled pixels are background,
    unlabelled ones stay unlabelled, and pixels that gained forest between
    the dates are unlabelled, because fed backwards they look like loss.
    """
    change = arrays.change
    target = np.where(change == IGNORE, IGNORE, (change > 0).astype("uint8"))
    target = target.astype("uint8")
    if reverse:
        target[target == 1] = 0
        target[~arrays.forest_a & arrays.forest_b] = IGNORE
    return target


def _pool_valid(image: np.ndarray, cell_px: int) -> np.ndarray:
    """Share of each cell that has imagery (any band above 0)."""
    valid = image.any(axis=0)
    rows, cols = valid.shape[0] // cell_px, valid.shape[1] // cell_px
    blocks = valid[: rows * cell_px, : cols * cell_px].reshape(
        rows, cell_px, cols, cell_px
    )
    return blocks.mean(axis=(1, 3))


def pool_target(
    target: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    cell_px: int,
    pooling: LabelPooling,
) -> np.ndarray:
    """Pool a 0.6 m target to cells, with cells lacking imagery as ignore."""
    pooled = pool_change(target, cell_px, pooling)
    pooled[pooled == 2] = 1
    for image in (a, b):
        pooled[_pool_valid(image, cell_px) <= 1.0 - NO_IMAGERY_SHARE] = IGNORE
    return pooled


def pool_agents(agent: np.ndarray, target: np.ndarray, cell_px: int) -> np.ndarray:
    """Agent class of each loss cell: 0 to 5, and 255 where there is none.

    Args:
        agent: Agent codes at 0.6 m (1 to 6, 255 none).
        target: The pooled target of the same sample (0, 1, 255). Only its
            loss cells get a class, so a reversed pair has none.
        cell_px: Pixels per cell side.

    Returns:
        The index into :data:`AGENT_NAMES` of the most common agent among
        the loss pixels of each loss cell. 255 for every other cell, and for
        loss cells on which the two vision models named different agents.
    """
    codes = pool_agent(agent, target, cell_px)
    out = np.full(codes.shape, IGNORE, dtype="uint8")
    known = (codes >= 1) & (codes <= len(AGENT_NAMES))
    out[known] = codes[known] - 1
    return out


def _jitter(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random gain, gamma and offset per band of a ``(4, r, c)`` image in 0..1."""
    # One draw for the whole image (exposure) and a smaller one per band
    # (colour balance), which is how NAIP years differ from each other.
    gain = rng.uniform(0.8, 1.2) * rng.uniform(0.93, 1.07, size=(4, 1, 1))
    gamma = rng.uniform(0.8, 1.25) * rng.uniform(0.95, 1.05, size=(4, 1, 1))
    bias = rng.uniform(-0.03, 0.03, size=(4, 1, 1))
    out = gain.astype("float32") * image ** gamma.astype("float32")
    return np.clip(out + bias.astype("float32"), 0.0, 1.0)


def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(image, sigma=(0.0, sigma, sigma), mode="nearest")


def _prepare(
    image: np.ndarray,
    native_m: float,
    aug: Augment | None,
    rng: np.random.Generator | None,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """A uint8 image as the model reads it: float32, normalised, 0 where empty."""
    empty = ~image.any(axis=0)
    out = image.astype("float32") / 255.0
    if aug is not None and rng is not None:
        fine = not np.isfinite(native_m) or native_m <= GSD + 1e-6
        if fine and rng.random() < aug.blur_prob:
            out = _blur(out, float(rng.uniform(*aug.blur_sigma)))
        if aug.jitter:
            out = _jitter(out, rng)
    out = (out - mean.reshape(4, 1, 1)) / std.reshape(4, 1, 1)
    out[:, empty] = 0.0
    return out.astype("float32")


def normalise(image: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """A uint8 ``(4, r, c)`` image as the model reads it, without augmentation."""
    return _prepare(image, float("nan"), None, None, mean, std)


def build_sample(
    chips_dir: Path,
    row: Any,
    *,
    cell_px: int,
    pooling: LabelPooling,
    mean: np.ndarray,
    std: np.ndarray,
    aug: Augment | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One sample of the index, ready for the model.

    Args:
        row: A row of :func:`load_index` (``chip_id``, ``k``, ``n_dates``,
            ``chip_px``, ``native_a``, ``native_b``).
        cell_px: Pixels per output cell side (16 for 9.6 m).
        pooling: Pooling thresholds of the label configuration.
        mean: Per-band mean of the digital numbers divided by 255.
        std: Per-band standard deviation, the same scale.
        aug: Augmentation. None for validation: the whole chip, as it is.
        rng: Random generator, required with ``aug``.

    Returns:
        ``(a, b, target, agent)``: two float32 ``(4, rows, cols)`` images,
        the uint8 target on cells (0 background, 1 loss, 255 ignore) and the
        uint8 agent class of the loss cells (:func:`pool_agents`).
    """
    window_a = window_b = None
    reverse = False
    if aug is not None:
        if rng is None:
            raise ValueError("Augmentation needs a random generator")
        chip_px = int(row.chip_px)
        size = min(aug.crop_px, chip_px)
        room = chip_px - size
        shift = aug.shift_px if rng.random() < aug.shift_prob else 0
        shift = min(shift, room // 2)
        row0 = int(rng.integers(shift, room - shift + 1))
        col0 = int(rng.integers(shift, room - shift + 1))
        window_a = (row0, col0, size)
        if shift:
            window_b = (
                row0 + int(rng.integers(-shift, shift + 1)),
                col0 + int(rng.integers(-shift, shift + 1)),
                size,
            )
        reverse = rng.random() < aug.reverse_prob
    arrays = read_pair(
        chips_dir,
        row.chip_id,
        int(row.k),
        int(row.n_dates),
        window_a=window_a,
        window_b=window_b,
    )
    target = pool_target(
        pair_target(arrays, reverse=reverse), arrays.a, arrays.b, cell_px, pooling
    )
    agent = pool_agents(arrays.agent, target, cell_px)
    first, second = arrays.a, arrays.b
    native_first, native_second = float(row.native_a), float(row.native_b)
    if reverse:
        first, second = second, first
        native_first, native_second = native_second, native_first
    a = _prepare(first, native_first, aug, rng, mean, std)
    b = _prepare(second, native_second, aug, rng, mean, std)
    if aug is not None and aug.flip and rng is not None:
        turns = int(rng.integers(0, 4))
        mirror = bool(rng.integers(0, 2))
        a, b, target, agent = (_turn(x, turns, mirror) for x in (a, b, target, agent))
    return a, b, target, agent


def _turn(array: np.ndarray, turns: int, mirror: bool) -> np.ndarray:
    out = np.rot90(array, turns, axes=(-2, -1))
    if mirror:
        out = out[..., ::-1]
    return np.ascontiguousarray(out)


def band_stats(
    chips_dir: Path, chip_ids: Sequence[str], *, n: int = 200, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation per band of ``n`` random chips.

    Values are digital numbers divided by 255, over every date and every
    pixel with imagery.
    """
    import rasterio

    ids = sorted(set(chip_ids))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(ids, size=min(n, len(ids)), replace=False)
    total = np.zeros(4)
    squares = np.zeros(4)
    count = 0
    for chip_id in chosen:
        with rasterio.open(Path(chips_dir) / f"{chip_id}_naip.tif") as src:
            data = src.read().reshape(-1, 4, src.height, src.width)
        # Every fourth pixel is enough for a mean and keeps this quick.
        data = data[:, :, ::4, ::4].astype("float64") / 255.0
        valid = data.any(axis=1)
        for band in range(4):
            values = data[:, band][valid]
            total[band] += values.sum()
            squares[band] += (values**2).sum()
        count += int(valid.sum())
    if count == 0:
        raise ValueError("The chips hold no imagery")
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean**2, 1e-6))
    return mean.astype("float32"), std.astype("float32")


class PairDataset:
    """Samples of an index for a PyTorch ``DataLoader``.

    Each item is ``(a, b, target, agent, unit_index)``. Training items are drawn
    with augmentation from a generator seeded per worker. Validation items
    are whole chips without any random change.
    """

    def __init__(
        self,
        index: pd.DataFrame,
        chips_dir: Path,
        *,
        cell_px: int,
        pooling: LabelPooling,
        mean: np.ndarray,
        std: np.ndarray,
        aug: Augment | None = None,
        units: Sequence[str] = (),
        seed: int = 0,
    ) -> None:
        self.index = index.reset_index(drop=True)
        self.chips_dir = Path(chips_dir)
        self.cell_px = cell_px
        self.pooling = pooling
        self.mean = np.asarray(mean, dtype="float32")
        self.std = np.asarray(std, dtype="float32")
        self.aug = aug
        self.units = list(units) or sorted(self.index["unit"].unique())
        self.seed = seed
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.index)

    def _generator(self) -> np.random.Generator:
        if self._rng is None:
            import torch

            info = torch.utils.data.get_worker_info()
            self._rng = np.random.default_rng(
                info.seed if info is not None else self.seed
            )
        return self._rng

    def __getitem__(self, item: int) -> tuple[Any, Any, Any, Any, int]:
        import torch

        row = self.index.iloc[item]
        a, b, target, agent = build_sample(
            self.chips_dir,
            row,
            cell_px=self.cell_px,
            pooling=self.pooling,
            mean=self.mean,
            std=self.std,
            aug=self.aug,
            rng=self._generator() if self.aug is not None else None,
        )
        return (
            torch.from_numpy(a),
            torch.from_numpy(b),
            torch.from_numpy(target),
            torch.from_numpy(agent),
            self.units.index(row.unit),
        )
