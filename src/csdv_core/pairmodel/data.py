"""csdv_core.pairmodel.data: pair samples from labelled chips.

Inputs: a chip folder of a label run (``chips.parquet`` and, per chip,
``_naip.tif``, ``_change.tif``, ``_agent.tif`` and ``_aux.tif``).
Outputs: an index with one row per chip and interval, splits by unit or by
blocks of ground inside the units, sampling weights, and samples as arrays
(date A, date B, target on cells).

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
from collections.abc import Mapping, Sequence
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
#: Sites table of a chip folder (see :func:`load_sites`).
SITES_FILE = "sites.csv"
#: A cell is ignore when at least this share of it has no imagery on a date.
NO_IMAGERY_SHARE = 0.5
#: A third image is used when at least this share of the crop has imagery.
THIRD_IMAGERY_SHARE = 0.9
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
    "agent_cells",
    "assign_blocks",
    "band_stats",
    "build_sample",
    "build_triple",
    "SITES_FILE",
    "load_index",
    "load_sites",
    "make_split",
    "missing_files",
    "pair_target",
    "read_date",
    "read_pair",
    "role_units",
    "sample_weights",
    "tied_units",
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
        third_drop: For a model with three dates: probability of withholding
            the third image of a sample that has one, so that the model also
            learns to work from two (the newest interval has no third).
    """

    crop_px: int = 384
    shift_px: int = 4
    shift_prob: float = 0.5
    flip: bool = True
    blur_prob: float = 0.5
    blur_sigma: tuple[float, float] = (0.3, 0.9)
    jitter: bool = True
    reverse_prob: float = 0.2
    third_drop: float = 0.3


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


def load_sites(chips_dir: Path) -> pd.DataFrame | None:
    """The sites table of a chip folder, or None when it has none.

    ``sites.csv`` in a chip folder (written by ``csdv pair merge-chips`` from
    the ``sites.csv`` of each label run) has one row per unit with the
    optional columns ``role`` (``train`` or ``test``), ``holdout_with`` (a
    unit that saw the same event or flight) and whatever else the label run
    recorded (``target``, ``kind``).
    """
    path = Path(chips_dir) / SITES_FILE
    if not path.is_file():
        return None
    sites = pd.read_csv(path, dtype=str, keep_default_na=False, comment="#")
    if "unit" not in sites:
        raise ValueError(f"{path} needs the column unit")
    return sites


def role_units(sites: pd.DataFrame | None, roles: Sequence[str]) -> list[str]:
    """Units of the sites table with one of the roles."""
    if sites is None or "role" not in sites or not roles:
        return []
    return sorted(sites.loc[sites["role"].str.strip().isin(set(roles)), "unit"])


def tied_units(sites: pd.DataFrame | None, holdout: Sequence[str]) -> list[str]:
    """Units that have to be held out with ``holdout``.

    A unit is tied to another when its ``holdout_with`` names it: both saw
    one event or one flight, so a model tested on one must not train on the
    other. Several units may be named, separated by ``;`` or blanks.
    """
    if sites is None or "holdout_with" not in sites or not holdout:
        return []
    held = set(holdout)
    tied = [
        unit
        for unit, names in zip(sites["unit"], sites["holdout_with"], strict=True)
        if held & set(names.replace(";", " ").split()) and unit not in held
    ]
    return sorted(tied)


def load_index(chips_dir: Path, *, min_valid: float = 0.05) -> pd.DataFrame:
    """One row per chip and interval, from ``chips.parquet``.

    Args:
        chips_dir: Chip folder of a label run.
        min_valid: Samples with a smaller labelled share are left out.

    Returns:
        Columns ``chip_id``, ``unit``, ``window_id``, ``k`` (interval index),
        ``n_dates``, ``date_a``, ``date_b``, ``native_a``, ``native_b``,
        ``native_c`` (metres, NaN when unknown, ``native_c`` of the date
        after ``date_b``, NaN when there is none), ``chip_px`` (side of the chip),
        ``loss_share`` and ``valid_share`` (of all pixels of the chip),
        ``epsg`` and ``ix``, ``iy`` (the chip's column and row on the
        lattice of its UTM zone, counted in chips), and ``run`` (the label
        run of a merged chip folder, see :mod:`csdv_core.pairmodel.merge`,
        empty otherwise).

    Raises:
        ValueError: If a chip is listed twice.
    """
    table = pd.read_parquet(Path(chips_dir) / "chips.parquet")
    if table["chip_id"].duplicated().any():
        raise ValueError(f"{chips_dir}/chips.parquet lists a chip twice")
    has_run = "run" in table.columns
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
        chip_px = math.isqrt(int(chip.pixels))
        bounds = _json_list(chip.bounds)
        side = chip_px * GSD
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
                    "native_c": _native(native, k + 2),
                    "chip_px": chip_px,
                    "loss_share": loss[k] / chip.pixels,
                    "valid_share": valid / chip.pixels,
                    "epsg": int(chip.epsg),
                    "ix": round(bounds[0] / side),
                    "iy": round(bounds[1] / side),
                    "run": str(chip.run) if has_run else "",
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


def agent_cells(chips_dir: Path, chip_ids: Sequence[str], cell_px: int) -> pd.DataFrame:
    """Loss cells per agent of every chip, summed over its intervals.

    Counted on the 0.6 m agent rasters and divided by the pixels of a cell,
    which is within a few percent of the count on pooled cells and needs no
    imagery.

    Returns:
        One row per chip (index ``chip_id``) and one column per name of
        :data:`AGENT_NAMES`.

    Raises:
        FileNotFoundError: If an agent raster is missing.
    """
    import rasterio

    rows = []
    for chip_id in chip_ids:
        path = Path(chips_dir) / f"{chip_id}_agent.tif"
        if not path.is_file():
            raise FileNotFoundError(f"No agent raster {path}")
        with rasterio.open(path) as src:
            codes = src.read()
        counts = np.bincount(codes.ravel(), minlength=256)
        rows.append(counts[1 : len(AGENT_NAMES) + 1] / float(cell_px**2))
    return pd.DataFrame(
        np.array(rows).reshape(len(rows), len(AGENT_NAMES)),
        index=pd.Index(list(chip_ids), name="chip_id"),
        columns=list(AGENT_NAMES),
    )


def assign_blocks(
    chips: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    block_chips: int = 8,
    val_share: float = 0.2,
    seed: int = 0,
) -> pd.Series:
    """Give every chip a side, by blocks of ground inside the units.

    The lattice of each unit is cut into square blocks of ``block_chips``
    chips. A window belongs to the block of its lower-left chip, so its
    chips stay together. Whole blocks go to validation until validation
    holds about ``val_share`` of the loss cells of every agent and of the
    chips of every unit: blocks are taken in order of their largest share,
    and one joins validation when that brings the totals closer to the
    target. Training chips that touch a validation chip are dropped.

    Args:
        chips: One row per chip with ``chip_id``, ``window_id``, ``unit``,
            ``epsg``, ``ix`` and ``iy`` (see :func:`load_index`).
        cells: Loss cells per agent of every chip (:func:`agent_cells`).
        block_chips: Side of a block, in chips.
        val_share: Share aimed at for validation.
        seed: Seed of the order among blocks of equal size.

    Returns:
        ``train``, ``val`` or ``dropped`` per chip, indexed by ``chip_id``.
    """
    chips = chips.drop_duplicates("chip_id").reset_index(drop=True)
    corner = chips.groupby("window_id")[["ix", "iy"]].transform("min")
    block = (
        chips["unit"].astype(str)
        + ":"
        + (corner["ix"] // block_chips).astype(str)
        + ":"
        + (corner["iy"] // block_chips).astype(str)
    )
    counts = cells.reindex(chips["chip_id"]).fillna(0.0).to_numpy(dtype="float64")
    units = sorted(chips["unit"].unique())
    in_unit = np.stack([(chips["unit"] == u).to_numpy(dtype="float64") for u in units])
    per_chip = np.concatenate([counts, in_unit.T], axis=1)
    totals = per_chip.sum(axis=0)
    used = totals > 0
    names = sorted(block.unique())
    codes = block.map({name: i for i, name in enumerate(names)}).to_numpy()
    shares = np.zeros((len(names), per_chip.shape[1]))
    np.add.at(shares, codes, per_chip)
    shares[:, used] /= totals[used]
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(names)), -shares.max(axis=1)))
    target = np.where(used, val_share, 0.0)
    held = np.zeros(per_chip.shape[1])
    in_val = np.zeros(len(names), dtype=bool)
    for i in order:
        if ((held + shares[i] - target) ** 2).sum() < ((held - target) ** 2).sum():
            held += shares[i]
            in_val[i] = True
    side = np.where(in_val[codes], "val", "train").astype(object)
    val_cells = {
        (e, x, y)
        for e, x, y in zip(
            chips["epsg"][side == "val"],
            chips["ix"][side == "val"],
            chips["iy"][side == "val"],
            strict=True,
        )
    }
    for i, (e, x, y) in enumerate(
        zip(chips["epsg"], chips["ix"], chips["iy"], strict=True)
    ):
        if side[i] != "train":
            continue
        near = ((e, x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        if any(cell in val_cells for cell in near):
            side[i] = "dropped"
    return pd.Series(side, index=pd.Index(chips["chip_id"], name="chip_id"))


def sample_weights(
    index: pd.DataFrame,
    *,
    loss_draws: float = 0.5,
    unit_power: float = 0.5,
    min_loss: float = 0.01,
    run_draws: Mapping[str, float] | None = None,
    exclude_runs: Sequence[str] = (),
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
        run_draws: Share of draws for the samples without loss of named
            label runs (the ``run`` column of a merged chip folder), for
            example ``{"stable_forest": 0.15}``. Each named run is a group of
            its own, and its share is taken from the draws without loss.
            None or empty leaves the weights exactly as without it. A share
            of 0 gives the run's samples without loss no weight, and the
            others the weights they have without that run.
        exclude_runs: Label runs whose samples are never drawn, with loss or
            without. The other samples get the weights they have without
            those runs: the control of a model trained with them, on the
            same chip folder, split and band statistics.

    Raises:
        ValueError: If a named run has no sample without loss, the named
            shares leave nothing for the other samples without loss, or an
            excluded run is not in the index or leaves no sample.
    """
    if exclude_runs:
        runs = index["run"] if "run" in index else pd.Series("", index=index.index)
        absent = sorted(set(exclude_runs) - set(runs))
        if absent:
            raise ValueError(f"No sample from the label runs {absent}")
        keep = ~runs.isin(set(exclude_runs)).to_numpy()
        if not keep.any():
            raise ValueError("The excluded runs leave no sample to draw")
        weights = np.zeros(len(index), dtype="float64")
        weights[keep] = sample_weights(
            index[keep].reset_index(drop=True),
            loss_draws=loss_draws,
            unit_power=unit_power,
            min_loss=min_loss,
            run_draws={
                k: v for k, v in (run_draws or {}).items() if k not in exclude_runs
            },
        )
        return weights
    has_loss = (index["loss_share"] >= min_loss).to_numpy()
    weights = np.zeros(len(index), dtype="float64")
    groups: dict[str, tuple[np.ndarray, float]] = {
        "loss": (has_loss, loss_draws),
        "no_loss": (~has_loss, 1.0 - loss_draws),
    }
    if not has_loss.any() or has_loss.all():
        groups = {"loss": (has_loss, 1.0), "no_loss": (~has_loss, 1.0)}
    if run_draws:
        runs = index["run"].to_numpy() if "run" in index else np.full(len(index), "")
        rest_rows, rest_share = groups["no_loss"]
        for name, share in run_draws.items():
            rows = rest_rows & (runs == name)
            if not rows.any():
                raise ValueError(f"No sample without loss from the label run {name!r}")
            groups[f"run:{name}"] = (rows, share)
            rest_rows = rest_rows & ~rows
            rest_share -= share
        if rest_share <= 0:
            raise ValueError("The run shares leave nothing for the other samples")
        groups["no_loss"] = (rest_rows, rest_share)
    for rows, share in groups.values():
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
    cache_dir: Path | None = None,
) -> PairArrays:
    """Read interval ``k`` of a chip.

    Args:
        n_dates: Number of dates of the chip.
        window_a: ``(row, col, size)`` of a square crop of date A and of the
            labels. Default: the whole chip.
        window_b: The same for date B. Default: ``window_a``.
        cache_dir: Read from this decoded cache instead of the chip rasters
            (:mod:`csdv_core.pairmodel.cache`). The arrays are the same.

    Raises:
        ValueError: If the files do not hold ``n_dates`` dates.
    """
    if cache_dir is not None:
        from csdv_core.pairmodel.cache import read_cached

        a, b, change, forest, agent = read_cached(
            cache_dir, chip_id, k, n_dates, window_a, window_b
        )
        return PairArrays(a, b, change, forest[0] == 1, forest[1] == 1, agent)

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


def read_date(
    chips_dir: Path,
    chip_id: str,
    date: int,
    n_dates: int,
    *,
    window: tuple[int, int, int] | None = None,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """The four bands of one date of a chip, ``(4, rows, cols)`` uint8.

    Args:
        date: Index of the date, from 0.
        n_dates: Number of dates of the chip.
        window: ``(row, col, size)`` of a square crop. Default: the whole chip.
        cache_dir: Read from this decoded cache instead of the chip raster.

    Raises:
        ValueError: If the chip does not hold ``n_dates`` dates, or ``date``
            is not one of them.
    """
    if cache_dir is not None:
        from csdv_core.pairmodel.cache import read_cached_date

        return read_cached_date(cache_dir, chip_id, date, n_dates, window)
    if not 0 <= date < n_dates:
        raise ValueError(f"Chip {chip_id} has no date {date} of {n_dates}")

    import rasterio
    from rasterio.windows import Window

    with rasterio.open(Path(chips_dir) / f"{chip_id}_naip.tif") as src:
        if src.count != 4 * n_dates:
            raise ValueError(
                f"{chip_id}_naip.tif has {src.count} bands, expected {4 * n_dates}"
            )
        spec = None
        if window is not None:
            spec = Window(window[1], window[0], window[2], window[2])
        return src.read([4 * date + i for i in (1, 2, 3, 4)], window=spec)


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
    cache_dir: Path | None = None,
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
        cache_dir: Decoded cache to read from, see :func:`read_pair`.

    Returns:
        ``(a, b, target, agent)``: two float32 ``(4, rows, cols)`` images,
        the uint8 target on cells (0 background, 1 loss, 255 ignore) and the
        uint8 agent class of the loss cells (:func:`pool_agents`).
    """
    a, b, target, agent, _, _ = _build(
        chips_dir,
        row,
        cell_px=cell_px,
        pooling=pooling,
        mean=mean,
        std=std,
        aug=aug,
        rng=rng,
        cache_dir=cache_dir,
        third=False,
    )
    return a, b, target, agent


def build_triple(
    chips_dir: Path,
    row: Any,
    *,
    cell_px: int,
    pooling: LabelPooling,
    mean: np.ndarray,
    std: np.ndarray,
    aug: Augment | None = None,
    rng: np.random.Generator | None = None,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """One sample with the image after the pair, for a model with three dates.

    The pair and its target are those of :func:`build_sample`. The third
    image is the chip's next date. It is absent for the last interval of a
    chip, for a pair fed in reverse, where it covers less than
    :data:`THIRD_IMAGERY_SHARE` of the crop, and in training for a share
    ``aug.third_drop`` of the samples.

    Args:
        row: A row of :func:`load_index`, with ``native_c`` as well.

    Returns:
        ``(a, b, target, agent, c, has_c)``. ``c`` is a float32
        ``(4, rows, cols)`` image, all zero when ``has_c`` is False.
    """
    a, b, target, agent, c, has_c = _build(
        chips_dir,
        row,
        cell_px=cell_px,
        pooling=pooling,
        mean=mean,
        std=std,
        aug=aug,
        rng=rng,
        cache_dir=cache_dir,
        third=True,
    )
    if c is None:
        c = np.zeros_like(a)
    return a, b, target, agent, c, has_c


def _build(
    chips_dir: Path,
    row: Any,
    *,
    cell_px: int,
    pooling: LabelPooling,
    mean: np.ndarray,
    std: np.ndarray,
    aug: Augment | None,
    rng: np.random.Generator | None,
    cache_dir: Path | None,
    third: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, bool]:
    """A sample, with the third image when ``third`` is set and it exists.

    Without ``third`` the random draws are those of the pair alone, so a
    pair model sees the same samples from a seed as before.
    """
    window_a = window_b = window_c = None
    reverse = False
    k, n_dates = int(row.k), int(row.n_dates)
    wanted = third and k + 2 < n_dates
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
        if third:
            # Drawn for every sample of a model with three dates, so that
            # the samples do not depend on which chips have a later date.
            dropped = rng.random() < aug.third_drop
            window_c = (
                row0 + int(rng.integers(-shift, shift + 1)),
                col0 + int(rng.integers(-shift, shift + 1)),
                size,
            )
            wanted = wanted and not dropped and not reverse
    arrays = read_pair(
        chips_dir,
        row.chip_id,
        k,
        n_dates,
        window_a=window_a,
        window_b=window_b,
        cache_dir=cache_dir,
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
    c = None
    if wanted:
        later = read_date(
            chips_dir,
            row.chip_id,
            k + 2,
            n_dates,
            window=window_c or window_a,
            cache_dir=cache_dir,
        )
        if later.any(axis=0).mean() >= THIRD_IMAGERY_SHARE:
            native_c = float(getattr(row, "native_c", float("nan")))
            c = _prepare(later, native_c, aug, rng, mean, std)
    if aug is not None and aug.flip and rng is not None:
        turns = int(rng.integers(0, 4))
        mirror = bool(rng.integers(0, 2))
        a, b, target, agent = (_turn(x, turns, mirror) for x in (a, b, target, agent))
        if c is not None:
            c = _turn(c, turns, mirror)
    return a, b, target, agent, c, c is not None


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

    With ``n_dates=3`` an item is ``(a, b, target, agent, unit_index, c,
    has_c)``: the image after the pair and whether the sample has one
    (:func:`build_triple`).
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
        cache_dir: Path | None = None,
        n_dates: int = 2,
    ) -> None:
        if n_dates not in (2, 3):
            raise ValueError("n_dates must be 2 or 3")
        self.n_dates = n_dates
        self.index = index.reset_index(drop=True)
        self.chips_dir = Path(chips_dir)
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
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

    def __getitem__(self, item: int) -> tuple[Any, ...]:
        import torch

        row = self.index.iloc[item]
        common = {
            "cell_px": self.cell_px,
            "pooling": self.pooling,
            "mean": self.mean,
            "std": self.std,
            "aug": self.aug,
            "rng": self._generator() if self.aug is not None else None,
            "cache_dir": self.cache_dir,
        }
        unit = self.units.index(row.unit)
        if self.n_dates == 2:
            a, b, target, agent = build_sample(self.chips_dir, row, **common)
            return (
                torch.from_numpy(a),
                torch.from_numpy(b),
                torch.from_numpy(target),
                torch.from_numpy(agent),
                unit,
            )
        a, b, target, agent, c, has_c = build_triple(self.chips_dir, row, **common)
        return (
            torch.from_numpy(a),
            torch.from_numpy(b),
            torch.from_numpy(target),
            torch.from_numpy(agent),
            unit,
            torch.from_numpy(c),
            torch.tensor(has_c),
        )
