"""csdv_core.pairmodel.score: a checkpoint against the label chips of many units.

Inputs: a checkpoint of ``csdv pair train``, a chip folder, and the units to
score (places the model did not train on). Outputs, in the run's ``score``
folder:

- ``samples.parquet``: one row per chip and interval with the columns of the
  index and two lists of 101 counts, ``pos`` and ``neg``: the labelled loss
  cells and background cells of the sample by the predicted probability in
  percent. Any grouping (season, target, sampling stratum) can be pooled
  from it without the model.
- ``units.csv``: scores per unit at the checkpoint's threshold.
- ``score.json``: the pooled scores, and those of the intervals that end on
  an image from November to April and of the others.
- ``agent_confusion.csv``: for a model with an agent head, the agent the
  model names on the loss cells whose label has one (rows the label, columns
  the model), as ``agent_confusion_val.csv`` of a training run.

``csdv pair predict`` and ``compare`` map and score one whole unit. This
module scores label chips only, which is what a test on many small places
needs: twenty reserved boxes take a few minutes on one GPU.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from csdv_core.pairmodel import data as D
from csdv_core.pairmodel.metrics import AgentCounts, CellCounts, to_percent

logger = logging.getLogger(__name__)

#: Months of the growing season for the split of the scores by season.
SUMMER_MONTHS = range(5, 11)

__all__ = ["ScoreResult", "counts_of", "score", "summarise", "write_tables"]


@dataclass
class ScoreResult:
    """What a scoring run wrote.

    Attributes:
        out_dir: Folder of the tables.
        units: Units that were scored.
        samples: Number of chip intervals.
        threshold: Threshold of the checkpoint.
        pooled: Scores over every labelled cell.
    """

    out_dir: Path
    units: tuple[str, ...]
    samples: int
    threshold: float
    pooled: dict[str, float]


def counts_of(rows: pd.DataFrame) -> CellCounts:
    """The cells of several samples as one pair of histograms."""
    if rows.empty:
        return CellCounts()
    return CellCounts(
        np.sum(np.stack(rows["pos"].to_numpy()), axis=0).astype("int64"),
        np.sum(np.stack(rows["neg"].to_numpy()), axis=0).astype("int64"),
    )


def summarise(counts: CellCounts, threshold: float) -> dict[str, float]:
    """Average precision, the scores at a threshold, and the false alarms."""
    at = counts.at(threshold)
    cut = int(round(threshold * 100))
    background = int(counts.neg.sum())
    return {
        "loss_cells": counts.n_pos,
        "background_cells": background,
        "average_precision": counts.average_precision(),
        "precision": at["precision"],
        "recall": at["recall"],
        "f1": at["f1"],
        "background_flagged": (
            float(counts.neg[cut:].sum() / background) if background else float("nan")
        ),
    }


def _off_season(date: str) -> bool:
    return int(str(date)[5:7]) not in SUMMER_MONTHS


def write_tables(
    samples: pd.DataFrame, threshold: float, out_dir: Path, extra: dict[str, Any]
) -> dict[str, float]:
    """Write ``samples.parquet``, ``units.csv`` and ``score.json``.

    Returns:
        The pooled scores.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(out_dir / "samples.parquet", index=False)
    rows = []
    for unit, group in samples.groupby("unit"):
        rows.append(
            {
                "unit": unit,
                "samples": len(group),
                **summarise(counts_of(group), threshold),
            }
        )
    units = pd.DataFrame(rows)
    units.to_csv(out_dir / "units.csv", index=False)
    pooled = summarise(counts_of(samples), threshold)
    off = samples["date_b"].map(_off_season)
    with_loss = units[units["loss_cells"] > 0]
    record = {
        **extra,
        "threshold": threshold,
        "units": int(samples["unit"].nunique()),
        "samples": len(samples),
        "pooled": pooled,
        "mean_unit_average_precision": (
            float(with_loss["average_precision"].mean()) if len(with_loss) else None
        ),
        "ends_november_to_april": summarise(counts_of(samples[off]), threshold),
        "ends_may_to_october": summarise(counts_of(samples[~off]), threshold),
    }
    text = json.dumps(record, indent=2, default=float).replace("NaN", "null")
    (out_dir / "score.json").write_text(text + "\n", encoding="utf-8")
    return pooled


def score(
    checkpoint: Path,
    chips_dir: Path,
    out_dir: Path,
    units: Sequence[str],
    *,
    batch_size: int = 16,
    workers: int = 4,
    device: str = "auto",
    cache_dir: Path | None = None,
) -> ScoreResult:
    """Score a checkpoint on every labelled chip interval of ``units``.

    Args:
        checkpoint: ``best.pt`` of a training run.
        chips_dir: Chip folder that holds the units' chips.
        out_dir: Folder for the tables.
        units: Units to score. Units without a chip are named in the log
            and left out.
        batch_size: Chips per step. Chips are read whole, without any
            random change, as in validation.
        workers: Loader processes.
        device: ``cuda``, ``cpu`` or ``auto``.
        cache_dir: Folder of decoded chips, when one was built.

    Raises:
        ValueError: If none of the units has a chip.
    """
    import torch
    from torch.utils.data import DataLoader

    from csdv_core.config import load_label
    from csdv_core.pairmodel.train import _device, _third, load_checkpoint

    index = D.load_index(chips_dir)
    wanted = sorted(set(units))
    absent = sorted(set(wanted) - set(index["unit"]))
    if absent:
        logger.info("%d units have no chip and are left out: %s", len(absent), absent)
    part = index[index["unit"].isin(wanted)].reset_index(drop=True)
    if part.empty:
        raise ValueError("None of the units to score has a chip in the folder")
    names = sorted(part["unit"].unique())
    dev = _device(device)
    model, ckpt = load_checkpoint(checkpoint, dev)
    threshold = float(ckpt.get("best_threshold", 0.5))
    dataset = D.PairDataset(
        part,
        chips_dir,
        cell_px=int(ckpt["cell_px"]),
        pooling=load_label().pooling,
        mean=np.asarray(ckpt["mean"]),
        std=np.asarray(ckpt["std"]),
        units=names,
        cache_dir=cache_dir,
        n_dates=int(ckpt.get("n_dates", 2)),
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers
    )
    pos: list[np.ndarray] = []
    neg: list[np.ndarray] = []
    agents = AgentCounts(D.AGENT_NAMES)
    amp = dev.type == "cuda"
    with torch.no_grad():
        for a, b, target, agent, _unit, *third in loader:
            a, b = a.to(dev, non_blocking=True), b.to(dev, non_blocking=True)
            c, has_c = _third(third, dev)
            with torch.autocast(dev.type, enabled=amp):
                logits, agent_logits = model.forward_agents(a, b, c, has_c)
            if agent_logits is not None:
                agents.update(agent_logits.argmax(dim=1).cpu().numpy(), agent.numpy())
            percent = to_percent(torch.sigmoid(logits.float()).squeeze(1).cpu().numpy())
            labels = target.numpy()
            for i in range(len(percent)):
                counts = CellCounts()
                counts.update(percent[i], labels[i])
                pos.append(counts.pos)
                neg.append(counts.neg)
    samples = part.copy()
    samples["pos"] = [p.tolist() for p in pos]
    samples["neg"] = [n.tolist() for n in neg]
    pooled = write_tables(
        samples,
        threshold,
        out_dir,
        {"checkpoint": str(checkpoint), "chips_dir": str(chips_dir)},
    )
    if agents.n:
        pd.DataFrame(agents.rows()).to_csv(
            Path(out_dir) / "agent_confusion.csv", index=False
        )
        logger.info(
            "Agent head: %d cells with an agent, accuracy %.3f, balanced %.3f",
            agents.n,
            agents.accuracy(),
            agents.balanced_accuracy(),
        )
    logger.info(
        "%d samples of %d units: average precision %.3f, recall %.3f, "
        "background flagged %.4f at %.2f",
        len(samples),
        len(names),
        pooled["average_precision"],
        pooled["recall"],
        pooled["background_flagged"],
        threshold,
    )
    return ScoreResult(Path(out_dir), tuple(names), len(samples), threshold, pooled)
