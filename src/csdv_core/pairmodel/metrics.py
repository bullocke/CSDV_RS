"""csdv_core.pairmodel.metrics: the training loss and cell-level scores.

:class:`CellCounts` keeps two histograms of predicted probability in steps
of one percent, one for loss cells and one for background cells. The map is
stored as uint8 percent, so the same class scores validation batches and a
finished raster, and counts of several units or intervals can be added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

IGNORE = 255

__all__ = ["AgentCounts", "CellCounts", "agent_loss", "masked_bce_dice", "to_percent"]


def to_percent(prob: np.ndarray) -> np.ndarray:
    """Probabilities in 0..1 as uint8 percent."""
    return np.clip(np.rint(np.asarray(prob, dtype="float64") * 100), 0, 100).astype(
        "uint8"
    )


@dataclass
class CellCounts:
    """Histograms of predicted percent for loss cells and background cells."""

    pos: np.ndarray = field(default_factory=lambda: np.zeros(101, dtype="int64"))
    neg: np.ndarray = field(default_factory=lambda: np.zeros(101, dtype="int64"))

    def update(self, percent: np.ndarray, target: np.ndarray) -> None:
        """Add cells.

        Args:
            percent: Predicted probability as uint8 percent (0 to 100).
                Values above 100 (no prediction) are left out.
            target: 0 background, 1 loss, 255 ignore, the same shape.
        """
        percent = np.asarray(percent).ravel()
        target = np.asarray(target).ravel()
        known = percent <= 100
        self.pos += np.bincount(percent[known & (target == 1)], minlength=101)[:101]
        self.neg += np.bincount(percent[known & (target == 0)], minlength=101)[:101]

    def __add__(self, other: CellCounts) -> CellCounts:
        return CellCounts(self.pos + other.pos, self.neg + other.neg)

    @property
    def n_pos(self) -> int:
        return int(self.pos.sum())

    @property
    def n_valid(self) -> int:
        return int(self.pos.sum() + self.neg.sum())

    def at(self, threshold: float) -> dict[str, float]:
        """Scores when cells at or above ``threshold`` (0..1) are called loss."""
        t = int(round(threshold * 100))
        tp = float(self.pos[t:].sum())
        fp = float(self.neg[t:].sum())
        fn = float(self.pos[:t].sum())
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
        iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
        return {
            "threshold": t / 100,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "n_pos": self.n_pos,
            "n_valid": self.n_valid,
        }

    def curve(self) -> list[dict[str, float]]:
        """Scores at every threshold from 0.01 to 1.00."""
        return [self.at(t / 100) for t in range(1, 101)]

    def average_precision(self) -> float:
        """Area under the precision-recall curve (step rule). NaN without loss."""
        if not self.n_pos:
            return float("nan")
        tp = np.cumsum(self.pos[::-1]).astype("float64")
        fp = np.cumsum(self.neg[::-1]).astype("float64")
        called = tp + fp
        precision = np.divide(tp, called, out=np.ones_like(tp), where=called > 0)
        recall = tp / self.n_pos
        steps = np.diff(np.concatenate([[0.0], recall]))
        return float((steps * precision).sum())

    def best_f1(self) -> tuple[float, float]:
        """The threshold with the highest F1, and that F1."""
        best = max(self.curve(), key=lambda row: np.nan_to_num(row["f1"], nan=-1.0))
        return best["threshold"], best["f1"]


def masked_bce_dice(logits: Any, target: Any, dice_weight: float = 1.0) -> Any:
    """Binary cross-entropy plus soft Dice over the labelled cells of a batch.

    Args:
        logits: ``(batch, 1, rows, cols)`` float tensor.
        target: ``(batch, rows, cols)`` uint8 tensor with 0, 1 and 255.
        dice_weight: Weight of the Dice term.

    Returns:
        A scalar tensor. Zero (with a gradient path) when no cell is labelled.
    """
    import torch
    import torch.nn.functional as F

    logits = logits.squeeze(1).float()
    valid = target != IGNORE
    if not bool(valid.any()):
        return logits.sum() * 0.0
    z = logits[valid]
    y = target[valid].float()
    bce = F.binary_cross_entropy_with_logits(z, y)
    p = torch.sigmoid(z)
    dice = 1.0 - (2.0 * (p * y).sum() + 1.0) / (p.sum() + y.sum() + 1.0)
    return bce + dice_weight * dice


def agent_loss(agent_logits: Any, agent: Any) -> Any:
    """Cross-entropy of the agent head over the cells that have an agent.

    Args:
        agent_logits: ``(batch, classes, rows, cols)`` float tensor.
        agent: ``(batch, rows, cols)`` uint8 tensor, a class index or 255.

    Returns:
        A scalar tensor. Zero (with a gradient path) when no cell has an agent.
    """
    import torch.nn.functional as F

    if not bool((agent != IGNORE).any()):
        return agent_logits.sum() * 0.0
    return F.cross_entropy(agent_logits.float(), agent.long(), ignore_index=IGNORE)


@dataclass
class AgentCounts:
    """Confusion matrix of the agent head: rows the label, columns the model."""

    names: tuple[str, ...]
    matrix: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.matrix is None:
            n = len(self.names)
            self.matrix = np.zeros((n, n), dtype="int64")

    def update(self, predicted: np.ndarray, agent: np.ndarray) -> None:
        """Add cells.

        Args:
            predicted: Class index the model gives each cell. Values at or
                above the number of classes (no prediction) are left out.
            agent: Class index of the label, 255 where there is none.
        """
        n = len(self.names)
        predicted = np.asarray(predicted).ravel()
        agent = np.asarray(agent).ravel()
        known = (agent < n) & (predicted < n)
        flat = agent[known].astype("int64") * n + predicted[known]
        self.matrix += np.bincount(flat, minlength=n * n).reshape(n, n)

    def __add__(self, other: AgentCounts) -> AgentCounts:
        return AgentCounts(self.names, self.matrix + other.matrix)

    @property
    def n(self) -> int:
        return int(self.matrix.sum())

    def accuracy(self) -> float:
        """Share of cells with an agent whose agent the model names."""
        return float(np.trace(self.matrix) / self.n) if self.n else float("nan")

    def recall(self) -> dict[str, float]:
        """Per agent: the share of its label cells the model names correctly."""
        totals = self.matrix.sum(axis=1)
        return {
            name: float(self.matrix[i, i] / totals[i]) if totals[i] else float("nan")
            for i, name in enumerate(self.names)
        }

    def balanced_accuracy(self) -> float:
        """Mean recall over the agents that have label cells."""
        values = [v for v in self.recall().values() if np.isfinite(v)]
        return float(np.mean(values)) if values else float("nan")

    def rows(self) -> list[dict[str, Any]]:
        """The matrix as table rows, with the recall of each agent."""
        recall = self.recall()
        out = []
        for i, name in enumerate(self.names):
            row: dict[str, Any] = {"label": name, "cells": int(self.matrix[i].sum())}
            row.update(
                {f"as_{n}": int(self.matrix[i, j]) for j, n in enumerate(self.names)}
            )
            row["recall"] = recall[name]
            out.append(row)
        return out
