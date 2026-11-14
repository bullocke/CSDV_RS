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

__all__ = ["CellCounts", "masked_bce_dice", "to_percent"]


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
