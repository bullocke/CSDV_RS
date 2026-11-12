"""csdv_core.label.pooling: pool 0.6 m labels to the model's output cells.

Labels are stored at 0.6 m. The classifier predicts on coarser cells
(2.4, 4.8 or 9.6 m), so the same rule serves the review sheets, the
diagnostics and the data loader.
"""

from __future__ import annotations

import numpy as np

from csdv_core.config._models import LabelPooling

IGNORE = 255

__all__ = ["pool_agent", "pool_change"]


def _blocks(array: np.ndarray, cell_px: int) -> np.ndarray:
    rows, cols = array.shape[-2] // cell_px, array.shape[-1] // cell_px
    trimmed = array[..., : rows * cell_px, : cols * cell_px]
    shape = (*array.shape[:-2], rows, cell_px, cols, cell_px)
    return trimmed.reshape(shape).swapaxes(-3, -2).reshape(*shape[:-4], rows, cols, -1)


def pool_change(change: np.ndarray, cell_px: int, cfg: LabelPooling) -> np.ndarray:
    """Pool a change mask to cells of ``cell_px`` pixels.

    A cell is ignore when at least ``ignore_share`` of it is ignore. Of its
    labelled pixels, a loss share of ``loss_share`` or more makes it loss
    (the more common severity wins, stand-replacing on a tie), a share of
    ``background_share`` or less makes it background, and anything between
    is ignore, because a mixed cell teaches neither class.

    Args:
        change: ``(..., rows, cols)`` uint8 with 0, 1, 2 and 255.
        cell_px: Pixels per cell side.
        cfg: Pooling thresholds.
    """
    blocks = _blocks(change, cell_px)
    ignore = (blocks == IGNORE).mean(axis=-1)
    partial = (blocks == 1).sum(axis=-1)
    full = (blocks == 2).sum(axis=-1)
    labelled = partial + full + (blocks == 0).sum(axis=-1)
    share = (partial + full) / np.maximum(labelled, 1)
    out = np.full(ignore.shape, IGNORE, dtype=np.uint8)
    usable = (ignore < cfg.ignore_share) & (labelled > 0)
    out[usable & (share <= cfg.background_share)] = 0
    lost = usable & (share >= cfg.loss_share)
    out[lost & (full >= partial)] = 2
    out[lost & (full < partial)] = 1
    return out


def pool_agent(
    agent: np.ndarray, pooled_change: np.ndarray, cell_px: int
) -> np.ndarray:
    """Most common agent code among the loss pixels of each loss cell."""
    blocks = _blocks(agent, cell_px)
    out = np.full(pooled_change.shape, IGNORE, dtype=np.uint8)
    codes = np.unique(agent[agent != IGNORE])
    best = np.zeros(pooled_change.shape, dtype=np.int64)
    for code in codes:
        count = (blocks == code).sum(axis=-1)
        better = count > best
        out[better] = code
        best[better] = count[better]
    out[~np.isin(pooled_change, (1, 2))] = IGNORE
    return out
