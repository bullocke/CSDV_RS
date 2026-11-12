"""csdv_core.label.diagnostics: numbers that say how usable the labels are.

Inputs: a window's stack, label rasters and candidates.
Outputs: plain rows (dicts) for three tables. One row per window and
interval with the share of every reason group at 0.6 m, the class shares
after pooling to each cell size, and where cells with a strong greenness
loss ended up. One row per accepted candidate and interval with how many
output cells its label keeps. One row per cluster of background cells with a
strong greenness loss, which are the places to look for missed disturbance.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi

from csdv_core.config._models import LabelConfig
from csdv_core.label.labels import JudgedCandidate, WindowLabels, burn
from csdv_core.label.lattice import GSD, Window
from csdv_core.label.pooling import pool_change
from csdv_core.label.routing import LOSS_OUTCOMES
from csdv_core.label.rules import IGNORE, REASON_GROUPS, Reason, reason_group
from csdv_core.label.screen import cell_ndvi, off_season, residual_z
from csdv_core.label.stacks import WindowStack

#: Cells this many robust deviations below reference forest count as strong loss.
STRONG_Z = -3.0
_NAMES = list(REASON_GROUPS)
_CLASS = {0: "background", 1: "partial", 2: "full", IGNORE: "ignore"}

__all__ = [
    "STRONG_Z",
    "event_rows",
    "greenness_z",
    "interval_rows",
    "suspect_rows",
]


def _majority_group(reason: np.ndarray, cell_px: int, pure: float = 0.9) -> np.ndarray:
    """Reason group of each cell.

    A cell counts as background only when at least ``pure`` of its pixels are
    background, and then takes its most common background group. Any other
    cell takes its most common loss or ignore group, so a cell on the edge of
    a labelled event is never counted as background.
    """
    group = reason_group(reason)
    rows, cols = group.shape[0] // cell_px, group.shape[1] // cell_px
    blocks = group[: rows * cell_px, : cols * cell_px].reshape(
        rows, cell_px, cols, cell_px
    )
    counts = np.stack(
        [(blocks == i).sum(axis=(1, 3)) for i in range(len(_NAMES))], axis=0
    )
    background = np.array([name.startswith("background") for name in _NAMES])
    is_background = counts[background].sum(axis=0) >= pure * cell_px * cell_px
    masked = np.where(background[:, None, None] == is_background[None], counts, -1)
    return masked.argmax(axis=0)


def greenness_z(
    stack: WindowStack, window: Window, labels: WindowLabels, k: int, cell_px: int = 16
) -> np.ndarray | None:
    """Greenness-loss z of every cell of a window in interval ``k``.

    Reference cells are those whose pixels are mostly unjudged forest
    background with a low disturbance probability.
    """
    data = stack.crop(window.bounds)
    pre = cell_ndvi(data[k], cell_px)
    post = cell_ndvi(data[k + 1], cell_px)
    majority = _majority_group(labels.reason[k], cell_px)
    reference = majority == _NAMES.index("background_forest")
    if labels.unstable[k]:
        # The forest background was withdrawn. Fall back on forest at both dates.
        both = labels.forest[k].astype(bool) & labels.forest[k + 1].astype(bool)
        rows, cols = pre.shape
        share = (
            both[: rows * cell_px, : cols * cell_px]
            .reshape(rows, cell_px, cols, cell_px)
            .mean(axis=(1, 3))
        )
        reference = share > 0.9
    return residual_z(pre, post, reference)


def interval_rows(
    stack: WindowStack,
    window: Window,
    labels: WindowLabels,
    cfg: LabelConfig,
) -> list[dict[str, Any]]:
    """One row per interval of a window."""
    rows = []
    for k, interval in enumerate(labels.intervals):
        reason = labels.reason[k]
        group = reason_group(reason)
        row: dict[str, Any] = {
            "window_id": window.window_id,
            "unit": window.unit,
            "interval": interval,
            "chips": window.nx * window.ny,
            "unstable": labels.unstable[k],
            "off_season": off_season(stack.dates[k]) or off_season(stack.dates[k + 1]),
            "native_m": f"{stack.native_m[k]}->{stack.native_m[k + 1]}",
            "resolution_change": stack.native_m[k] != stack.native_m[k + 1],
        }
        for i, name in enumerate(_NAMES):
            row[name] = float((group == i).mean())
        for code in (
            Reason.BOUNDARY_INSIDE,
            Reason.BOUNDARY_OUTSIDE,
            Reason.MARGIN,
            Reason.RAISED_PROB,
            Reason.FOREST_LOST_UNJUDGED,
            Reason.SPARSE_WOODY,
            Reason.UNSTABLE_FOREST,
            Reason.EXTENSION_IGNORED,
            Reason.REJECTED_FLAGGED,
        ):
            row[code.name.lower()] = float((reason == code).mean())
        for cell_m in cfg.pooling.cells_m:
            pooled = pool_change(labels.change[k], round(cell_m / GSD), cfg.pooling)
            for value, name in _CLASS.items():
                row[f"cell{cell_m:g}_{name}"] = float((pooled == value).mean())
        z = greenness_z(stack, window, labels, k)
        if z is not None:
            majority = _majority_group(reason, 16)
            strong = np.nan_to_num(z, nan=0.0) <= STRONG_Z
            row["strong_cells"] = int(strong.sum())
            for i, name in enumerate(_NAMES):
                row[f"strong_{name}"] = int((strong & (majority == i)).sum())
        rows.append(row)
    return rows


def event_rows(
    window: Window,
    labels: WindowLabels,
    candidates: list[JudgedCandidate],
    cfg: LabelConfig,
) -> list[dict[str, Any]]:
    """How much of each accepted candidate survives as labelled loss.

    For every candidate and interval with agreed loss: the candidate's
    pixels inside the window, the pixels labelled loss, and the number of
    output cells labelled loss among the cells the candidate mostly covers.
    """
    rows = []
    for cand in candidates:
        loss_at = [k for k, o in enumerate(cand.routed.outcomes) if o in LOSS_OUTCOMES]
        if not loss_at or cand.event is None:
            continue
        mask = burn(cand.geometry, window.bounds)
        if not mask.any():
            continue
        for k in loss_at:
            change = labels.change[k]
            row: dict[str, Any] = {
                "window_id": window.window_id,
                "cand_id": cand.cand_id,
                "interval": labels.intervals[k],
                "source": cand.event.source,
                "quality": cand.event.quality,
                "crisp": cand.event.crisp,
                "change_type": cand.routed.change_type,
                "types": "/".join(str(t) for t in cand.routed.change_types),
                "outcome": cand.routed.outcomes[k],
                "candidate_ha": float(mask.sum() * GSD * GSD / 1e4),
                "loss_share_px": float(np.isin(change[mask], (1, 2)).mean()),
                "band_share_px": float(
                    (labels.reason[k][mask] == Reason.BOUNDARY_INSIDE).mean()
                ),
            }
            for cell_m in cfg.pooling.cells_m:
                c = round(cell_m / GSD)
                pooled = pool_change(change, c, cfg.pooling)
                r, q = mask.shape[0] // c, mask.shape[1] // c
                inside = (
                    mask[: r * c, : q * c].reshape(r, c, q, c).mean(axis=(1, 3)) >= 0.5
                )
                row[f"cells{cell_m:g}_inside"] = int(inside.sum())
                row[f"cells{cell_m:g}_loss"] = int(
                    np.isin(pooled[inside], (1, 2)).sum()
                )
            rows.append(row)
    return rows


def suspect_rows(
    stack: WindowStack,
    window: Window,
    labels: WindowLabels,
    cell_px: int = 16,
    min_cells: int = 4,
) -> list[dict[str, Any]]:
    """Clusters of forest-background cells with a strong greenness loss.

    These are the places where the background may hide a disturbance that no
    candidate source flagged. Each row gives the cluster's bounds, size and
    mean z, so that a crop of the two dates can be drawn and looked at.
    """
    rows = []
    forest = _NAMES.index("background_forest")
    for k, interval in enumerate(labels.intervals):
        z = greenness_z(stack, window, labels, k, cell_px)
        if z is None:
            continue
        majority = _majority_group(labels.reason[k], cell_px)
        strong = (np.nan_to_num(z, nan=0.0) <= STRONG_Z) & (majority == forest)
        found, n = ndi.label(strong, structure=np.ones((3, 3), bool))
        for index in range(1, n + 1):
            cells = found == index
            if cells.sum() < min_cells:
                continue
            rr, cc = np.nonzero(cells)
            x0 = window.bounds[0] + cc.min() * cell_px * GSD
            y1 = window.bounds[3] - rr.min() * cell_px * GSD
            rows.append(
                {
                    "window_id": window.window_id,
                    "unit": window.unit,
                    "interval": interval,
                    "k": k,
                    "cells": int(cells.sum()),
                    "mean_z": float(np.nanmean(z[cells])),
                    "x0": float(x0),
                    "y0": float(y1 - (rr.max() - rr.min() + 1) * cell_px * GSD),
                    "x1": float(x0 + (cc.max() - cc.min() + 1) * cell_px * GSD),
                    "y1": float(y1),
                }
            )
    return rows
