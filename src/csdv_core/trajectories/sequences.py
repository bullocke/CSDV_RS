"""csdv_core.trajectories.sequences — shape one stand's series as a cube.

The trajectory rule engine works on a stack of dates over a raster: a
``(T, H, W)`` stage cube and one ``(T, H, W)`` cube per metric. Nothing in it
depends on ``H`` and ``W`` being larger than one, so a stand is a legal raster
of a single pixel. Building that one-pixel cube lets the stand path reuse
:func:`csdv_core.trajectories.rules.evaluate_rule` and
:func:`csdv_core.trajectories.classify.classify_trajectories` unchanged, rather
than growing a second copy of the predicate logic that could drift from the
first.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from csdv_core.config import StagesConfig

logger = logging.getLogger(__name__)

__all__ = ["StandSequence", "build_sequence", "stand_cubes"]


@dataclass(frozen=True)
class StandSequence:
    """One stand's stage and metric series across the imagery record.

    Attributes:
        stand_id: Stand identifier.
        dates: Acquisition dates in ascending order.
        years: Imagery years in the same order.
        stages: Stage code per date, None where unclassified.
        metrics: Metric name to a tuple of one value per date.
    """

    stand_id: str
    dates: tuple[str, ...]
    years: tuple[int, ...]
    stages: tuple[str | None, ...]
    metrics: Mapping[str, tuple[float, ...]]

    def __len__(self) -> int:
        return len(self.dates)


def build_sequence(
    stand_id: str,
    frame: pd.DataFrame,
    stages: Sequence[str | None],
    *,
    metric_names: Sequence[str],
) -> StandSequence:
    """Assemble a :class:`StandSequence` from one stand's rows.

    Args:
        stand_id: Stand identifier.
        frame: Rows for this stand only, any order. Needs ``date`` and ``year``.
        stages: Stage code per row, in ascending year order.
        metric_names: Metrics to carry. A name absent from ``frame`` becomes a
            tuple of NaN rather than being dropped, so a rule that references
            an unimplemented metric reports as inert instead of raising.

    Raises:
        ValueError: If ``stages`` and ``frame`` disagree on length.
    """
    subset = frame.sort_values("year")
    if len(stages) != len(subset):
        raise ValueError(
            f"Stand {stand_id!r}: {len(stages)} stages for {len(subset)} dates"
        )
    metrics: dict[str, tuple[float, ...]] = {}
    for name in metric_names:
        if name in subset.columns:
            metrics[name] = tuple(float(v) for v in subset[name])
        else:
            metrics[name] = tuple(float("nan") for _ in range(len(subset)))
    return StandSequence(
        stand_id=stand_id,
        dates=tuple(str(d) for d in subset["date"]),
        years=tuple(int(y) for y in subset["year"]),
        stages=tuple(stages),
        metrics=metrics,
    )


def stand_cubes(
    sequence: StandSequence,
    stages_cfg: StagesConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Shape a stand's series as the ``(T, 1, 1)`` cubes the rule engine takes.

    Returns:
        ``(stage_cube, metric_cubes)``. The stage cube is uint8, with 0 for a
        date left unclassified, matching the raster convention.
    """
    codes = dict(stages_cfg.stage_codes)
    if not codes:
        order = list(stages_cfg.stage_order) or list(stages_cfg.stages.keys())
        codes = {name: i + 1 for i, name in enumerate(order)}

    n_dates = len(sequence)
    stage_cube = np.zeros((n_dates, 1, 1), dtype="uint8")
    for t, stage in enumerate(sequence.stages):
        if stage is not None:
            stage_cube[t, 0, 0] = np.uint8(codes.get(stage, 0))

    metric_cubes: dict[str, np.ndarray] = {}
    for name, values in sequence.metrics.items():
        cube = np.full((n_dates, 1, 1), np.nan, dtype="float32")
        for t, value in enumerate(values):
            cube[t, 0, 0] = np.float32(value)
        metric_cubes[name] = cube
    return stage_cube, metric_cubes
