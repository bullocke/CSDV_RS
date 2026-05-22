"""Evaluate metric envelopes from ``stages.yaml``.

Pure functions. NaN-safe: NaN metric values count as not evaluable, so
they reduce ``n_evaluated`` but never count as a match.

A single stage envelope is a mapping ``metric_name -> Range(min, max)``
where either bound may be ``None`` (unconstrained on that side).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from csdv_core.config._models import Range


@dataclass(frozen=True)
class StageMatch:
    """Result of comparing one set of metrics against one stage envelope.

    Attributes:
        n_evaluated: Number of metrics in the envelope that had a finite
            value in the input.
        n_matched: Of those evaluated, how many fell inside their range.
        score: ``n_matched / n_evaluated`` or 0.0 if nothing was evaluable.
        failed_metrics: Tuple of names that were evaluated but out of range.
    """

    n_evaluated: int
    n_matched: int
    score: float
    failed_metrics: tuple[str, ...]


def evaluate_envelope(value: float, range_: Range) -> bool:
    """Return True if ``value`` is finite and within ``range_``.

    A ``None`` bound is treated as unconstrained on that side.
    """
    if value is None or not np.isfinite(value):
        return False
    if range_.min is not None and value < range_.min:
        return False
    if range_.max is not None and value > range_.max:
        return False
    return True


def match_stage(
    metric_values: Mapping[str, float],
    envelopes: Mapping[str, Range],
) -> StageMatch:
    """Compare per-window metrics against one stage's envelope.

    Args:
        metric_values: Dict ``metric_name -> value``. Values may be NaN
            for missing metrics; those count as not evaluable.
        envelopes: Dict ``metric_name -> Range``.

    Returns:
        :class:`StageMatch`. ``score`` is 0.0 when no envelope metric is
        evaluable in the inputs.
    """
    n_eval = 0
    n_match = 0
    failed: list[str] = []
    for name, range_ in envelopes.items():
        value = metric_values.get(name)
        if value is None or not np.isfinite(float(value)):
            continue
        n_eval += 1
        if evaluate_envelope(float(value), range_):
            n_match += 1
        else:
            failed.append(name)
    score = (n_match / n_eval) if n_eval > 0 else 0.0
    return StageMatch(
        n_evaluated=n_eval,
        n_matched=n_match,
        score=score,
        failed_metrics=tuple(failed),
    )
