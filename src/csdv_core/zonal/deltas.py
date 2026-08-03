"""csdv_core.zonal.deltas — change between consecutive dates for one stand.

A single date says what a stand looks like. The difference between two dates
says what happened to it, and several events that look alike in one snapshot
separate cleanly in the differences. A clearcut drives crown fraction and crown
width P90 down together. A commercial thinning lowers crown fraction while
leaving P90 flat or slightly higher, because the trees left standing are the
larger ones. Highgrading does the reverse, pulling P90 down while crown fraction
barely moves.

Differences are taken on an identical grid with no resampling, which the
canonical analysis grid guarantees upstream. The error in a difference compounds
the error of both inputs, so a change threshold has to sit above the combined
noise floor rather than above the error of a single date.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

logger = logging.getLogger(__name__)

#: Metrics differenced between consecutive dates, and the prefix used for the
#: resulting columns.
CHANGE_METRICS: tuple[str, ...] = (
    "crown_fraction",
    "gap_fraction",
    "crown_p90",
    "crown_count",
    "edge_density",
)
DELTA_PREFIX = "d_"

__all__ = ["CHANGE_METRICS", "DELTA_PREFIX", "add_change_metrics", "delta_columns"]


def delta_columns(metrics: Sequence[str] = CHANGE_METRICS) -> list[str]:
    """Return the column names :func:`add_change_metrics` produces."""
    return [f"{DELTA_PREFIX}{name}" for name in metrics]


def add_change_metrics(
    frame: pd.DataFrame,
    *,
    metrics: Sequence[str] = CHANGE_METRICS,
    by: str = "stand_id",
    order: str = "year",
) -> pd.DataFrame:
    """Add consecutive-date differences per stand.

    The value at date *t* is the change from *t-1* to *t*, so the first date of
    every stand is NaN. A metric absent from ``frame`` produces an all-NaN
    column rather than being skipped, so downstream code can rely on the column
    existing.

    Args:
        frame: Tidy stand-date frame, as from
            :func:`csdv_core.zonal.record.records_to_frame`.
        metrics: Metric columns to difference.
        by: Grouping column, one group per stand.
        order: Column defining date order within a stand.

    Returns:
        A copy of ``frame`` with one ``d_<metric>`` column per entry in
        ``metrics``.
    """
    if frame.empty:
        return frame.copy()
    out = frame.sort_values([by, order]).copy()
    for name in metrics:
        column = f"{DELTA_PREFIX}{name}"
        if name not in out.columns:
            out[column] = float("nan")
            logger.info("No %s column, %s reported as NaN", name, column)
            continue
        out[column] = out.groupby(by, sort=False)[name].diff()
    return out.reset_index(drop=True)
