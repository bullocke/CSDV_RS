"""csdv_core.zonal.record — the per-stand, per-date metric record.

One :class:`StandMetricRecord` is the unit the classification system consumes:
every metric for one stand on one imagery date, plus enough context to say how
far each number can be trusted. A metric that could not be computed is NaN and
carries a reason, so a blank in a table can always be traced to a cause rather
than left to guesswork.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["StandMetricRecord", "records_to_frame"]


@dataclass(frozen=True)
class StandMetricRecord:
    """Metrics for one stand on one imagery date.

    Attributes:
        stand_id: Stand identifier from :func:`csdv_core.io.stands.read_ais_stands`.
        date: Acquisition date as ``YYYY-MM-DD``.
        year: Imagery year, kept separately because it labels every figure axis.
        native_res_m: Ground sample distance of the source imagery. NAIP is
            0.6 m from 2016 onward in most states and 1.0 m before that, and a
            canopy height model derived from 1.0 m imagery is not equivalent to
            one derived from 0.6 m.
        area_m2: Stand area from the geometry.
        n_pixels: Valid in-stand pixels the metrics were computed over.
        bbox_fill_fraction: In-stand share of the bounding box. Read any texture
            or spatial metric against this.
        metrics: Metric name to value. Names match what the windowed metric
            functions emit, so a record can be matched against a stage envelope
            without translation.
        support: Auxiliary counts, for example ``n_crowns``.
        reasons: Metric name to the reason its value is NaN.
    """

    stand_id: str
    date: str
    year: int
    native_res_m: float
    area_m2: float
    n_pixels: int
    bbox_fill_fraction: float
    metrics: Mapping[str, float] = field(default_factory=dict)
    support: Mapping[str, float] = field(default_factory=dict)
    reasons: Mapping[str, str] = field(default_factory=dict)

    def value(self, name: str) -> float:
        """Return a metric value, or NaN if it was not computed."""
        return float(self.metrics.get(name, float("nan")))

    def to_row(self) -> dict[str, Any]:
        """Flatten to one row, prefixing support columns with ``support_``."""
        row: dict[str, Any] = {
            "stand_id": self.stand_id,
            "date": self.date,
            "year": self.year,
            "native_res_m": self.native_res_m,
            "area_m2": self.area_m2,
            "n_pixels": self.n_pixels,
            "bbox_fill_fraction": self.bbox_fill_fraction,
        }
        row.update({name: float(value) for name, value in self.metrics.items()})
        row.update({f"support_{k}": v for k, v in self.support.items()})
        row["unavailable"] = "; ".join(
            f"{name}: {why}" for name, why in sorted(self.reasons.items())
        )
        return row


def records_to_frame(records: Sequence[StandMetricRecord]) -> pd.DataFrame:
    """Stack records into a tidy frame sorted by stand and date.

    Metric columns absent from some records become NaN rather than being
    dropped, so a stand missing one date still lines up with the rest.
    """
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame([record.to_row() for record in records])
    frame = frame.sort_values(["stand_id", "year"]).reset_index(drop=True)
    n_missing = int(frame["unavailable"].astype(bool).sum())
    if n_missing:
        logger.info(
            "%d of %d stand-date records have at least one unavailable metric",
            n_missing,
            len(frame),
        )
    return frame


def metric_matrix(
    frame: pd.DataFrame,
    stand_id: str,
    metric_names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Return one stand's metrics as a ``(n_metrics, n_dates)`` array.

    Missing columns come back as all-NaN rows rather than raising, because the
    trajectory rules reference metrics that have no implementation yet and a
    rule holding an unimplemented metric should report as inert, not crash.

    Returns:
        ``(values, dates)`` with dates in ascending year order.
    """
    subset = frame[frame["stand_id"] == stand_id].sort_values("year")
    dates = [str(d) for d in subset["date"]]
    out = np.full((len(metric_names), len(subset)), np.nan, dtype=np.float32)
    for i, name in enumerate(metric_names):
        if name in subset.columns:
            out[i, :] = subset[name].to_numpy(dtype=np.float32)
    return out, dates
