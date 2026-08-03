"""Tests for the per-stand, per-date metric record."""

from __future__ import annotations

import numpy as np
import pytest

from csdv_core.zonal.record import StandMetricRecord, metric_matrix, records_to_frame


def _record(stand_id: str, year: int, **metrics: float) -> StandMetricRecord:
    return StandMetricRecord(
        stand_id=stand_id,
        date=f"{year}-07-01",
        year=year,
        native_res_m=0.6,
        area_m2=10_000.0,
        n_pixels=27_777,
        bbox_fill_fraction=0.5,
        metrics=metrics,
        support={"n_crowns": 42},
    )


def test_value_returns_nan_for_an_absent_metric():
    record = _record("A", 2018, gap_fraction=0.3)
    assert record.value("gap_fraction") == pytest.approx(0.3)
    assert np.isnan(record.value("crown_cv"))


def test_to_row_flattens_metrics_and_prefixes_support():
    row = _record("A", 2018, gap_fraction=0.3).to_row()
    assert row["stand_id"] == "A"
    assert row["gap_fraction"] == pytest.approx(0.3)
    assert row["support_n_crowns"] == 42
    assert row["unavailable"] == ""


def test_reasons_are_reported_rather_than_left_blank():
    record = StandMetricRecord(
        stand_id="A",
        date="2018-07-01",
        year=2018,
        native_res_m=0.6,
        area_m2=1.0,
        n_pixels=1,
        bbox_fill_fraction=1.0,
        metrics={"crown_cv": float("nan")},
        reasons={"crown_cv": "n_crowns=2 < min_crowns=3"},
    )
    assert "crown_cv: n_crowns=2 < min_crowns=3" in record.to_row()["unavailable"]


def test_records_to_frame_sorts_and_fills_missing_columns():
    records = [
        _record("B", 2018, gap_fraction=0.1),
        _record("A", 2022, gap_fraction=0.2, crown_cv=0.4),
        _record("A", 2018, gap_fraction=0.3),
    ]
    frame = records_to_frame(records)
    assert list(frame["stand_id"]) == ["A", "A", "B"]
    assert list(frame["year"]) == [2018, 2022, 2018]
    # crown_cv exists only on one record, so the others are NaN, not dropped.
    assert np.isnan(frame.loc[0, "crown_cv"])
    assert frame.loc[1, "crown_cv"] == pytest.approx(0.4)


def test_records_to_frame_handles_an_empty_input():
    assert records_to_frame([]).empty


def test_metric_matrix_orders_by_year_and_pads_unknown_metrics():
    frame = records_to_frame(
        [
            _record("A", 2022, gap_fraction=0.2),
            _record("A", 2018, gap_fraction=0.3),
        ]
    )
    values, dates = metric_matrix(frame, "A", ["gap_fraction", "ndvi_trend"])
    assert dates == ["2018-07-01", "2022-07-01"]
    assert values.shape == (2, 2)
    assert values[0].tolist() == pytest.approx([0.3, 0.2])
    # A metric with no implementation comes back all-NaN rather than raising.
    assert np.isnan(values[1]).all()
