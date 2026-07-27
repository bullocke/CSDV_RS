"""Tests for consecutive-date change metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from csdv_core.zonal.deltas import CHANGE_METRICS, add_change_metrics, delta_columns


def _frame(rows: list[dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_delta_columns_names_every_change_metric():
    assert delta_columns() == [f"d_{name}" for name in CHANGE_METRICS]


def test_first_date_of_each_stand_is_nan():
    frame = _frame(
        [
            {"stand_id": "A", "year": 2018, "crown_fraction": 0.8},
            {"stand_id": "A", "year": 2020, "crown_fraction": 0.6},
            {"stand_id": "B", "year": 2018, "crown_fraction": 0.5},
        ]
    )
    out = add_change_metrics(frame)
    assert np.isnan(out.loc[0, "d_crown_fraction"])
    assert out.loc[1, "d_crown_fraction"] == pytest.approx(-0.2)
    assert np.isnan(out.loc[2, "d_crown_fraction"])


def test_differences_do_not_leak_between_stands():
    frame = _frame(
        [
            {"stand_id": "A", "year": 2022, "crown_fraction": 0.9},
            {"stand_id": "B", "year": 2018, "crown_fraction": 0.1},
            {"stand_id": "A", "year": 2018, "crown_fraction": 0.8},
        ]
    )
    out = add_change_metrics(frame)
    b_first = out[(out["stand_id"] == "B")].iloc[0]
    assert np.isnan(b_first["d_crown_fraction"])


def test_rows_are_ordered_by_date_within_a_stand():
    frame = _frame(
        [
            {"stand_id": "A", "year": 2022, "crown_fraction": 0.9},
            {"stand_id": "A", "year": 2018, "crown_fraction": 0.8},
        ]
    )
    out = add_change_metrics(frame)
    assert list(out["year"]) == [2018, 2022]
    assert out.loc[1, "d_crown_fraction"] == pytest.approx(0.1)


def test_a_clearcut_drives_crown_fraction_and_p90_down_together():
    frame = _frame(
        [
            {"stand_id": "A", "year": 2016, "crown_fraction": 0.90, "crown_p90": 9.0},
            {"stand_id": "A", "year": 2018, "crown_fraction": 0.05, "crown_p90": 2.0},
        ]
    )
    out = add_change_metrics(frame)
    assert out.loc[1, "d_crown_fraction"] < -0.5
    assert out.loc[1, "d_crown_p90"] < -5.0


def test_highgrading_moves_p90_without_moving_crown_fraction():
    """The separation Section 5 relies on: the largest trees leave, the canopy
    barely changes."""
    frame = _frame(
        [
            {"stand_id": "A", "year": 2016, "crown_fraction": 0.85, "crown_p90": 11.0},
            {"stand_id": "A", "year": 2018, "crown_fraction": 0.82, "crown_p90": 6.5},
        ]
    )
    out = add_change_metrics(frame)
    assert abs(out.loc[1, "d_crown_fraction"]) < 0.05
    assert out.loc[1, "d_crown_p90"] < -4.0


def test_missing_metric_gives_an_all_nan_column_rather_than_no_column():
    frame = _frame(
        [
            {"stand_id": "A", "year": 2018, "crown_fraction": 0.8},
            {"stand_id": "A", "year": 2020, "crown_fraction": 0.6},
        ]
    )
    out = add_change_metrics(frame)
    assert "d_crown_p90" in out.columns
    assert out["d_crown_p90"].isna().all()


def test_empty_frame_passes_through():
    assert add_change_metrics(pd.DataFrame()).empty
