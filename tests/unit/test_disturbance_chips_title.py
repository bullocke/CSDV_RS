"""Tests for the chip figure title of csdv_core.disturbance.chips."""

from __future__ import annotations

import numpy as np
import pandas as pd

from csdv_core.disturbance.chips import _unit_title


def _sample(**extra: object) -> pd.Series:
    """One sample row with the columns the title reads.

    Args:
        **extra: Additional or replaced columns.

    Returns:
        The sample row.
    """
    values: dict[str, object] = {
        "sample_id": "S01",
        "stratum": "detected",
        "interval": "2021-2023",
        "area_ha": 0.81,
        "max_prob": 0.92,
        "object_id": 4,
        "centroid_lat": 44.1,
        "centroid_lon": -71.6,
    }
    values.update(extra)
    return pd.Series(values)


def test_title_flags_missing_follow_up() -> None:
    sample = _sample(follow_up=False)
    title = _unit_title(sample)
    assert title.endswith("no follow-up image")


def test_title_without_flag_when_follow_up() -> None:
    sample = _sample(follow_up=True)
    title = _unit_title(sample)
    assert "follow-up" not in title


def test_title_handles_absent_or_missing_follow_up() -> None:
    absent_sample = _sample()
    missing_sample = _sample(follow_up=np.nan)
    absent_title = _unit_title(absent_sample)
    missing_title = _unit_title(missing_sample)
    assert "follow-up" not in absent_title
    assert "follow-up" not in missing_title
