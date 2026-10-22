"""Unit tests for csdv_core.national.cost_model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from csdv_core.national.cost_model import (
    CensusCalibration,
    expand_total,
    fit_block_cost,
    fit_census_calibration,
    fit_ratio,
    predict_tiles,
    wall_days,
)


def _blocks() -> pd.DataFrame:
    canopy = np.array([0.0, 0.2e6, 0.5e6, 1.0e6, 0.8e6])
    return pd.DataFrame(
        {
            "engine": "watershed",
            "canopy_px": canopy,
            "compute_s": 0.1 + 0.9 * canopy / 1e6,
            "error": "",
            "block_px": 1024,
        }
    )


def test_fit_block_cost_recovers_line_and_skips_empty_blocks() -> None:
    model = fit_block_cost(_blocks(), "watershed")
    assert model.intercept_s == pytest.approx(0.1)
    assert model.per_canopy_mpx_s == pytest.approx(0.9)
    assert model.n == 4
    assert model.block_px == 1024


def test_fit_ratio_through_origin() -> None:
    assert fit_ratio(pd.Series([2.0, 4.0]), pd.Series([1.0, 2.0])) == pytest.approx(2.0)


def test_census_calibration_clips_to_unit_interval() -> None:
    tiles = pd.DataFrame(
        {"frac_ge2": [0.1, 0.5, 0.9], "frac_ge2_full": [0.05, 0.45, 0.85]}
    )
    cal = fit_census_calibration(tiles)
    assert cal.slope == pytest.approx(1.0)
    assert cal.intercept == pytest.approx(-0.05)
    assert cal(np.array([0.0]))[0] == 0.0


def test_predict_tiles_counts_blocks_per_engine() -> None:
    census = pd.DataFrame(
        {
            "full_width": [2048],
            "full_height": [2048],
            "valid_frac": [1.0],
            "frac_ge2": [0.5],
        }
    )
    model = fit_block_cost(_blocks(), "watershed")
    out = predict_tiles(
        census,
        cost={"watershed": model},
        calibration=CensusCalibration(0.0, 1.0, 1.0, 3),
        crowns_per_canopy_mpx=1000.0,
    )
    canopy = 2048 * 2048 * 0.5
    assert out["canopy_px"].iloc[0] == pytest.approx(canopy)
    assert out["n_crowns"].iloc[0] == pytest.approx(canopy / 1e3)
    # 2048 px tile in 1024 px blocks is 4 blocks.
    assert out["watershed_s"].iloc[0] == pytest.approx(4 * 0.1 + 0.9 * canopy / 1e6)


def test_expand_total_scales_sample_sum() -> None:
    tot = expand_total(pd.Series([1.0, 2.0, 3.0, 4.0]), 40, n_boot=200)
    assert tot.total == pytest.approx(100.0)
    assert tot.lo <= tot.total <= tot.hi


def test_wall_days() -> None:
    assert wall_days(24.0 * 100, 100, efficiency=1.0) == pytest.approx(1.0)
