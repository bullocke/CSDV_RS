"""Tests for csdv_core.trajectories.classify: per-pixel engine."""

from __future__ import annotations

import numpy as np
import pytest

from csdv_core.config._models import (
    StageEnvelopes,
    StagesConfig,
    TrajectoriesConfig,
    TrajectoryPredicate,
    TrajectoryRule,
)
from csdv_core.trajectories.classify import (
    all_required_metrics,
    classify_trajectories,
)

STAGE_CODES = {"ESE": 3, "LSE": 4}


def _stages_cfg() -> StagesConfig:
    return StagesConfig(
        stages={
            "ESE": StageEnvelopes(envelopes={}),
            "LSE": StageEnvelopes(envelopes={}),
        },
        stage_order=["ESE", "LSE"],
        stage_codes=STAGE_CODES,
    )


def _two_rule_cfg() -> TrajectoriesConfig:
    """RuleA: stage all in (ESE, LSE) AND gap<=0.10. RuleB: stage all == LSE."""
    rule_a = TrajectoryRule(
        name="A",
        group="DS",
        signature=[
            TrajectoryPredicate(
                dim="stage", reducer="all", op="in", value=["ESE", "LSE"]
            ),
            TrajectoryPredicate(
                dim="metric",
                var="gap_fraction",
                reducer="max",
                op="<=",
                value=0.10,
            ),
        ],
    )
    rule_b = TrajectoryRule(
        name="B",
        group="LC",
        signature=[
            TrajectoryPredicate(dim="stage", reducer="all", op="==", value="LSE"),
        ],
    )
    return TrajectoriesConfig(
        trajectories={"A": rule_a, "B": rule_b},
        trajectory_order=["A", "B"],
        trajectory_codes={"A": 1, "B": 2},
    )


def _stage_cube() -> np.ndarray:
    """(2, 2, 2). Year-0: all ESE. Year-1: all LSE."""
    return np.stack(
        [
            np.full((2, 2), STAGE_CODES["ESE"], dtype="uint8"),
            np.full((2, 2), STAGE_CODES["LSE"], dtype="uint8"),
        ],
        axis=0,
    )


def test_first_match_wins_precedence() -> None:
    """At pixels where both A and B match, A (first) should win."""
    # Stage cube: year-0 LSE, year-1 LSE -> RuleB matches everywhere.
    cube = np.full((2, 2, 2), STAGE_CODES["LSE"], dtype="uint8")
    # Gap: pixel (0,0) has small max so RuleA matches; others don't.
    gap = np.zeros((2, 2, 2), dtype="float32")
    gap[:, 0, 0] = 0.05
    gap[:, 0, 1] = 0.5
    gap[:, 1, 0] = 0.5
    gap[:, 1, 1] = 0.5

    traj, n_pred, n_dates = classify_trajectories(
        cube, {"gap_fraction": gap}, None, _two_rule_cfg(), _stages_cfg()
    )
    assert traj[0, 0] == 1  # RuleA wins
    # Other pixels: RuleA fails; RuleB matches (stage all LSE) -> code 2.
    assert traj[0, 1] == 2
    assert traj[1, 0] == 2
    assert traj[1, 1] == 2
    # n_predicates: RuleA evaluated 2; RuleB evaluated 1.
    assert n_pred[0, 0] == 2
    assert n_pred[0, 1] == 1
    # n_dates: both dates non-zero -> 2 everywhere.
    assert (n_dates == 2).all()


def test_unclassified_default_zero() -> None:
    cfg = TrajectoriesConfig(
        trajectories={
            "A": TrajectoryRule(
                name="A",
                group="DS",
                signature=[
                    TrajectoryPredicate(
                        dim="metric", var="x", reducer="max", op=">=", value=99.0
                    )
                ],
            ),
        },
        trajectory_order=["A"],
        trajectory_codes={"A": 1},
    )
    cube = _stage_cube()
    metrics = {"x": np.zeros((2, 2, 2), dtype="float32")}
    traj, n_pred, n_dates = classify_trajectories(
        cube, metrics, None, cfg, _stages_cfg()
    )
    assert (traj == 0).all()
    assert (n_pred == 0).all()
    assert (n_dates == 2).all()


def test_n_dates_counts_nonzero_stages() -> None:
    cube = _stage_cube().copy()
    cube[0, 0, 0] = 0  # one missing date at one pixel
    traj, _, n_dates = classify_trajectories(
        cube,
        {"gap_fraction": np.zeros((2, 2, 2), dtype="float32")},
        None,
        _two_rule_cfg(),
        _stages_cfg(),
    )
    assert n_dates[0, 0] == 1
    assert n_dates[0, 1] == 2


def test_shape_validation() -> None:
    bad = np.zeros((2, 2), dtype="uint8")
    with pytest.raises(ValueError, match="must be"):
        classify_trajectories(bad, {}, None, _two_rule_cfg(), _stages_cfg())


def test_metric_shape_mismatch_raises() -> None:
    cube = _stage_cube()
    bad = np.zeros((3, 3, 3), dtype="float32")
    with pytest.raises(ValueError, match="shape"):
        classify_trajectories(
            cube, {"gap_fraction": bad}, None, _two_rule_cfg(), _stages_cfg()
        )


def test_all_required_metrics_dedupes() -> None:
    cfg = _two_rule_cfg()
    assert all_required_metrics(cfg) == ["gap_fraction"]
