"""End-to-end synthetic test: stage cube + metric cubes -> trajectory raster.

Builds an in-memory stages/trajectories config, runs the classifier, and
verifies code assignment for two rules with first-match-wins precedence.
"""

from __future__ import annotations

import numpy as np

from csdv_core.config._models import (
    StageEnvelopes,
    StagesConfig,
    TrajectoriesConfig,
    TrajectoryPredicate,
    TrajectoryRule,
)
from csdv_core.trajectories.classify import classify_trajectories

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


def _trajectories_cfg() -> TrajectoriesConfig:
    """Synthetic DS1-like and LC1-like rules."""
    ds1 = TrajectoryRule(
        name="DS1",
        group="DS",
        signature=[
            TrajectoryPredicate(
                dim="stage", reducer="all", op="in", value=["ESE", "LSE"]
            ),
            TrajectoryPredicate(
                dim="metric",
                var="crown_width_cv",
                reducer="max",
                op="<=",
                value=0.15,
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
    lc1 = TrajectoryRule(
        name="LC1",
        group="LC",
        signature=[
            TrajectoryPredicate(
                dim="metric",
                var="crown_fraction",
                reducer="max",
                op="<=",
                value=0.05,
            ),
            TrajectoryPredicate(
                dim="metric",
                var="gap_fraction",
                reducer="min",
                op=">=",
                value=0.85,
            ),
        ],
    )
    return TrajectoriesConfig(
        trajectories={"DS1": ds1, "LC1": lc1},
        trajectory_order=["DS1", "LC1"],
        trajectory_codes={"DS1": 1, "LC1": 9},
    )


def test_synthetic_two_rule_classification() -> None:
    n = 16
    rng = np.random.default_rng(0)

    # Build a (2, 16, 16) stage cube. Half the pixels are stably ESE/LSE
    # (DS1-eligible); the other half are LSE-LSE (RuleA stage qualifier).
    stage_cube = np.full((2, n, n), STAGE_CODES["LSE"], dtype="uint8")
    stage_cube[0, :, : n // 2] = STAGE_CODES["ESE"]

    # Metrics: left half has tiny CV and gap (DS1 fires);
    # right half is open (LC1 fires when crown_fraction is tiny).
    crown_cv = np.full((2, n, n), 0.05, dtype="float32")
    crown_cv[:, :, n // 2 :] = 0.40
    gap = np.full((2, n, n), 0.05, dtype="float32")
    gap[:, :, n // 2 :] = 0.90
    crown_fraction = np.full((2, n, n), 0.50, dtype="float32")
    crown_fraction[:, :, n // 2 :] = 0.02
    # Add some NaN noise to verify nan-safe behavior doesn't break classification
    nan_mask = rng.random((2, n, n)) < 0.05
    crown_cv = np.where(nan_mask, np.nan, crown_cv)

    traj, n_pred, n_dates = classify_trajectories(
        stage_cube,
        {
            "crown_width_cv": crown_cv,
            "gap_fraction": gap,
            "crown_fraction": crown_fraction,
        },
        None,
        _trajectories_cfg(),
        _stages_cfg(),
    )

    # Left half should be predominantly DS1 (code 1); right half should be
    # LC1 (code 9). NaN noise on the left can yield 0 at a few pixels.
    left = traj[:, : n // 2]
    right = traj[:, n // 2 :]
    assert (left == 1).sum() >= int(0.8 * left.size)
    assert (right == 9).all()
    assert n_pred[0, 0] in (0, 3)
    assert (n_dates == 2).all()
