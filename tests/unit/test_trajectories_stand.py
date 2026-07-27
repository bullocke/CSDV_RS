"""Tests for stand-level trajectory classification.

The stand path shapes a stand as a one-pixel raster so it can reuse the raster
rule engine unchanged. These tests check that the reuse is faithful, and that a
rule which cannot fire is reported as such rather than silently producing no
result.
"""

from __future__ import annotations

import pandas as pd
import pytest

from csdv_core.config import load_stages, load_trajectories
from csdv_core.config._models import (
    StageEnvelopes,
    StagesConfig,
    TrajectoriesConfig,
    TrajectoryPredicate,
    TrajectoryRule,
)
from csdv_core.trajectories.sequences import build_sequence, stand_cubes
from csdv_core.trajectories.stand import (
    BLOCKED_EVALUATED_NO_MATCH,
    BLOCKED_METRIC_MISSING,
    BLOCKED_NO_THRESHOLD,
    blocking_report,
    classify_stand_trajectory,
    rule_is_fireable,
)

STAGES = StagesConfig(
    stages={code: StageEnvelopes(envelopes={}) for code in ("ESE", "LSE")},
    stage_order=["ESE", "LSE"],
    stage_codes={"ESE": 3, "LSE": 4},
)


def _frame(years: list[int], **columns: list[float]) -> pd.DataFrame:
    data = {"year": years, "date": [f"{y}-07-01" for y in years]}
    data.update(columns)
    return pd.DataFrame(data)


def _sequence(years: list[int], stages: list[str | None], **columns: list[float]):
    return build_sequence(
        "A", _frame(years, **columns), stages, metric_names=list(columns)
    )


def test_stand_cubes_have_one_pixel_and_one_plane_per_date():
    seq = _sequence([2016, 2018], ["ESE", "LSE"], gap_fraction=[0.3, 0.1])
    stage_cube, metric_cubes = stand_cubes(seq, STAGES)
    assert stage_cube.shape == (2, 1, 1)
    assert stage_cube[0, 0, 0] == 3
    assert stage_cube[1, 0, 0] == 4
    assert metric_cubes["gap_fraction"].shape == (2, 1, 1)


def test_unclassified_dates_become_stage_code_zero():
    seq = _sequence([2016, 2018], [None, "LSE"], gap_fraction=[0.3, 0.1])
    stage_cube, _ = stand_cubes(seq, STAGES)
    assert stage_cube[0, 0, 0] == 0


def test_build_sequence_pads_a_metric_the_frame_does_not_carry():
    seq = build_sequence(
        "A",
        _frame([2016, 2018], gap_fraction=[0.3, 0.1]),
        ["ESE", "LSE"],
        metric_names=["gap_fraction", "ndvi_trend"],
    )
    assert seq.metrics["ndvi_trend"] == (pytest.approx(float("nan"), nan_ok=True),) * 2


def test_build_sequence_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="2 stages for 3 dates"):
        build_sequence(
            "A",
            _frame([2016, 2018, 2020], gap_fraction=[0.3, 0.2, 0.1]),
            ["ESE", "LSE"],
            metric_names=["gap_fraction"],
        )


# A complete, fireable rule shaped like DS1: stem exclusion throughout, with
# uniform crowns and a closed canopy.
_UNIFORM = TrajectoriesConfig(
    trajectory_order=["DS1"],
    trajectory_codes={"DS1": 1},
    trajectories={
        "DS1": TrajectoryRule(
            name="Beech bark disease sprout trap",
            group="DS",
            signature=[
                TrajectoryPredicate(
                    dim="stage", reducer="all", op="in", value=["ESE", "LSE"]
                ),
                TrajectoryPredicate(
                    dim="metric", var="crown_cv", reducer="max", op="<=", value=0.15
                ),
                TrajectoryPredicate(
                    dim="metric", var="gap_fraction", reducer="max", op="<=", value=0.10
                ),
            ],
        )
    },
)


def test_a_complete_rule_fires_on_matching_values():
    """Proves the one-pixel cube really does drive the shared rule engine."""
    seq = _sequence(
        [2016, 2018, 2020],
        ["ESE", "LSE", "LSE"],
        crown_cv=[0.10, 0.12, 0.11],
        gap_fraction=[0.05, 0.06, 0.04],
    )
    result = classify_stand_trajectory(seq, _UNIFORM, STAGES)
    assert result.code == "DS1"
    assert result.name == "Beech bark disease sprout trap"
    assert result.n_predicates == 3
    assert result.n_dates == 3


def test_a_complete_rule_does_not_fire_when_a_condition_fails():
    seq = _sequence(
        [2016, 2018, 2020],
        ["ESE", "LSE", "LSE"],
        crown_cv=[0.10, 0.45, 0.11],  # too variable at one date
        gap_fraction=[0.05, 0.06, 0.04],
    )
    result = classify_stand_trajectory(seq, _UNIFORM, STAGES)
    assert result.code is None
    assert result.blocked["DS1"] == (BLOCKED_EVALUATED_NO_MATCH,)


def test_a_stage_precondition_can_block_a_match():
    seq = _sequence(
        [2016, 2018],
        ["ESE", None],  # one date never got a stage
        crown_cv=[0.10, 0.11],
        gap_fraction=[0.05, 0.04],
    )
    result = classify_stand_trajectory(seq, _UNIFORM, STAGES)
    assert result.code is None
    assert result.n_dates == 1


def test_a_rule_with_an_unfilled_threshold_is_reported_as_inert():
    cfg = TrajectoriesConfig(
        trajectory_order=["EF3"],
        trajectory_codes={"EF3": 8},
        trajectories={
            "EF3": TrajectoryRule(
                name="Unknown cause",
                group="EF",
                signature=[
                    TrajectoryPredicate(
                        dim="metric",
                        var="gap_fraction",
                        reducer="min",
                        op=">=",
                        value=None,
                    )
                ],
            )
        },
    )
    fireable, reasons = rule_is_fireable(cfg.trajectories["EF3"], ["gap_fraction"])
    assert not fireable
    assert reasons == (f"{BLOCKED_NO_THRESHOLD}: gap_fraction",)


def test_a_rule_naming_an_unavailable_metric_is_reported_as_inert():
    cfg = TrajectoriesConfig(
        trajectory_order=["LC1"],
        trajectory_codes={"LC1": 9},
        trajectories={
            "LC1": TrajectoryRule(
                name="Agriculture",
                group="LC",
                signature=[
                    TrajectoryPredicate(
                        dim="metric",
                        var="ndvi_seasonal_amplitude",
                        reducer="max",
                        op=">=",
                        value=0.4,
                    )
                ],
            )
        },
    )
    report = blocking_report(cfg, ["gap_fraction"])
    assert report["LC1"] == (f"{BLOCKED_METRIC_MISSING}: ndvi_seasonal_amplitude",)


def test_one_date_cannot_produce_a_trajectory():
    seq = _sequence([2018], ["LSE"], gap_fraction=[0.1])
    with pytest.raises(ValueError, match="needs at least two"):
        classify_stand_trajectory(seq, _UNIFORM, STAGES)


def test_production_config_blocks_most_rules():
    """The state of the specification, pinned as a test.

    Only a handful of the nineteen trajectory classes carry complete
    thresholds, and the examples must report that rather than imply the layer
    is working. If a rule is completed later this test should be updated with
    the change, deliberately.
    """
    traj_cfg = load_trajectories()
    available = [
        "gap_fraction",
        "crown_fraction",
        "crown_cv",
        "crown_p90",
        "glcm_texture",
        "shrub_fraction",
        "gap_persistence",
        "linearity_index",
        "row_directionality",
    ]
    report = blocking_report(traj_cfg, available)
    n_rules = len(traj_cfg.trajectory_order)
    assert n_rules == 19
    assert len(report) >= 16
    # Every block has a stated reason.
    assert all(reasons for reasons in report.values())


def test_production_config_runs_end_to_end_without_raising():
    """A stand carrying only the implemented metrics must classify cleanly,
    including against rules that reference metrics that do not exist."""
    seq = build_sequence(
        "ELKNE-U48-0-0",
        _frame(
            [2016, 2018, 2020],
            gap_fraction=[0.55, 0.40, 0.28],
            crown_fraction=[0.45, 0.60, 0.72],
            crown_cv=[0.22, 0.28, 0.31],
            glcm_texture=[5.4, 5.1, 4.8],
            shrub_fraction=[0.22, 0.14, 0.08],
            gap_persistence=[float("nan"), 0.35, 0.30],
        ),
        ["LSI", "ESE", "ESE"],
        metric_names=[
            "gap_fraction",
            "crown_fraction",
            "crown_cv",
            "glcm_texture",
            "shrub_fraction",
            "gap_persistence",
        ],
    )
    result = classify_stand_trajectory(seq, load_trajectories(), load_stages())
    assert result.n_dates == 3
    assert isinstance(result.blocked, dict)
    assert len(result.blocked) >= 16
