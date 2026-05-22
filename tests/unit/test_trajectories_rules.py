"""Tests for csdv_core.trajectories.rules: predicate evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from csdv_core.config._models import TrajectoryPredicate, TrajectoryRule
from csdv_core.trajectories.rules import (
    evaluate_predicate,
    evaluate_rule,
    required_metrics,
)

STAGE_CODES = {
    "ESI": 1,
    "LSI": 2,
    "ESE": 3,
    "LSE": 4,
    "UR": 5,
    "MA_OW": 6,
    "OG": 7,
}


def _stage_cube_t2() -> np.ndarray:
    """(2, 2, 2) cube with year-0 = ESE everywhere, year-1 = LSE everywhere."""
    a = np.full((2, 2), STAGE_CODES["ESE"], dtype="uint8")
    b = np.full((2, 2), STAGE_CODES["LSE"], dtype="uint8")
    return np.stack([a, b], axis=0)


def test_predicate_stage_in_all_dates() -> None:
    pred = TrajectoryPredicate(
        dim="stage", reducer="all", op="in", value=["ESE", "LSE"]
    )
    out = evaluate_predicate(
        pred,
        stage_cube=_stage_cube_t2(),
        metric_cubes={},
        site_type=None,
        stage_codes=STAGE_CODES,
    )
    assert out.shape == (2, 2)
    assert out.all()


def test_predicate_stage_in_excludes_other_codes() -> None:
    cube = _stage_cube_t2().copy()
    cube[1, 0, 0] = STAGE_CODES["MA_OW"]
    pred = TrajectoryPredicate(
        dim="stage", reducer="all", op="in", value=["ESE", "LSE"]
    )
    out = evaluate_predicate(
        pred, stage_cube=cube, metric_cubes={}, site_type=None, stage_codes=STAGE_CODES
    )
    assert out[0, 0] == False  # noqa: E712
    assert out[0, 1] == True  # noqa: E712


def test_predicate_stage_latest_eq() -> None:
    pred = TrajectoryPredicate(dim="stage", reducer="latest", op="==", value="LSE")
    out = evaluate_predicate(
        pred,
        stage_cube=_stage_cube_t2(),
        metric_cubes={},
        site_type=None,
        stage_codes=STAGE_CODES,
    )
    assert out.all()


def test_predicate_metric_max_le() -> None:
    cube = np.array([[[0.05, 0.20], [0.10, 0.30]], [[0.07, 0.25], [0.05, 0.20]]])
    pred = TrajectoryPredicate(
        dim="metric", var="gap_fraction", reducer="max", op="<=", value=0.10
    )
    out = evaluate_predicate(
        pred,
        stage_cube=_stage_cube_t2(),
        metric_cubes={"gap_fraction": cube.astype("float32")},
        site_type=None,
        stage_codes=STAGE_CODES,
    )
    # max over time at each pixel; only (0,0) and (1,0) <= 0.10
    expected = np.array([[True, False], [True, False]])
    np.testing.assert_array_equal(out, expected)


def test_predicate_metric_nan_is_false() -> None:
    cube = np.full((2, 2, 2), np.nan, dtype="float32")
    pred = TrajectoryPredicate(
        dim="metric", var="x", reducer="max", op="<=", value=0.10
    )
    out = evaluate_predicate(
        pred,
        stage_cube=_stage_cube_t2(),
        metric_cubes={"x": cube},
        site_type=None,
        stage_codes=STAGE_CODES,
    )
    assert not out.any()


def test_predicate_metric_all_reducer() -> None:
    """`all` reduces a per-date boolean mask, treating NaN as False."""
    cube = np.array([[[0.1, 0.5], [0.1, 0.1]], [[0.1, 0.5], [0.1, np.nan]]])
    pred = TrajectoryPredicate(dim="metric", var="g", reducer="all", op="<=", value=0.2)
    out = evaluate_predicate(
        pred,
        stage_cube=_stage_cube_t2(),
        metric_cubes={"g": cube.astype("float32")},
        site_type=None,
        stage_codes=STAGE_CODES,
    )
    # (0,0): both 0.1 -> True; (0,1): 0.5 fails; (1,0): both 0.1 -> True;
    # (1,1): nan fails.
    expected = np.array([[True, False], [True, False]])
    np.testing.assert_array_equal(out, expected)


def test_predicate_persistence() -> None:
    """Persistence: fraction of dates with metric >= threshold must >= value."""
    cube = np.array(
        [[[0.9, 0.4], [0.9, 0.9]], [[0.9, 0.9], [0.4, 0.9]]], dtype="float32"
    )
    pred = TrajectoryPredicate(
        dim="persistence",
        var="gap_fraction",
        reducer="scalar",
        op=">=",
        value=1.0,
        threshold=0.85,
    )
    out = evaluate_predicate(
        pred,
        stage_cube=_stage_cube_t2(),
        metric_cubes={"gap_fraction": cube},
        site_type=None,
        stage_codes=STAGE_CODES,
    )
    # (0,0): 2/2 >=1.0 True; (0,1): 1/2 fails; (1,0): 1/2 fails; (1,1): 2/2 True.
    expected = np.array([[True, False], [False, True]])
    np.testing.assert_array_equal(out, expected)


def test_predicate_site_type_eq() -> None:
    site = np.array([[1, 6], [6, 7]], dtype="uint8")
    pred = TrajectoryPredicate(dim="site_type", reducer="scalar", op="==", value=6)
    out = evaluate_predicate(
        pred,
        stage_cube=_stage_cube_t2(),
        metric_cubes={},
        site_type=site,
        stage_codes=STAGE_CODES,
    )
    np.testing.assert_array_equal(out, np.array([[False, True], [True, False]]))


def test_predicate_placeholder_value_none_is_false() -> None:
    pred = TrajectoryPredicate(
        dim="metric", var="x", reducer="max", op="<=", value=None
    )
    out = evaluate_predicate(
        pred,
        stage_cube=_stage_cube_t2(),
        metric_cubes={"x": np.zeros((2, 2, 2), dtype="float32")},
        site_type=None,
        stage_codes=STAGE_CODES,
    )
    assert not out.any()


def test_evaluate_rule_and_combines() -> None:
    rule = TrajectoryRule(
        name="t",
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
    cube = np.array([[[0.05, 0.20], [0.10, 0.30]], [[0.07, 0.25], [0.05, 0.20]]])
    mask, n = evaluate_rule(
        rule,
        stage_cube=_stage_cube_t2(),
        metric_cubes={"gap_fraction": cube.astype("float32")},
        site_type=None,
        stage_codes=STAGE_CODES,
    )
    assert n == 2
    np.testing.assert_array_equal(mask, np.array([[True, False], [True, False]]))


def test_evaluate_rule_empty_signature_never_fires() -> None:
    rule = TrajectoryRule(name="t", group="DS", signature=[])
    mask, n = evaluate_rule(
        rule,
        stage_cube=_stage_cube_t2(),
        metric_cubes={},
        site_type=None,
        stage_codes=STAGE_CODES,
    )
    assert n == 0
    assert not mask.any()


def test_required_metrics_dedupes_by_var() -> None:
    rule = TrajectoryRule(
        name="t",
        group="DS",
        signature=[
            TrajectoryPredicate(dim="stage", reducer="all", op="in", value=["ESE"]),
            TrajectoryPredicate(
                dim="metric", var="g", reducer="max", op="<=", value=0.1
            ),
            TrajectoryPredicate(
                dim="persistence",
                var="g",
                reducer="scalar",
                op=">=",
                value=1.0,
                threshold=0.5,
            ),
            TrajectoryPredicate(
                dim="metric", var="h", reducer="max", op="<=", value=0.1
            ),
        ],
    )
    assert required_metrics(rule) == ["g", "g", "h"]


def test_predicate_metric_unknown_var_raises() -> None:
    pred = TrajectoryPredicate(
        dim="metric", var="missing", reducer="max", op="<=", value=0.1
    )
    with pytest.raises(KeyError):
        evaluate_predicate(
            pred,
            stage_cube=_stage_cube_t2(),
            metric_cubes={},
            site_type=None,
            stage_codes=STAGE_CODES,
        )
