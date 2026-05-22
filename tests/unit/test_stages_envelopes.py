"""Tests for :mod:`csdv_core.stages.envelopes`."""

from __future__ import annotations

import numpy as np
import pytest

from csdv_core.config._models import Range
from csdv_core.stages import envelopes


def test_evaluate_envelope_inside_range() -> None:
    assert envelopes.evaluate_envelope(0.5, Range(min=0.0, max=1.0))


def test_evaluate_envelope_outside_range() -> None:
    assert not envelopes.evaluate_envelope(2.0, Range(min=0.0, max=1.0))
    assert not envelopes.evaluate_envelope(-1.0, Range(min=0.0, max=1.0))


def test_evaluate_envelope_unconstrained_bound() -> None:
    assert envelopes.evaluate_envelope(99.0, Range(min=0.0, max=None))
    assert envelopes.evaluate_envelope(-99.0, Range(min=None, max=10.0))


def test_evaluate_envelope_nan_false() -> None:
    assert not envelopes.evaluate_envelope(float("nan"), Range(min=0.0, max=1.0))


def test_match_stage_basic() -> None:
    envs = {
        "gap_fraction": Range(min=0.0, max=0.1),
        "crown_width_cv": Range(min=0.0, max=0.5),
    }
    m = envelopes.match_stage({"gap_fraction": 0.05, "crown_width_cv": 0.3}, envs)
    assert m.n_evaluated == 2
    assert m.n_matched == 2
    assert m.score == pytest.approx(1.0)
    assert m.failed_metrics == ()


def test_match_stage_partial_failure() -> None:
    envs = {
        "gap_fraction": Range(min=0.0, max=0.1),
        "crown_width_cv": Range(min=0.0, max=0.5),
    }
    m = envelopes.match_stage({"gap_fraction": 0.5, "crown_width_cv": 0.3}, envs)
    assert m.n_evaluated == 2
    assert m.n_matched == 1
    assert m.score == pytest.approx(0.5)
    assert m.failed_metrics == ("gap_fraction",)


def test_match_stage_nan_skipped_from_evaluation() -> None:
    envs = {
        "a": Range(min=0.0, max=1.0),
        "b": Range(min=0.0, max=1.0),
    }
    m = envelopes.match_stage({"a": 0.5, "b": float("nan")}, envs)
    assert m.n_evaluated == 1
    assert m.n_matched == 1
    assert m.score == pytest.approx(1.0)


def test_match_stage_nothing_evaluable() -> None:
    envs = {"a": Range(min=0.0, max=1.0)}
    m = envelopes.match_stage({"a": np.nan}, envs)
    assert m.n_evaluated == 0
    assert m.score == 0.0
