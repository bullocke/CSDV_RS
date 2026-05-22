"""Tests for :mod:`csdv_core.stages.classify`."""

from __future__ import annotations

import numpy as np
import pytest

from csdv_core.config._models import Range, StageEnvelopes, StagesConfig
from csdv_core.stages import classify


def _cfg() -> StagesConfig:
    """Two stages, one site type. ESI = high gap; LSE = low gap."""
    return StagesConfig(
        stages={
            "ESI": StageEnvelopes(
                envelopes={
                    "type_01": {
                        "gap_fraction": Range(min=0.5, max=1.0),
                        "crown_width_cv": Range(min=0.5, max=2.0),
                    }
                }
            ),
            "LSE": StageEnvelopes(
                envelopes={
                    "type_01": {
                        "gap_fraction": Range(min=0.0, max=0.1),
                        "crown_width_cv": Range(min=0.0, max=0.3),
                    }
                }
            ),
        },
        stage_order=["ESI", "LSE"],
        stage_codes={"ESI": 1, "LSE": 4},
    )


def test_classify_basic_assignment() -> None:
    site_type = np.array([[1, 1]], dtype="uint8")
    metrics = {
        "gap_fraction": np.array([[0.05, 0.7]], dtype="float32"),
        "crown_width_cv": np.array([[0.2, 1.0]], dtype="float32"),
    }
    stage, score, evaluated = classify.classify_stages(
        metrics, site_type, _cfg(), min_score=0.5
    )
    # (0,0) low gap, low cv -> LSE (code 4)
    assert stage[0, 0] == 4
    # (0,1) high gap, high cv -> ESI (code 1)
    assert stage[0, 1] == 1
    assert score[0, 0] == pytest.approx(1.0)
    assert evaluated[0, 0] == 2


def test_classify_unclassified_site_type_is_zero() -> None:
    site_type = np.array([[0]], dtype="uint8")
    metrics = {
        "gap_fraction": np.array([[0.05]], dtype="float32"),
        "crown_width_cv": np.array([[0.2]], dtype="float32"),
    }
    stage, score, evaluated = classify.classify_stages(
        metrics, site_type, _cfg(), min_score=0.5
    )
    assert stage[0, 0] == 0
    assert np.isnan(score[0, 0])
    assert evaluated[0, 0] == 0


def test_classify_below_min_score_is_zero() -> None:
    site_type = np.array([[1]], dtype="uint8")
    # Values match neither envelope.
    metrics = {
        "gap_fraction": np.array([[0.3]], dtype="float32"),
        "crown_width_cv": np.array([[0.4]], dtype="float32"),
    }
    stage, _, _ = classify.classify_stages(metrics, site_type, _cfg(), min_score=0.5)
    assert stage[0, 0] == 0


def test_classify_tie_break_uses_stage_order() -> None:
    """When scores tie, earlier in stage_order should win (no override)."""
    cfg = StagesConfig(
        stages={
            "ESI": StageEnvelopes(
                envelopes={"type_01": {"a": Range(min=0.0, max=1.0)}}
            ),
            "LSE": StageEnvelopes(
                envelopes={"type_01": {"a": Range(min=0.0, max=1.0)}}
            ),
        },
        stage_order=["ESI", "LSE"],
        stage_codes={"ESI": 1, "LSE": 4},
    )
    site_type = np.array([[1]], dtype="uint8")
    metrics = {"a": np.array([[0.5]], dtype="float32")}
    stage, _, _ = classify.classify_stages(metrics, site_type, cfg, min_score=0.5)
    # On a strict tie (>) my impl keeps the first; ESI (code 1) wins.
    assert stage[0, 0] == 1


def test_classify_shape_mismatch_raises() -> None:
    site_type = np.zeros((2, 2), dtype="uint8")
    metrics = {"gap_fraction": np.zeros((3, 3), dtype="float32")}
    with pytest.raises(ValueError, match="shape"):
        classify.classify_stages(metrics, site_type, _cfg())


def test_classify_empty_metrics_raises() -> None:
    with pytest.raises(ValueError):
        classify.classify_stages({}, np.zeros((2, 2), dtype="uint8"), _cfg())
