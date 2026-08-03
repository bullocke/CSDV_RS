"""Tests for stand-level stage assignment."""

from __future__ import annotations

import pytest

from csdv_core.config._models import Range, StageEnvelopes, StagesConfig
from csdv_core.stages.stand import (
    UNSTRATIFIED_SITE_TYPE,
    classify_stand_sequence,
    classify_stand_stage,
    envelope_key,
)

KEY = UNSTRATIFIED_SITE_TYPE


def _cfg(envelopes: dict[str, dict[str, Range]]) -> StagesConfig:
    return StagesConfig(
        stages={
            code: StageEnvelopes(envelopes={KEY: env})
            for code, env in envelopes.items()
        },
        stage_order=list(envelopes),
        stage_codes={code: i + 1 for i, code in enumerate(envelopes)},
    )


@pytest.fixture()
def two_stages() -> StagesConfig:
    """An open stage and a closed stage, separated on two metrics."""
    return _cfg(
        {
            "ESI": {
                "gap_fraction": Range(min=0.75, max=1.00),
                "crown_cv": Range(min=0.00, max=0.10),
            },
            "LSE": {
                "gap_fraction": Range(min=0.05, max=0.20),
                "crown_cv": Range(min=0.10, max=0.25),
            },
        }
    )


def test_the_matching_stage_wins(two_stages):
    result = classify_stand_stage({"gap_fraction": 0.90, "crown_cv": 0.05}, two_stages)
    assert result.stage == "ESI"
    assert result.score == pytest.approx(1.0)
    assert result.n_evaluated == 2
    assert result.reason == ""


def test_ranked_carries_every_evaluable_stage(two_stages):
    result = classify_stand_stage({"gap_fraction": 0.90, "crown_cv": 0.05}, two_stages)
    assert [code for code, _, _ in result.ranked] == ["ESI", "LSE"]
    assert result.ranked[0][1] > result.ranked[1][1]


def test_missing_metrics_lower_the_evidence_not_the_score(two_stages):
    """A score of 1.0 on one metric is weaker than 1.0 on two, and the record
    has to make that visible."""
    partial = classify_stand_stage({"gap_fraction": 0.90}, two_stages)
    full = classify_stand_stage({"gap_fraction": 0.90, "crown_cv": 0.05}, two_stages)
    assert partial.score == pytest.approx(full.score)
    assert partial.n_evaluated == 1
    assert full.n_evaluated == 2


def test_nan_is_treated_as_missing(two_stages):
    result = classify_stand_stage(
        {"gap_fraction": 0.90, "crown_cv": float("nan")}, two_stages
    )
    assert result.n_evaluated == 1
    assert result.stage == "ESI"


def test_failed_metrics_are_named(two_stages):
    result = classify_stand_stage(
        {"gap_fraction": 0.90, "crown_cv": 0.50}, two_stages, min_score=0.5
    )
    assert result.stage == "ESI"
    assert result.failed_metrics == ("crown_cv",)


def test_below_the_minimum_score_nothing_is_assigned(two_stages):
    result = classify_stand_stage(
        {"gap_fraction": 0.50, "crown_cv": 0.50}, two_stages, min_score=0.5
    )
    assert result.stage is None
    assert "min_score" in result.reason
    # The ranking survives, so a figure can still show how close the call was.
    assert result.ranked


def test_a_tie_goes_to_the_earlier_stage_in_the_sequence():
    """ESE and UR carry identical envelopes in the current config, so UR can
    never be assigned. This pins that consequence rather than hiding it."""
    identical = {
        "gap_fraction": Range(min=0.20, max=0.45),
        "crown_cv": Range(min=0.25, max=0.40),
    }
    cfg = _cfg({"ESE": dict(identical), "UR": dict(identical)})
    result = classify_stand_stage({"gap_fraction": 0.30, "crown_cv": 0.30}, cfg)
    assert result.stage == "ESE"
    assert result.ranked[0][1] == result.ranked[1][1]


def test_an_unconstrained_metric_would_score_a_free_point():
    """Documents why unconstrained metrics are omitted from an envelope rather
    than written with null bounds: a null-null range matches anything."""
    cfg = _cfg(
        {
            "OPEN": {"gap_fraction": Range(min=0.75, max=1.00)},
            "ANY": {"gap_fraction": Range(min=None, max=None)},
        }
    )
    result = classify_stand_stage({"gap_fraction": 0.10}, cfg)
    assert result.stage == "ANY"
    assert result.score == pytest.approx(1.0)


def test_empty_envelope_is_skipped_not_scored():
    cfg = StagesConfig(
        stages={
            "ESI": StageEnvelopes(envelopes={KEY: {}}),
            "LSE": StageEnvelopes(
                envelopes={KEY: {"gap_fraction": Range(min=0.0, max=0.2)}}
            ),
        },
        stage_order=["ESI", "LSE"],
        stage_codes={"ESI": 1, "LSE": 4},
    )
    result = classify_stand_stage({"gap_fraction": 0.10}, cfg)
    assert result.stage == "LSE"
    assert [code for code, _, _ in result.ranked] == ["LSE"]


def test_missing_site_type_envelope_reports_a_reason(two_stages):
    result = classify_stand_stage(
        {"gap_fraction": 0.90}, two_stages, site_type_key="type_07"
    )
    assert result.stage is None
    assert "type_07" in result.reason


def test_no_evaluable_metric_reports_a_reason(two_stages):
    result = classify_stand_stage({"ndvi_trend": 0.4}, two_stages)
    assert result.stage is None
    assert result.reason == "no envelope metric was evaluable"


def test_envelope_key_formats_a_site_type_code():
    assert envelope_key(None) == "type_00"
    assert envelope_key(0) == "type_00"
    assert envelope_key(13) == "type_13"


def test_sequence_classifies_every_date(two_stages):
    results = classify_stand_sequence(
        [
            {"gap_fraction": 0.90, "crown_cv": 0.05},
            {"gap_fraction": 0.10, "crown_cv": 0.15},
        ],
        two_stages,
    )
    assert [r.stage for r in results] == ["ESI", "LSE"]


def test_production_config_assigns_a_stage_from_the_provisional_envelopes():
    """The packaged type_00 envelopes must actually be able to assign a stage,
    which the earlier all-null placeholders could not."""
    from csdv_core.config import load_stages

    result = classify_stand_stage(
        {
            "gap_fraction": 0.10,
            "crown_cv": 0.18,
            "glcm_texture": 3.0,
            "shrub_fraction": 0.05,
            "gap_persistence": 0.85,
        },
        load_stages(),
    )
    assert result.stage == "LSE"
    assert result.n_evaluated == 5
    assert result.score == pytest.approx(1.0)
