"""Routing of two verdicts into outcomes per interval."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from csdv_core.config._models import LabelRouting
from csdv_core.label import routing as R
from csdv_core.label.verdicts import Verdict, parse_verdict

CFG = LabelRouting()
FIXTURE = Path(__file__).parents[1] / "fixtures" / "label_probe_states.json"


def verdict(
    states: str,
    confidence: int = 3,
    change_type: str | None = None,
    fit: str | None = None,
    share: str | None = None,
) -> Verdict:
    names = {"f": "forest", "p": "partial_loss", "x": "full_loss", "o": "open"}
    return Verdict(
        model="m",
        states=[names[c] for c in states],
        confidence=confidence,
        change_type=change_type,
        outline_fit=fit,
        affected_share=share,
        valid=True,
    )


def test_accepted_same_first_date() -> None:
    a = verdict("ffxoo", change_type="clearcut")
    b = verdict("ffxoo", change_type="clearcut", confidence=2)
    out = R.route(a, b, 1, CFG)
    assert out.status == "accepted"
    assert out.outcomes == [R.INTACT, R.LOSS_FULL, R.OPEN, R.OPEN]
    assert out.first_loss == 1 and out.severity == "stand_replacing"
    assert out.change_type == "clearcut" and out.agent == "harvest"
    assert out.confidence == 2


def test_severity_dispute_and_later_loss_by_one_model() -> None:
    a = verdict("fxooo", change_type="fire")
    b = verdict("fpxoo", change_type="fire")
    out = R.route(a, b, 0, CFG)
    assert out.status == "accepted"
    assert out.outcomes == [R.LOSS_DISPUTED, R.DISAGREE, R.OPEN, R.OPEN]
    assert out.severity == "partial"


def test_types_differ_but_agent_agrees() -> None:
    a = verdict("fpo", change_type="thinning")
    b = verdict("fpo", change_type="partial_harvest")
    out = R.route(a, b, 0, CFG)
    assert out.change_type is None and out.agent == "harvest"
    assert out.outcomes[0] == R.LOSS_PARTIAL


def test_date_disputed_keeps_the_rest() -> None:
    a = verdict("ffxoo")
    b = verdict("fffxo")
    out = R.route(a, b, 1, CFG)
    assert out.status == "date_disputed"
    assert out.outcomes == [R.INTACT, R.DATE_DISPUTED, R.DATE_DISPUTED, R.OPEN]
    assert out.first_loss is None and not out.has_loss


def test_rejection_is_unlabelled_only_where_flagged() -> None:
    out = R.route(verdict("ffff"), verdict("ffff"), 2, CFG)
    assert out.status == "rejected"
    assert out.outcomes == [R.INTACT, R.INTACT, R.REJECTED_FLAGGED]
    weak = R.route(verdict("ffff"), verdict("ffff", confidence=2), 0, CFG)
    assert weak.status == "rejected_weak"
    assert weak.outcomes == [R.REJECTED_WEAK, R.INTACT, R.INTACT]
    # Ground that was never forest is "open" on every date.
    field = R.route(verdict("oooo"), verdict("oofo"), None, CFG)
    assert field.outcomes == [R.OPEN, R.OPEN, R.OPEN]


def test_disagreement_is_per_interval() -> None:
    out = R.route(verdict("ffxo"), verdict("ffff"), 1, CFG)
    assert out.status == "disagree"
    assert out.outcomes == [R.INTACT, R.DISAGREE, R.OPEN]


def test_unusable_verdict_fails_every_interval() -> None:
    bad = Verdict(model="m")
    out = R.route(verdict("ffxo"), bad, 1, CFG)
    assert out.status == "failed" and out.outcomes == [R.FAILED] * 3


def test_unjudged_touches_only_the_flagged_interval() -> None:
    small = R.unjudged(3, 1, small=True)
    assert small.outcomes == ["", R.UNDER_MIN, ""]
    large = R.unjudged(3, 2, small=False)
    assert large.outcomes == ["", "", R.UNJUDGED]


def test_parse_verdict_rejects_wrong_length_or_state() -> None:
    good = {
        "states": ["forest", "full_loss"],
        "confidence": "3",
        "change_type": "clearcut",
    }
    assert parse_verdict("m", good, 2).valid
    assert parse_verdict("m", good, 2).confidence == 3
    assert not parse_verdict("m", good, 3).valid
    assert not parse_verdict("m", {"states": ["forest", "gone"]}, 2).valid
    assert not parse_verdict("m", None, 2).valid


def test_probe_answers_under_both_rules() -> None:
    """The 82 probe sites: the first wording left 18% unlabelled."""
    sites = json.loads(FIXTURE.read_text())
    assert len(sites) == 82
    literal: Counter = Counter()
    status: Counter = Counter()
    real_in_rejected = 0
    for site in sites:
        a = parse_verdict("gem37", site["gem37"], site["n_dates"])
        b = parse_verdict("terra", site["terra"], site["n_dates"])
        literal[R.plan_literal(a, b, CFG)] += 1
        routed = R.route(a, b, None, CFG)
        status[routed.status] += 1
        if routed.status == "rejected" and site["truth"] == "disturbance":
            real_in_rejected += 1
    assert literal == {"accepted": 31, "rejected": 36, "ignore": 15}
    assert status == {
        "accepted": 31,
        "date_disputed": 5,
        "rejected": 36,
        "rejected_weak": 6,
        "disagree": 4,
    }
    assert real_in_rejected == 2


@pytest.mark.parametrize("states_a,states_b", [("ffx", "ffx"), ("fpx", "fpx")])
def test_plan_literal_accepts_agreement(states_a: str, states_b: str) -> None:
    assert R.plan_literal(verdict(states_a), verdict(states_b), CFG) == "accepted"
