"""csdv_core.label.routing: combine two verdicts into one outcome per interval.

Inputs: the :class:`~csdv_core.label.verdicts.Verdict` of two models for one
candidate. Outputs: a :class:`Routed` with one outcome for every interval
between consecutive dates. No person takes part.

A candidate is accepted when both models report canopy loss beginning on the
same date. It is rejected when neither reports loss on any date. When both
report loss but name different first dates the candidate is kept
(``date_disputed``). The intervals before the later date are left
unlabelled. The interval of the later date is labelled as loss, or left
unlabelled too (``disputed_date``). Apart from that interval, an interval
is labelled only where the two models agree about it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from csdv_core.config._models import LabelRouting
from csdv_core.label.verdicts import Verdict

#: Outcomes of one candidate in one interval.
LOSS_FULL = "loss_full"
LOSS_PARTIAL = "loss_partial"
LOSS_DISPUTED = "loss_disputed"  # both report loss, severities differ
DISAGREE = "disagree"  # one model reports loss, the other does not
DATE_DISPUTED = "date_disputed"
INTACT = "intact"  # both say forest
OPEN = "open"  # no new loss, at least one model says open
REJECTED_FLAGGED = "rejected_flagged"  # confident joint rejection, flagged interval
REJECTED_WEAK = "rejected_weak"  # joint rejection below the confidence bar
FLAGGED_UNCONFIRMED = "flagged_unconfirmed"  # flagged here, loss placed elsewhere
LOSS_REPEATED = "loss_repeated"  # loss again right after an agreed loss
STRESSED = "stressed"  # a model sees a leafless or thin canopy, not a loss
FAILED = "failed"  # a verdict is missing or unusable
UNJUDGED = "unjudged"  # large enough, never sent to the models
UNDER_MIN = "under_min"  # too small to judge

LOSS_OUTCOMES = frozenset({LOSS_FULL, LOSS_PARTIAL, LOSS_DISPUTED})
BACKGROUND_OUTCOMES = frozenset({INTACT, OPEN})

__all__ = [
    "BACKGROUND_OUTCOMES",
    "LOSS_OUTCOMES",
    "Routed",
    "plan_literal",
    "route",
    "unjudged",
]


@dataclass
class Routed:
    """What two models agreed about one candidate.

    Attributes:
        status: ``accepted``, ``date_disputed``, ``rejected``,
            ``rejected_weak``, ``disagree``, ``failed`` or ``unjudged``.
        outcomes: One outcome per interval. Interval *k* lies between date
            *k* and date *k + 1*.
        first_loss: Interval of the first loss both models report, or None.
        severity: ``stand_replacing`` or ``partial`` for that interval.
        change_type: The type both models gave, or None when they differ.
        agent: The agent group both models gave, or None when they differ.
        confidence: The lower of the two stated confidences.
        outline_fit: What each model said about the outline.
        affected_share: What each model said about the share affected.
    """

    status: str
    outcomes: list[str]
    first_loss: int | None = None
    severity: str | None = None
    change_type: str | None = None
    agent: str | None = None
    confidence: int | None = None
    outline_fit: tuple[str | None, str | None] = (None, None)
    affected_share: tuple[str | None, str | None] = (None, None)
    change_types: tuple[str | None, str | None] = field(default=(None, None))

    @property
    def has_loss(self) -> bool:
        return any(o in LOSS_OUTCOMES for o in self.outcomes)


def unjudged(n_intervals: int, flagged: int | None, small: bool) -> Routed:
    """A candidate that was not sent to the models.

    Only the interval that flagged it is affected. The others follow the
    rules for ground outside every candidate.
    """
    outcome = UNDER_MIN if small else UNJUDGED
    outcomes = [""] * n_intervals
    if flagged is not None and 0 <= flagged < n_intervals:
        outcomes[flagged] = outcome
    return Routed(status="unjudged", outcomes=outcomes)


def _quiet(a: Verdict, b: Verdict, date: int) -> str:
    """Outcome of an interval in which neither model reports loss."""
    states = (a.states[date], b.states[date])
    if "stressed" in states:
        # Defoliated, or dead and standing: the pair of images differs, and
        # the models cannot say that the canopy is intact.
        return STRESSED
    return INTACT if states == ("forest", "forest") else OPEN


def route(a: Verdict, b: Verdict, flagged: int | None, cfg: LabelRouting) -> Routed:
    """Combine two verdicts on the same candidate.

    Args:
        a: First model's verdict.
        b: Second model's verdict.
        flagged: Interval in which the candidate's source flagged it. A
            jointly rejected candidate stays unlabelled there, because about
            one rejection in twelve was a real disturbance in the pilot.
        cfg: Routing thresholds.
    """
    n = max(len(a.states), len(b.states)) - 1
    if not (a.valid and b.valid) or len(a.states) != len(b.states):
        return Routed(status="failed", outcomes=[FAILED] * max(n, 0))
    loss_a, loss_b = set(a.loss_index), set(b.loss_index)
    confidences = [c for c in (a.confidence, b.confidence) if c is not None]
    common = {
        "confidence": min(confidences) if confidences else None,
        "outline_fit": (a.outline_fit, b.outline_fit),
        "affected_share": (a.affected_share, b.affected_share),
        "change_types": (a.change_type, b.change_type),
    }
    quiet = [_quiet(a, b, k + 1) for k in range(n)]
    if not loss_a and not loss_b:
        strong = len(confidences) == 2 and min(confidences) >= cfg.reject_confidence
        outcomes = list(quiet)
        if flagged is not None and 0 <= flagged < n:
            outcomes[flagged] = REJECTED_FLAGGED if strong else REJECTED_WEAK
        return Routed(
            status="rejected" if strong else "rejected_weak",
            outcomes=outcomes,
            **common,
        )
    if not loss_a or not loss_b:
        outcomes = [
            DISAGREE if (k + 1) in (loss_a | loss_b) else quiet[k] for k in range(n)
        ]
        _guard_flagged(outcomes, flagged, cfg)
        return Routed(status="disagree", outcomes=outcomes, **common)
    first_a, first_b = min(loss_a), min(loss_b)
    lo, hi = min(first_a, first_b), max(first_a, first_b)
    outcomes = []
    for k in range(n):
        date = k + 1
        in_a, in_b = date in loss_a, date in loss_b
        if lo != hi and date == hi and cfg.disputed_date == "later":
            # Both models saw the canopy go, one of them earlier. By the
            # later date they agree that it is gone.
            outcomes.append(_first_severity(a, b, first_a, first_b))
        elif lo != hi and lo <= date <= hi:
            outcomes.append(DATE_DISPUTED)
        elif in_a and in_b:
            states = {a.states[date], b.states[date]}
            if states == {"full_loss"}:
                outcomes.append(LOSS_FULL)
            elif states == {"partial_loss"}:
                outcomes.append(LOSS_PARTIAL)
            else:
                outcomes.append(LOSS_DISPUTED)
        elif in_a or in_b:
            outcomes.append(DISAGREE)
        else:
            outcomes.append(quiet[k])
    if cfg.consecutive_loss == "ignore":
        previous = list(outcomes)
        for k in range(1, n):
            if previous[k] in LOSS_OUTCOMES and previous[k - 1] in LOSS_OUTCOMES:
                outcomes[k] = LOSS_REPEATED
    _guard_flagged(outcomes, flagged, cfg)
    agreed = [k for k, o in enumerate(outcomes) if o in LOSS_OUTCOMES]
    first = agreed[0] if agreed else None
    severity = None
    if first is not None:
        severity = "stand_replacing" if outcomes[first] == LOSS_FULL else "partial"
    same_type = a.change_type is not None and a.change_type == b.change_type
    same_agent = a.agent is not None and a.agent == b.agent
    return Routed(
        status="accepted" if lo == hi else "date_disputed",
        outcomes=outcomes,
        first_loss=first,
        severity=severity,
        change_type=a.change_type if same_type else None,
        agent=a.agent if same_agent else None,
        **common,
    )


def _first_severity(a: Verdict, b: Verdict, first_a: int, first_b: int) -> str:
    """Loss outcome from the state each model gave on its own first loss date."""
    states = {a.states[first_a], b.states[first_b]}
    if states == {"full_loss"}:
        return LOSS_FULL
    if states == {"partial_loss"}:
        return LOSS_PARTIAL
    return LOSS_DISPUTED


def _guard_flagged(outcomes: list[str], flagged: int | None, cfg: LabelRouting) -> None:
    """Leave the flagged interval unlabelled when the models see no loss in it.

    The candidate's source saw a change in that interval. When the models
    place the loss in another interval, their "no new loss" there is not
    enough to call the ground background.
    """
    if not cfg.guard_flagged_interval or flagged is None:
        return
    if 0 <= flagged < len(outcomes) and outcomes[flagged] in BACKGROUND_OUTCOMES:
        outcomes[flagged] = FLAGGED_UNCONFIRMED


def plan_literal(a: Verdict, b: Verdict, cfg: LabelRouting) -> str:
    """Status under the first wording of the rule, kept for comparison.

    Accept when both models report loss with the same first date. Reject
    when both report none with confidence at the bar. Everything else is
    ignore.
    """
    if not (a.valid and b.valid):
        return "ignore"
    loss_a, loss_b = a.loss_index, b.loss_index
    if loss_a and loss_b:
        return "accepted" if loss_a[0] == loss_b[0] else "ignore"
    if not loss_a and not loss_b:
        sure = min(a.confidence or 0, b.confidence or 0) >= cfg.reject_confidence
        return "rejected" if sure else "ignore"
    return "ignore"
