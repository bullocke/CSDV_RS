"""csdv_core.label.verdicts: what a vision model said about one candidate.

Inputs: the parsed JSON of a verdict answer.
Outputs: a :class:`Verdict` with one canopy state per date.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

#: Canopy states a model may give for a date.
STATES = ("forest", "partial_loss", "full_loss", "open")
#: States that mean canopy was lost since the previous date.
LOSS = ("partial_loss", "full_loss")

#: Agent group of each change type in the review vocabulary.
AGENT_OF: dict[str, str] = {
    "clearcut": "harvest",
    "partial_harvest": "harvest",
    "thinning": "harvest",
    "wind": "wind",
    "fire": "fire",
    "insect_disease": "biotic_mortality",
    "gap_mortality": "biotic_mortality",
    "flood": "other",
    "clearing_infrastructure": "conversion",
    "mining": "conversion",
    "other": "other",
}
#: Code of each agent group in the agent raster.
AGENT_CODES: dict[str, int] = {
    "harvest": 1,
    "wind": 2,
    "fire": 3,
    "biotic_mortality": 4,
    "conversion": 5,
    "other": 6,
}

__all__ = ["AGENT_CODES", "AGENT_OF", "LOSS", "STATES", "Verdict", "parse_verdict"]


@dataclass
class Verdict:
    """One model's answer about one candidate.

    Attributes:
        model: Short model name.
        states: One state per date, in date order. Empty when the answer
            could not be used.
        affected_share: ``none``, ``under_25``, ``25_to_75`` or ``over_75``.
        change_type: Type of the first loss, or None.
        outline_fit: ``about_right``, ``too_large``, ``too_small``,
            ``wrong_shape`` or None.
        confidence: 3 certain, 2 probable, 1 possible, or None.
        reason: The model's short explanation.
        valid: The answer had one known state per date.
        cost: What the call cost, US dollars.
    """

    model: str
    states: list[str] = field(default_factory=list)
    affected_share: str | None = None
    change_type: str | None = None
    outline_fit: str | None = None
    confidence: int | None = None
    reason: str = ""
    valid: bool = False
    cost: float = 0.0

    @property
    def loss_index(self) -> list[int]:
        """Dates (by index, from 1) on which loss since the previous date is reported."""
        return [i for i, s in enumerate(self.states) if i > 0 and s in LOSS]

    @property
    def agent(self) -> str | None:
        return AGENT_OF.get(self.change_type or "")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_verdict(model: str, data: Any, n_dates: int, cost: float = 0.0) -> Verdict:
    """Read a verdict answer. An unusable answer gives ``valid=False``.

    An answer is usable when it has exactly one known state per date.
    """
    if not isinstance(data, dict):
        return Verdict(model=model, cost=cost)
    states = [str(s) for s in (data.get("states") or [])]
    valid = len(states) == n_dates and all(s in STATES for s in states)
    confidence = data.get("confidence")
    try:
        confidence = int(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None
    change_type = data.get("change_type")
    return Verdict(
        model=model,
        states=states if valid else [],
        affected_share=data.get("affected_share"),
        change_type=str(change_type) if change_type else None,
        outline_fit=data.get("outline_fit"),
        confidence=confidence,
        reason=str(data.get("reason") or ""),
        valid=valid,
        cost=cost,
    )
