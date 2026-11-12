"""csdv_core.label.rules: the label of every pixel in one interval, and why.

Inputs: boolean and float arrays on one 0.6 m grid: the forest mask at both
dates, canopy cover, height and disturbance probability, and the candidates
with their routed outcome for the interval.
Outputs: a ``reason`` raster (:class:`Reason` codes) and an agent raster.
The change label (background, partial, stand-replacing, ignore) is a lookup
of the reason through a policy table (:func:`to_change`), so the policy can
change without new labels.

Rules are painted from the weakest to the strongest, so for a pixel claimed
by several, loss wins over ignore and ignore wins over background.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from scipy import ndimage as ndi

from csdv_core.config._models import LabelRules
from csdv_core.label import routing as R
from csdv_core.label.lattice import GSD

IGNORE = 255

__all__ = [
    "IGNORE",
    "REASON_GROUPS",
    "BaseLayers",
    "CandidateLayer",
    "Reason",
    "interval_reason",
    "reason_group",
    "to_change",
]


class Reason(IntEnum):
    """Why a pixel has its label in one interval."""

    UNSET = 0
    NO_DATA = 1
    # Loss inside an accepted candidate.
    ACCEPTED_FULL = 10
    ACCEPTED_PARTIAL = 11
    ACCEPTED_SEVERITY_DISPUTED = 12
    EXTENSION_FULL = 13
    EXTENSION_PARTIAL = 14
    # Unlabelled because of an outline.
    BOUNDARY_INSIDE = 20
    BOUNDARY_OUTSIDE = 21
    EXTENSION_IGNORED = 22
    OUTLINE_TRIMMED = 23
    # Unlabelled because of the verdicts.
    DISAGREE = 24
    DATE_DISPUTED = 25
    REJECTED_FLAGGED = 26
    REJECTED_WEAK = 27
    VERDICT_FAILED = 28
    UNJUDGED = 29
    # Unlabelled as a precaution, without a verdict.
    MARGIN = 30
    UNDER_MIN = 31
    RAISED_PROB = 32
    FOREST_LOST_UNJUDGED = 33
    SPARSE_WOODY = 34
    UNSTABLE_FOREST = 35
    # Background.
    JUDGED_INTACT = 40
    POST_EVENT_OPEN = 41
    NONFOREST = 42
    STABLE_FOREST = 43


#: Reasons grouped for sheets and summary tables.
REASON_GROUPS: dict[str, tuple[Reason, ...]] = {
    "loss_full": (Reason.ACCEPTED_FULL, Reason.EXTENSION_FULL),
    "loss_partial": (
        Reason.ACCEPTED_PARTIAL,
        Reason.ACCEPTED_SEVERITY_DISPUTED,
        Reason.EXTENSION_PARTIAL,
    ),
    "ignore_outline": (
        Reason.BOUNDARY_INSIDE,
        Reason.BOUNDARY_OUTSIDE,
        Reason.EXTENSION_IGNORED,
        Reason.OUTLINE_TRIMMED,
    ),
    "ignore_verdict": (
        Reason.DISAGREE,
        Reason.DATE_DISPUTED,
        Reason.REJECTED_FLAGGED,
        Reason.REJECTED_WEAK,
        Reason.VERDICT_FAILED,
        Reason.UNJUDGED,
    ),
    "ignore_guard": (
        Reason.MARGIN,
        Reason.UNDER_MIN,
        Reason.RAISED_PROB,
        Reason.FOREST_LOST_UNJUDGED,
        Reason.SPARSE_WOODY,
        Reason.UNSTABLE_FOREST,
        Reason.NO_DATA,
    ),
    "background_verified": (Reason.JUDGED_INTACT, Reason.POST_EVENT_OPEN),
    "background_nonforest": (Reason.NONFOREST,),
    "background_forest": (Reason.STABLE_FOREST,),
}

_OUTCOME_REASON = {
    R.DISAGREE: Reason.DISAGREE,
    R.DATE_DISPUTED: Reason.DATE_DISPUTED,
    R.REJECTED_FLAGGED: Reason.REJECTED_FLAGGED,
    R.REJECTED_WEAK: Reason.REJECTED_WEAK,
    R.FAILED: Reason.VERDICT_FAILED,
    R.UNJUDGED: Reason.UNJUDGED,
}
_LOSS_REASON = {
    R.LOSS_FULL: (Reason.ACCEPTED_FULL, Reason.EXTENSION_FULL),
    R.LOSS_PARTIAL: (Reason.ACCEPTED_PARTIAL, Reason.EXTENSION_PARTIAL),
    R.LOSS_DISPUTED: (Reason.ACCEPTED_SEVERITY_DISPUTED, Reason.EXTENSION_PARTIAL),
}


@dataclass
class BaseLayers:
    """What is known about every pixel without a verdict.

    Attributes:
        no_data: Imagery is missing on either date.
        forest_before: Forest at the earlier date.
        forest_after: Forest at the later date.
        cover_before: Canopy cover (0 to 1) at the earlier date, NaN unknown.
        height_before: Mean canopy height (m) at the earlier date.
        prob: Disturbance probability of the interval, NaN where not monitored.
        unstable: The unit's forest mask is unreliable in this interval.
    """

    no_data: np.ndarray
    forest_before: np.ndarray
    forest_after: np.ndarray
    cover_before: np.ndarray
    height_before: np.ndarray
    prob: np.ndarray
    unstable: bool = False


@dataclass
class CandidateLayer:
    """One candidate in one interval, on the pixel grid.

    Attributes:
        outcome: Routed outcome of the interval (:mod:`csdv_core.label.routing`).
        mask: Pixels inside the candidate.
        core: Pixels labelled as loss (loss outcomes only).
        extension: Pixels of the model polygon outside the candidate.
        trimmed: Candidate pixels the model polygon leaves out.
        crisp: The extension counts as loss.
        precise: The outline is precise (narrow band), else an envelope.
        wide_margin: Leave a wider ring unlabelled around it.
        agent: Agent code of the loss, 255 when the models disagreed.
    """

    outcome: str
    mask: np.ndarray
    core: np.ndarray | None = None
    extension: np.ndarray | None = None
    trimmed: np.ndarray | None = None
    crisp: bool = False
    precise: bool = False
    wide_margin: bool = False
    agent: int = IGNORE


def _px(metres: float) -> int:
    return int(round(metres / GSD))


def _within(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Pixels at most ``pixels`` away from ``mask`` (including the mask)."""
    if not mask.any():
        return np.zeros_like(mask)
    if pixels <= 0:
        return mask.copy()
    return ndi.distance_transform_edt(~mask) <= pixels


def interval_reason(
    base: BaseLayers, candidates: list[CandidateLayer], cfg: LabelRules
) -> tuple[np.ndarray, np.ndarray]:
    """Reason and agent of every pixel in one interval.

    Args:
        base: Forest, cover, height and probability layers.
        candidates: Every candidate that touches the grid, with its outcome.
            Candidates with an empty outcome are skipped.
        cfg: Rule thresholds.

    Returns:
        ``reason`` (uint8 :class:`Reason` codes) and ``agent`` (uint8 agent
        codes, 255 where there is no agreed loss).
    """
    before, after = base.forest_before, base.forest_after
    with np.errstate(invalid="ignore"):
        woody = (base.cover_before >= cfg.sparse_cover) | (
            base.height_before >= cfg.sparse_height_m
        )
        raised = base.prob >= cfg.raised_prob
    reason = np.full(before.shape, Reason.NONFOREST, dtype=np.uint8)
    reason[~before & woody] = Reason.SPARSE_WOODY
    reason[before & after] = (
        Reason.UNSTABLE_FOREST if base.unstable else Reason.STABLE_FOREST
    )
    reason[before & ~after] = Reason.FOREST_LOST_UNJUDGED
    reason[before & raised] = Reason.RAISED_PROB
    agent = np.full(before.shape, IGNORE, dtype=np.uint8)

    live = [c for c in candidates if c.outcome]
    for cand in live:
        if cand.outcome == R.UNDER_MIN:
            reason[cand.mask] = Reason.UNDER_MIN

    # Ring around every candidate whose interior is loss or unlabelled.
    normal = np.zeros(before.shape, dtype=bool)
    wide = np.zeros(before.shape, dtype=bool)
    sharp = np.zeros(before.shape, dtype=bool)
    for cand in live:
        if cand.outcome in R.BACKGROUND_OUTCOMES or cand.outcome == R.UNDER_MIN:
            continue
        target = normal
        if cand.wide_margin:
            target = wide
        elif cand.precise and cand.outcome in R.LOSS_OUTCOMES:
            target = sharp
        target |= cand.mask
        if cand.extension is not None:
            target |= cand.extension
    ring = (
        _within(normal, _px(cfg.margin_m))
        | _within(wide, _px(cfg.margin_wide_m))
        | _within(sharp, _px(cfg.margin_precise_m))
    )
    reason[ring] = Reason.MARGIN

    for cand in live:
        if cand.outcome == R.INTACT:
            reason[cand.mask] = Reason.JUDGED_INTACT
        elif cand.outcome == R.OPEN:
            reason[cand.mask] = Reason.POST_EVENT_OPEN
    for cand in live:
        if cand.outcome in _OUTCOME_REASON:
            reason[cand.mask] = _OUTCOME_REASON[cand.outcome]
    losses = [c for c in live if c.outcome in R.LOSS_OUTCOMES]
    for cand in losses:
        if cand.trimmed is not None:
            reason[cand.trimmed] = Reason.OUTLINE_TRIMMED
        if cand.extension is not None and not cand.crisp:
            reason[cand.extension] = Reason.EXTENSION_IGNORED

    loss = np.zeros(before.shape, dtype=bool)
    precise = np.zeros(before.shape, dtype=bool)
    footprint = np.zeros(before.shape, dtype=bool)
    for cand in losses:
        core_reason, extension_reason = _LOSS_REASON[cand.outcome]
        parts = [(cand.core if cand.core is not None else cand.mask, core_reason)]
        if cand.extension is not None:
            footprint |= cand.extension
            if cand.crisp:
                parts.append((cand.extension, extension_reason))
        for pixels, code in parts:
            reason[pixels] = code
            agent[pixels] = cand.agent
            loss |= pixels
            if cand.precise:
                precise |= pixels
    footprint |= loss

    if loss.any():
        # Band on both sides of the outline of the labelled loss. Where the
        # event is known to continue (an unlabelled extension) there is no
        # outline to be unsure about, so the band is measured from the edge
        # of the whole footprint.
        depth = ndi.distance_transform_edt(footprint)
        away = ndi.distance_transform_edt(~loss)
        for part, metres_out, metres_in in (
            (precise, cfg.band_precise_m, cfg.band_inside_precise_m),
            (loss & ~precise, cfg.band_envelope_m, cfg.band_inside_envelope_m),
        ):
            if not part.any():
                continue
            outer, inner = _px(metres_out), _px(metres_in)
            if outer > 0:
                near = _within(part, outer)
                reason[near & ~footprint & (away <= outer)] = Reason.BOUNDARY_OUTSIDE
            if inner > 0:
                inside = part & (depth <= inner)
                reason[inside] = Reason.BOUNDARY_INSIDE
                agent[inside] = IGNORE

    with np.errstate(invalid="ignore"):
        unknown = np.isnan(base.cover_before)
    reason[base.no_data | unknown] = Reason.NO_DATA
    agent[base.no_data | unknown] = IGNORE
    return reason, agent


def to_change(reason: np.ndarray, policy: dict[str, int]) -> np.ndarray:
    """Change label of each pixel: 0 background, 1 partial, 2 full, 255 ignore.

    Args:
        reason: :class:`Reason` codes.
        policy: Label of each reason by lower-case name. Reasons that are not
            listed are ignore.

    Raises:
        KeyError: If ``policy`` names an unknown reason.
    """
    table = np.full(256, IGNORE, dtype=np.uint8)
    for name, label in policy.items():
        table[Reason[name.upper()]] = label
    return table[reason]


def reason_group(reason: np.ndarray) -> np.ndarray:
    """Index into :data:`REASON_GROUPS` (in order) for each pixel, 255 unset."""
    table = np.full(256, IGNORE, dtype=np.uint8)
    for index, codes in enumerate(REASON_GROUPS.values()):
        for code in codes:
            table[code] = index
    return table[reason]
