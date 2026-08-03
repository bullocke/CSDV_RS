"""csdv_core.stages.stand — assign a developmental stage to one stand.

At each imagery date a stand's measured metrics are compared against every
stage's envelope. Each stage scores the proportion of its evaluable metrics that
fall inside range, and the highest scorer wins. Below a minimum score the stand
is left unclassified rather than forced into a class, and where two stages tie
the earlier one in the developmental sequence is chosen.

:func:`csdv_core.stages.envelopes.match_stage` already does the comparison and
already takes a plain mapping, so it is reused unchanged. The raster classifier
in :mod:`csdv_core.stages.classify` is not, because it loops over pixels and
derives its site-type key from an integer raster.

Two properties of the scoring are worth keeping in view when reading a result.
A metric missing from the input lowers the number of metrics evaluated rather
than counting as a failure, so a score of 1.0 on two metrics is weaker evidence
than 0.8 on five. And a metric whose envelope leaves both bounds unset counts as
matched for any finite value, so an envelope of all-unset metrics scores 1.0 for
everything. Unconstrained metrics belong out of the envelope, not in it with
null bounds.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

from csdv_core.config import StagesConfig
from csdv_core.stages.envelopes import match_stage

logger = logging.getLogger(__name__)

#: Site-type key for stands whose site type has not been assigned. The terrain
#: and soils stratification is not implemented, so this is the key the worked
#: examples run on.
UNSTRATIFIED_SITE_TYPE = "type_00"

DEFAULT_MIN_SCORE = 0.5

__all__ = [
    "DEFAULT_MIN_SCORE",
    "UNSTRATIFIED_SITE_TYPE",
    "StandStage",
    "classify_stand_stage",
    "classify_stand_sequence",
    "envelope_key",
]


@dataclass(frozen=True)
class StandStage:
    """The stage assigned to one stand on one date, and the evidence for it.

    Attributes:
        stage: Winning stage code, or None where nothing scored high enough.
        score: Proportion of the winning stage's evaluable metrics in range.
        n_evaluated: How many metrics that score rests on.
        ranked: Every stage that could be evaluated, best first, as
            ``(stage, score, n_evaluated)``.
        failed_metrics: Metrics of the winning stage that were out of range.
        reason: Empty when a stage was assigned, otherwise why it was not.
    """

    stage: str | None
    score: float
    n_evaluated: int
    ranked: tuple[tuple[str, float, int], ...] = ()
    failed_metrics: tuple[str, ...] = ()
    reason: str = ""


def envelope_key(site_type_code: int | None) -> str:
    """Return the ``stages.yaml`` envelope key for a site-type code.

    Mirrors the raster classifier, which formats the code as ``type_NN``. Code
    0, or no code at all, means the site type has not been assigned.
    """
    if site_type_code is None:
        return UNSTRATIFIED_SITE_TYPE
    return f"type_{int(site_type_code):02d}"


def classify_stand_stage(
    metric_values: Mapping[str, float],
    stages_cfg: StagesConfig,
    *,
    site_type_key: str = UNSTRATIFIED_SITE_TYPE,
    min_score: float = DEFAULT_MIN_SCORE,
) -> StandStage:
    """Assign a developmental stage to one stand on one date.

    Args:
        metric_values: Metric name to value. Names must match the envelope
            keys in ``stages.yaml``, which are the names the metric functions
            emit. NaN and missing values are treated the same way: not
            evaluable, so they lower ``n_evaluated`` without counting against
            the stage.
        stages_cfg: Loaded ``stages.yaml``.
        site_type_key: Which envelope set to use.
        min_score: Below this the stand is left unclassified.

    Returns:
        A :class:`StandStage`. When nothing is assigned, ``ranked`` still
        carries every stage that could be evaluated, so a figure can show how
        close the call was.
    """
    order = list(stages_cfg.stage_order) or list(stages_cfg.stages.keys())
    ranked: list[tuple[str, float, int]] = []
    failures: dict[str, tuple[str, ...]] = {}
    n_no_envelope = 0

    for code in order:
        stage = stages_cfg.stages.get(code)
        if stage is None:
            continue
        envelope = stage.envelopes.get(site_type_key)
        if not envelope:
            n_no_envelope += 1
            continue
        match = match_stage(metric_values, envelope)
        if match.n_evaluated == 0:
            continue
        ranked.append((code, match.score, match.n_evaluated))
        failures[code] = match.failed_metrics

    if not ranked:
        reason = (
            f"no stage has an envelope for {site_type_key!r}"
            if n_no_envelope
            else "no envelope metric was evaluable"
        )
        return StandStage(None, 0.0, 0, (), (), reason)

    # Strict greater-than preserves the order of `order`, so a tie goes to the
    # earlier stage in the developmental sequence.
    best_code, best_score, best_n = ranked[0]
    for code, score, n_eval in ranked[1:]:
        if score > best_score:
            best_code, best_score, best_n = code, score, n_eval

    ranked_sorted = tuple(
        sorted(ranked, key=lambda item: (-item[1], order.index(item[0])))
    )
    if best_score < min_score:
        return StandStage(
            None,
            best_score,
            best_n,
            ranked_sorted,
            failures.get(best_code, ()),
            f"best score {best_score:.2f} < min_score {min_score:.2f}",
        )
    return StandStage(
        best_code,
        best_score,
        best_n,
        ranked_sorted,
        failures.get(best_code, ()),
    )


def classify_stand_sequence(
    per_date: list[Mapping[str, float]],
    stages_cfg: StagesConfig,
    *,
    site_type_key: str = UNSTRATIFIED_SITE_TYPE,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[StandStage]:
    """Assign a stage at every date of one stand's series."""
    return [
        classify_stand_stage(
            values,
            stages_cfg,
            site_type_key=site_type_key,
            min_score=min_score,
        )
        for values in per_date
    ]
