"""csdv_core.trajectories.stand — assign a trajectory class to one stand.

A single stage assignment says what a stand looks like now. The sequence of
assignments says whether it is going anywhere. Each trajectory class is a list
of conditions that must all hold, classes are evaluated in a fixed priority
order, and the first match wins.

Most of them cannot currently fire. Sixteen of the nineteen hold at least one
predicate whose threshold has never been filled in, and several of the rest
depend on metrics that have no implementation. That is a fact about the state of
the specification, not a bug, and it matters more than any class the engine does
assign: a worked example that quietly reported "no trajectory" would hide it.
:func:`blocking_report` therefore states, rule by rule, why a class was not
available, separating a rule that ran and did not match from one that could
never have run.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from csdv_core.config import StagesConfig, TrajectoriesConfig
from csdv_core.config._models import TrajectoryRule
from csdv_core.trajectories.classify import classify_trajectories
from csdv_core.trajectories.rules import required_metrics
from csdv_core.trajectories.sequences import StandSequence, stand_cubes

logger = logging.getLogger(__name__)

#: Why a rule could not produce a result.
BLOCKED_NO_THRESHOLD = "threshold not set"
BLOCKED_METRIC_MISSING = "metric not available"
BLOCKED_EVALUATED_NO_MATCH = "evaluated, conditions not met"

__all__ = [
    "BLOCKED_EVALUATED_NO_MATCH",
    "BLOCKED_METRIC_MISSING",
    "BLOCKED_NO_THRESHOLD",
    "StandTrajectory",
    "blocking_report",
    "classify_stand_trajectory",
    "rule_is_fireable",
]


@dataclass(frozen=True)
class StandTrajectory:
    """The trajectory assigned to one stand, and why the others were not.

    Attributes:
        code: Winning trajectory code, or None where nothing matched.
        name: Human-readable label of the winning class.
        n_predicates: Predicates evaluated for the matching rule.
        n_dates: Dates on which the stand carried a stage assignment.
        evaluable: Rules that could have fired given the available metrics.
        blocked: Rule code to the reasons it did not fire.
    """

    code: str | None
    name: str | None = None
    n_predicates: int = 0
    n_dates: int = 0
    evaluable: tuple[str, ...] = ()
    blocked: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


def rule_is_fireable(
    rule: TrajectoryRule,
    available_metrics: Sequence[str],
) -> tuple[bool, tuple[str, ...]]:
    """Return whether a rule could fire at all, and the reasons if not.

    A rule is unfireable when any predicate has no threshold, when it names a
    metric that is not available, or when it has no predicates.
    """
    reasons: list[str] = []
    if not rule.signature:
        reasons.append("rule has no predicates")
    have = set(available_metrics)
    for pred in rule.signature:
        if pred.value is None:
            label = pred.var or pred.dim
            reasons.append(f"{BLOCKED_NO_THRESHOLD}: {label}")
        if pred.var is not None and pred.var not in have:
            reasons.append(f"{BLOCKED_METRIC_MISSING}: {pred.var}")
    # Preserve order while removing duplicates.
    unique = tuple(dict.fromkeys(reasons))
    return (not unique), unique


def blocking_report(
    traj_cfg: TrajectoriesConfig,
    available_metrics: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """State, for every trajectory class, why it could not fire.

    Classes that could fire are absent from the result. This is the honest
    summary of how much of the trajectory layer is currently reachable.
    """
    out: dict[str, tuple[str, ...]] = {}
    order = list(traj_cfg.trajectory_order) or list(traj_cfg.trajectories.keys())
    for code in order:
        rule = traj_cfg.trajectories.get(code)
        if rule is None:
            out[code] = ("rule not defined",)
            continue
        fireable, reasons = rule_is_fireable(rule, available_metrics)
        if not fireable:
            out[code] = reasons
    return out


def classify_stand_trajectory(
    sequence: StandSequence,
    traj_cfg: TrajectoriesConfig,
    stages_cfg: StagesConfig,
    *,
    site_type_code: int = 0,
) -> StandTrajectory:
    """Classify one stand's stage and metric sequence into a trajectory class.

    The stand is shaped as a one-pixel raster and passed to the raster engine,
    so the predicate semantics are exactly those of the wall-to-wall path.

    Args:
        sequence: The stand's series, from
            :func:`csdv_core.trajectories.sequences.build_sequence`.
        traj_cfg: Loaded ``trajectories.yaml``.
        stages_cfg: Loaded ``stages.yaml``, for the stage codes.
        site_type_code: Site type, 0 where unassigned. Rules that test site
            type cannot match at 0.

    Returns:
        A :class:`StandTrajectory`, carrying the blocking report whether or not
        a class was assigned.

    Raises:
        ValueError: If the sequence has fewer than two dates. A trajectory is
            a statement about change, so one date cannot produce one.
    """
    if len(sequence) < 2:
        raise ValueError(
            f"Stand {sequence.stand_id!r} has {len(sequence)} date(s); "
            "a trajectory needs at least two"
        )

    stage_cube, metric_cubes = stand_cubes(sequence, stages_cfg)

    # Supply an all-NaN cube for any metric a rule references but the stand
    # does not carry, so an unimplemented metric blocks its rule rather than
    # raising out of the engine.
    needed: set[str] = set()
    for rule in traj_cfg.trajectories.values():
        needed.update(required_metrics(rule))
    for name in sorted(needed - set(metric_cubes)):
        metric_cubes[name] = np.full((len(sequence), 1, 1), np.nan, dtype="float32")

    available = sorted(
        name for name, cube in metric_cubes.items() if bool(np.isfinite(cube).any())
    )
    blocked = blocking_report(traj_cfg, available)
    order = list(traj_cfg.trajectory_order) or list(traj_cfg.trajectories.keys())
    evaluable = tuple(code for code in order if code not in blocked)

    site_type = np.full((1, 1), np.uint8(site_type_code), dtype="uint8")
    trajectory, n_predicates, n_dates = classify_trajectories(
        stage_cube, metric_cubes, site_type, traj_cfg, stages_cfg
    )

    code_value = int(trajectory[0, 0])
    if code_value == 0:
        for code in evaluable:
            blocked.setdefault(code, (BLOCKED_EVALUATED_NO_MATCH,))
        logger.info(
            "Stand %s: no trajectory assigned; %d of %d rules were evaluable",
            sequence.stand_id,
            len(evaluable),
            len(order),
        )
        return StandTrajectory(
            code=None,
            n_dates=int(n_dates[0, 0]),
            evaluable=evaluable,
            blocked=blocked,
        )

    by_code = {int(v): k for k, v in traj_cfg.trajectory_codes.items()}
    name = by_code.get(code_value)
    rule = traj_cfg.trajectories.get(name) if name else None
    return StandTrajectory(
        code=name,
        name=rule.name if rule is not None else None,
        n_predicates=int(n_predicates[0, 0]),
        n_dates=int(n_dates[0, 0]),
        evaluable=evaluable,
        blocked=blocked,
    )
