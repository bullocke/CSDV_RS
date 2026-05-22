"""Per-pixel trajectory classification engine.

Pure function. Iterates trajectory rules in ``trajectory_order``,
first-match-wins. For each pixel, returns the assigned trajectory code,
the count of predicates evaluated for the matched rule, and the count
of valid (non-zero) stage dates available at that pixel.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np

from csdv_core.config import StagesConfig, TrajectoriesConfig
from csdv_core.trajectories.rules import evaluate_rule, required_metrics

logger = logging.getLogger(__name__)

UNCLASSIFIED = np.uint8(0)


def all_required_metrics(traj_cfg: TrajectoriesConfig) -> list[str]:
    """Union of metric names referenced by any rule in the config."""
    out: set[str] = set()
    for rule in traj_cfg.trajectories.values():
        out.update(required_metrics(rule))
    return sorted(out)


def classify_trajectories(
    stage_cube: np.ndarray,
    metric_cubes: Mapping[str, np.ndarray],
    site_type: np.ndarray | None,
    traj_cfg: TrajectoriesConfig,
    stages_cfg: StagesConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify each pixel into a trajectory class.

    Args:
        stage_cube: uint8 array of shape ``(T, H, W)`` from
            :func:`csdv_core.io.stages_io.read_stage_cube`. Code 0 marks
            unclassified dates.
        metric_cubes: Mapping ``metric_name -> float32 (T, H, W)`` cubes.
            Must contain every metric referenced by the rules; missing
            metrics raise ``KeyError``.
        site_type: Optional uint8 ``(H, W)`` site-type raster. Required
            when any rule uses ``dim=site_type``.
        traj_cfg: Loaded ``trajectories.yaml`` configuration.
        stages_cfg: Loaded ``stages.yaml`` configuration (for ``stage_codes``).

    Returns:
        ``(trajectory, n_predicates, n_dates)``:

        - ``trajectory`` uint8 ``(H, W)``: code from ``trajectory_codes``,
          0 = unclassified.
        - ``n_predicates`` uint8 ``(H, W)``: number of predicates evaluated
          for the matched rule (0 where unclassified).
        - ``n_dates`` uint8 ``(H, W)``: count of dates where the stage
          raster was non-zero at this pixel.
    """
    if stage_cube.ndim != 3:
        raise ValueError(f"stage_cube must be (T, H, W); got shape {stage_cube.shape}")
    t, h, w = stage_cube.shape
    if t < 2:
        raise ValueError(f"stage_cube must have T >= 2; got T={t}")

    for name, cube in metric_cubes.items():
        if cube.shape != (t, h, w):
            raise ValueError(
                f"metric {name} shape {cube.shape} != stage_cube shape {(t, h, w)}"
            )
    if site_type is not None and site_type.shape != (h, w):
        raise ValueError(
            f"site_type shape {site_type.shape} != stage_cube (H, W) {(h, w)}"
        )

    trajectory = np.zeros((h, w), dtype="uint8")
    n_predicates = np.zeros((h, w), dtype="uint8")

    n_dates = (stage_cube != 0).sum(axis=0).astype("uint8")

    order = list(traj_cfg.trajectory_order) or list(traj_cfg.trajectories.keys())
    codes = dict(traj_cfg.trajectory_codes)
    if not codes:
        codes = {name: i + 1 for i, name in enumerate(order)}

    stage_codes = dict(stages_cfg.stage_codes)

    unassigned = trajectory == 0
    for rule_name in order:
        rule = traj_cfg.trajectories.get(rule_name)
        if rule is None:
            logger.warning(
                "trajectory_order references missing rule %s; skipping", rule_name
            )
            continue
        code = int(codes.get(rule_name, 0))
        if code == 0:
            logger.warning(
                "trajectory %s has no code in trajectory_codes; skipping", rule_name
            )
            continue
        mask, n_preds = evaluate_rule(
            rule,
            stage_cube=stage_cube,
            metric_cubes=metric_cubes,
            site_type=site_type,
            stage_codes=stage_codes,
        )
        winners = mask & unassigned
        if not winners.any():
            continue
        trajectory[winners] = code
        n_predicates[winners] = n_preds
        unassigned = trajectory == 0
        if not unassigned.any():
            break

    n_assigned = int((trajectory != 0).sum())
    logger.info(
        "classify_trajectories: %d/%d pixels assigned across %d rules",
        n_assigned,
        trajectory.size,
        len(order),
    )
    return trajectory, n_predicates, n_dates
