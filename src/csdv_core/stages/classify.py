"""Per-window stage classification engine.

Pure function: takes a stack of metric arrays plus a per-pixel site-type
raster plus the loaded :class:`csdv_core.config.StagesConfig` and returns
per-pixel ``stage`` (uint8), ``stage_score`` (float32), and
``stage_evaluated_count`` (uint8).

Tie-break order: V5 stage order from ``stages.yaml`` (``stage_order``).
Pixels with best ``score < min_score`` get ``stage = 0`` (unclassified).
"""

from __future__ import annotations

import logging

import numpy as np

from csdv_core.config import StagesConfig
from csdv_core.stages.envelopes import match_stage

logger = logging.getLogger(__name__)

UNCLASSIFIED = np.uint8(0)


def classify_stages(
    metrics: dict[str, np.ndarray],
    site_type: np.ndarray,
    stages_cfg: StagesConfig,
    *,
    min_score: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify each window into a developmental stage.

    Args:
        metrics: Dict mapping metric name to a 2-D array. All arrays must
            share a shape with ``site_type``.
        site_type: 2-D uint8 array of site-type codes from
            :func:`csdv_core.stratification.assign.assign_site_types`.
            Pixels with code 0 are unclassified and get stage 0.
        stages_cfg: Loaded ``stages.yaml`` configuration.
        min_score: Minimum match-score for a stage to be assigned.

    Returns:
        ``(stage, stage_score, stage_evaluated)``:

        - ``stage`` uint8: code from ``stages_cfg.stage_codes`` or 0.
        - ``stage_score`` float32: best stage's match score; NaN if unclassified.
        - ``stage_evaluated`` uint8: number of metrics evaluated for the
          best stage (a QA hint).
    """
    if not metrics:
        raise ValueError("metrics cannot be empty")

    shapes = {name: arr.shape for name, arr in metrics.items()}
    ref_shape = site_type.shape
    for name, shp in shapes.items():
        if shp != ref_shape:
            raise ValueError(
                f"metric {name} shape {shp} != site_type shape {ref_shape}"
            )

    stage = np.zeros(ref_shape, dtype="uint8")
    stage_score = np.full(ref_shape, np.nan, dtype="float32")
    stage_evaluated = np.zeros(ref_shape, dtype="uint8")

    stage_order = list(stages_cfg.stage_order) or list(stages_cfg.stages.keys())
    stage_codes = dict(stages_cfg.stage_codes)
    if not stage_codes:
        # Fall back to position in stage_order, starting at 1.
        stage_codes = {code: i + 1 for i, code in enumerate(stage_order)}

    metric_names = list(metrics.keys())
    rows, cols = ref_shape
    for r in range(rows):
        for c in range(cols):
            st_code = int(site_type[r, c])
            if st_code == 0:
                continue
            site_type_key = f"type_{st_code:02d}"
            values = {n: float(metrics[n][r, c]) for n in metric_names}

            best_code: int = 0
            best_score: float = -1.0
            best_eval: int = 0
            for stage_code in stage_order:
                stage_envs = stages_cfg.stages.get(stage_code)
                if stage_envs is None:
                    continue
                env_for_site = stage_envs.envelopes.get(site_type_key)
                if env_for_site is None:
                    continue
                m = match_stage(values, env_for_site)
                if m.n_evaluated == 0:
                    continue
                if m.score > best_score:
                    best_score = m.score
                    best_code = stage_codes.get(stage_code, 0)
                    best_eval = m.n_evaluated

            if best_score >= min_score and best_code != 0:
                stage[r, c] = best_code
                stage_score[r, c] = best_score
                stage_evaluated[r, c] = best_eval

    n_assigned = int((stage != 0).sum())
    logger.info(
        "classify_stages: %d/%d pixels assigned (min_score=%.2f)",
        n_assigned,
        stage.size,
        min_score,
    )
    return stage, stage_score, stage_evaluated
