"""Per-pixel evaluation of trajectory rule predicates over multi-date cubes.

Pure functions. NaN-safe: pixels with no usable data return False.

Inputs are typically:
    stage_cube     uint8  (T, H, W)  0 = unclassified
    metric_cubes   dict[name, float32 (T, H, W)]
    site_type      uint8  (H, W)     0 = unclassified

A rule signature is a list of :class:`TrajectoryPredicate`. The rule fires
for a pixel when every predicate evaluates True at that pixel (logical AND).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np

from csdv_core.config._models import TrajectoryPredicate, TrajectoryRule

logger = logging.getLogger(__name__)


_OP_TABLE = {
    "<": np.less,
    "<=": np.less_equal,
    "==": np.equal,
    "!=": np.not_equal,
    ">=": np.greater_equal,
    ">": np.greater,
}


def _stage_value_to_code(value, stage_codes: Mapping[str, int]) -> int:
    """Resolve a stage abbreviation or integer to its uint8 code."""
    if isinstance(value, str):
        try:
            return int(stage_codes[value])
        except KeyError as exc:
            raise ValueError(
                f"Unknown stage abbreviation {value!r}; "
                f"known: {sorted(stage_codes)}"
            ) from exc
    if isinstance(value, bool):
        raise ValueError("stage value must be a code or abbreviation, not bool")
    if isinstance(value, (int, float)):
        return int(value)
    raise ValueError(f"Unsupported stage value: {value!r}")


def _site_type_value_to_code(value) -> int:
    """Resolve a site-type abbreviation (``type_06``) or int to its code."""
    if isinstance(value, str):
        if value.startswith("type_"):
            return int(value.split("_", 1)[1])
        return int(value)
    return int(value)


def _compare(arr: np.ndarray, op: str, rhs) -> np.ndarray:
    """Element-wise comparison returning a boolean array. Non-finite -> False."""
    fn = _OP_TABLE[op]
    finite = np.isfinite(arr) if np.issubdtype(arr.dtype, np.floating) else None
    with np.errstate(invalid="ignore"):
        out = fn(arr, rhs)
    if finite is not None:
        out = out & finite
    return out.astype(bool, copy=False)


def _reduce_bool(per_date_mask: np.ndarray, reducer: str) -> np.ndarray:
    """Reduce a (T, H, W) boolean mask along T with ``all``/``any``."""
    if reducer == "all":
        return per_date_mask.all(axis=0)
    if reducer == "any":
        return per_date_mask.any(axis=0)
    raise ValueError(f"Unsupported boolean reducer: {reducer!r}")


def _reduce_numeric(cube: np.ndarray, reducer: str) -> np.ndarray:
    """Reduce a (T, H, W) numeric cube along T to a (H, W) array."""
    if reducer == "latest":
        return cube[-1]
    if reducer == "earliest":
        return cube[0]
    if reducer == "mean":
        return np.nanmean(cube, axis=0)
    if reducer == "min":
        return np.nanmin(cube, axis=0)
    if reducer == "max":
        return np.nanmax(cube, axis=0)
    raise ValueError(f"Unsupported numeric reducer: {reducer!r}")


def evaluate_predicate(
    pred: TrajectoryPredicate,
    *,
    stage_cube: np.ndarray,
    metric_cubes: Mapping[str, np.ndarray],
    site_type: np.ndarray | None,
    stage_codes: Mapping[str, int],
) -> np.ndarray:
    """Evaluate a single predicate, returning a (H, W) boolean mask.

    A predicate with ``value is None`` is a placeholder and never fires.
    """
    if pred.value is None:
        # Placeholder: rule cannot match.
        if stage_cube.ndim == 3:
            return np.zeros(stage_cube.shape[1:], dtype=bool)
        return np.zeros(stage_cube.shape, dtype=bool)

    if pred.dim == "stage":
        return _eval_stage(pred, stage_cube, stage_codes)
    if pred.dim == "stage_delta":
        return _eval_stage_delta(pred, stage_cube)
    if pred.dim == "site_type":
        if site_type is None:
            raise ValueError("Predicate uses dim=site_type but site_type is None")
        return _eval_site_type(pred, site_type)
    if pred.dim == "metric":
        return _eval_metric(pred, metric_cubes)
    if pred.dim == "persistence":
        return _eval_persistence(pred, metric_cubes)
    raise ValueError(f"Unsupported predicate dim: {pred.dim!r}")


def _eval_stage(
    pred: TrajectoryPredicate,
    stage_cube: np.ndarray,
    stage_codes: Mapping[str, int],
) -> np.ndarray:
    if pred.var is not None:
        raise ValueError("dim=stage predicate must not set var")
    if pred.op == "in":
        if not isinstance(pred.value, list):
            raise ValueError("op=in requires value to be a list")
        codes = {_stage_value_to_code(v, stage_codes) for v in pred.value}
        per_date = np.isin(stage_cube, list(codes))
    else:
        rhs = _stage_value_to_code(pred.value, stage_codes)
        per_date = _compare(stage_cube, pred.op, rhs)
    if pred.reducer in ("all", "any"):
        return _reduce_bool(per_date, pred.reducer)
    if pred.reducer in ("latest", "earliest"):
        return per_date[-1] if pred.reducer == "latest" else per_date[0]
    raise ValueError(
        f"reducer={pred.reducer!r} not supported for dim=stage; "
        "use all/any/latest/earliest"
    )


def _eval_stage_delta(
    pred: TrajectoryPredicate,
    stage_cube: np.ndarray,
) -> np.ndarray:
    if pred.var is not None:
        raise ValueError("dim=stage_delta predicate must not set var")
    if stage_cube.shape[0] < 2:
        raise ValueError("dim=stage_delta requires T >= 2")
    deltas = stage_cube[1:].astype("int16") - stage_cube[:-1].astype("int16")
    if pred.op == "in":
        if not isinstance(pred.value, list):
            raise ValueError("op=in requires value to be a list")
        per_pair = np.isin(deltas, [int(v) for v in pred.value])
    else:
        per_pair = _compare(deltas.astype("float32"), pred.op, float(pred.value))
    if pred.reducer in ("all", "any"):
        return _reduce_bool(per_pair, pred.reducer)
    if pred.reducer == "latest":
        return per_pair[-1]
    if pred.reducer == "earliest":
        return per_pair[0]
    raise ValueError(f"reducer={pred.reducer!r} not supported for dim=stage_delta")


def _eval_site_type(
    pred: TrajectoryPredicate,
    site_type: np.ndarray,
) -> np.ndarray:
    if pred.reducer != "scalar":
        raise ValueError("dim=site_type requires reducer=scalar")
    if pred.op == "in":
        if not isinstance(pred.value, list):
            raise ValueError("op=in requires value to be a list")
        codes = [_site_type_value_to_code(v) for v in pred.value]
        return np.isin(site_type, codes)
    rhs = _site_type_value_to_code(pred.value)
    return _compare(site_type.astype("int32"), pred.op, rhs)


def _eval_metric(
    pred: TrajectoryPredicate,
    metric_cubes: Mapping[str, np.ndarray],
) -> np.ndarray:
    if pred.var is None:
        raise ValueError("dim=metric requires var")
    if pred.var not in metric_cubes:
        raise KeyError(
            f"Metric {pred.var!r} not provided; available: {sorted(metric_cubes)}"
        )
    cube = metric_cubes[pred.var]
    if pred.op == "in":
        raise ValueError("op=in is not supported for dim=metric")
    rhs = float(pred.value)  # type: ignore[arg-type]
    if pred.reducer in ("all", "any"):
        per_date = _compare(cube, pred.op, rhs)
        return _reduce_bool(per_date, pred.reducer)
    reduced = _reduce_numeric(cube, pred.reducer)
    return _compare(reduced, pred.op, rhs)


def _eval_persistence(
    pred: TrajectoryPredicate,
    metric_cubes: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Fraction of dates where ``var`` crosses ``threshold`` >= ``value``."""
    if pred.var is None:
        raise ValueError("dim=persistence requires var")
    if pred.threshold is None:
        raise ValueError("dim=persistence requires threshold")
    if pred.var not in metric_cubes:
        raise KeyError(
            f"Metric {pred.var!r} not provided; available: {sorted(metric_cubes)}"
        )
    if pred.reducer != "scalar":
        raise ValueError("dim=persistence requires reducer=scalar")
    cube = metric_cubes[pred.var]
    per_date = _compare(cube, pred.op, float(pred.threshold))
    fraction = per_date.mean(axis=0)
    return fraction >= float(pred.value)  # type: ignore[arg-type]


def evaluate_rule(
    rule: TrajectoryRule,
    *,
    stage_cube: np.ndarray,
    metric_cubes: Mapping[str, np.ndarray],
    site_type: np.ndarray | None,
    stage_codes: Mapping[str, int],
) -> tuple[np.ndarray, int]:
    """AND-combine all predicates in a rule.

    Returns:
        ``(mask, n_predicates)`` where ``mask`` is a (H, W) boolean array
        of pixels matching every predicate, and ``n_predicates`` is the
        number of predicates evaluated (i.e. ``len(rule.signature)``).
        An empty signature returns an all-False mask (rule cannot fire).
    """
    h, w = stage_cube.shape[1:]
    if not rule.signature:
        return np.zeros((h, w), dtype=bool), 0

    mask = np.ones((h, w), dtype=bool)
    for pred in rule.signature:
        sub = evaluate_predicate(
            pred,
            stage_cube=stage_cube,
            metric_cubes=metric_cubes,
            site_type=site_type,
            stage_codes=stage_codes,
        )
        mask &= sub
        if not mask.any():
            break  # short-circuit: no pixel can still match
    return mask, len(rule.signature)


def required_metrics(rule: TrajectoryRule) -> list[str]:
    """Return metric names referenced by ``dim=metric`` or ``dim=persistence``."""
    names: list[str] = []
    for pred in rule.signature:
        if pred.dim in ("metric", "persistence") and pred.var:
            names.append(pred.var)
    return names
