"""Rule-based site-type assignment.

Pure function: given a stack of stratification variables and the
:class:`csdv_core.config.SiteTypesConfig` rules, return a per-pixel uint8
site-type code plus a float32 ``match_score`` for QA. No I/O.

Predicates from ``site_types.yaml`` are AND-combined within a site type;
the first site type whose predicates all evaluate True wins. Pixels that
match no site type get ``site_type = 0`` ("unclassified"). Predicates
referencing variables that are absent from the input dict are treated as
False, so callers can run with partial inputs.
"""

from __future__ import annotations

import logging
import operator as op
from collections.abc import Mapping

import numpy as np

from csdv_core.config import SiteTypesConfig
from csdv_core.config._models import Predicate, SiteTypeRule

logger = logging.getLogger(__name__)

UNCLASSIFIED = np.uint8(0)

_OPS = {
    "<": op.lt,
    "<=": op.le,
    "==": op.eq,
    "!=": op.ne,
    ">=": op.ge,
    ">": op.gt,
}


def _evaluate_predicate(
    pred: Predicate,
    stratvars: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Evaluate one predicate against the stratification stack.

    Returns a boolean array. NaN inputs propagate as False (predicates over
    missing data never match). A null ``value`` means the predicate is
    unconfigured and is treated as a no-op (always True), which lets
    placeholder rules in ``site_types.yaml`` parse without matching anything
    spurious during rule assembly.
    """
    if pred.value is None:
        # Unconfigured threshold: skip this predicate (vacuously true).
        sample = next(iter(stratvars.values()))
        return np.ones(sample.shape, dtype=bool)

    arr = stratvars.get(pred.var)
    if arr is None:
        sample = next(iter(stratvars.values()))
        return np.zeros(sample.shape, dtype=bool)

    fn = _OPS[pred.op]
    with np.errstate(invalid="ignore"):
        result = fn(arr, pred.value)

    if np.issubdtype(arr.dtype, np.floating):
        result = np.where(np.isnan(arr), False, result)
    return result.astype(bool)


def _evaluate_rule(
    rule: SiteTypeRule,
    stratvars: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one site-type rule.

    Returns ``(matched, n_evaluated)`` where ``matched`` is bool (all
    predicates True) and ``n_evaluated`` counts non-null predicates that
    actually contributed to the decision. With no configured predicates,
    ``matched`` is all-False (avoids matching everything to a placeholder).
    """
    if not stratvars:
        raise ValueError("stratvars cannot be empty")
    sample = next(iter(stratvars.values()))
    if not rule.rules:
        return (
            np.zeros(sample.shape, dtype=bool),
            np.zeros(sample.shape, dtype="uint8"),
        )

    matched = np.ones(sample.shape, dtype=bool)
    n_eval = np.zeros(sample.shape, dtype="uint8")
    any_configured = False
    for pred in rule.rules:
        if pred.value is None:
            continue
        any_configured = True
        m = _evaluate_predicate(pred, stratvars)
        matched &= m
        n_eval += 1

    if not any_configured:
        # All predicates are placeholders; do not match (avoid false claims).
        matched = np.zeros(sample.shape, dtype=bool)
    return matched, n_eval


def assign_site_types(
    stratvars: Mapping[str, np.ndarray],
    rules: SiteTypesConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign a site-type code to each pixel.

    Args:
        stratvars: 2-D arrays keyed by variable name (e.g. ``"twi"``,
            ``"slope_deg"``, ``"hydric_pct"``). All arrays must share a shape.
        rules: Loaded site-type rules.

    Returns:
        Tuple ``(site_type, match_score)``:

        - ``site_type`` is uint8; the integer suffix of the matching key
          in ``site_types.yaml`` (e.g. ``"type_06"`` -> 6). Unmatched
          pixels are 0.
        - ``match_score`` is float32 in ``[0, 1]``: number of configured
          predicates that fired true for the winning rule, divided by the
          number configured. NaN if no rule matched.
    """
    if not stratvars:
        raise ValueError("stratvars cannot be empty")
    sample = next(iter(stratvars.values()))
    shape = sample.shape

    site_type = np.zeros(shape, dtype="uint8")
    match_score = np.full(shape, np.nan, dtype="float32")
    assigned = np.zeros(shape, dtype=bool)

    # Stable order: dict iteration order preserves YAML order.
    for key, rule in rules.site_types.items():
        code = _code_from_key(key)
        matched, n_eval = _evaluate_rule(rule, stratvars)
        new = matched & ~assigned
        if not new.any():
            continue
        site_type[new] = code
        n_configured = sum(1 for p in rule.rules if p.value is not None)
        score = 1.0 if n_configured == 0 else n_eval[new] / n_configured
        match_score[new] = np.asarray(score, dtype="float32")
        assigned |= new
        logger.debug("site type %s assigned to %d pixels", key, int(new.sum()))

    return site_type, match_score


def _code_from_key(key: str) -> int:
    """Parse the trailing integer from a key like ``"type_06"``."""
    suffix = key.split("_")[-1]
    try:
        return int(suffix)
    except ValueError as exc:
        raise ValueError(
            f"site-type key {key!r} must end in an integer (e.g. 'type_06')"
        ) from exc
