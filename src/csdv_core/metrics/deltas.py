"""csdv_core.metrics.deltas - Inter-NAIP temporal-change metrics.

A delta metric takes two :class:`MetricResult` instances on identical grids
and returns ``a.array - b.array`` as a new MetricResult named
``delta_<source>``. Grid alignment is strict: if ``transform``, ``crs``,
``window_m``, or array shape differ, a :class:`ValueError` is raised. No
resampling is performed.

The five V5 inter-NAIP deltas (dCF, dGF, dCW90, dCD, dEdge) are registered
as named two-argument callables via
:func:`csdv_core.metrics.registry.register_delta`. They are thin wrappers
over :func:`metric_delta` that also validate the source metric name.
"""

from __future__ import annotations

import logging

import numpy as np

from csdv_core.metrics._result import MetricResult, make_result
from csdv_core.metrics.registry import register_delta

logger = logging.getLogger(__name__)


def _check_aligned(a: MetricResult, b: MetricResult) -> None:
    if a.array.shape != b.array.shape:
        raise ValueError(
            f"shape mismatch: {a.array.shape} vs {b.array.shape}",
        )
    if a.transform != b.transform:
        raise ValueError("transform mismatch between MetricResults")
    if a.crs != b.crs:
        raise ValueError(f"crs mismatch: {a.crs!r} vs {b.crs!r}")
    if a.window_m != b.window_m:
        raise ValueError(f"window_m mismatch: {a.window_m} vs {b.window_m}")


def metric_delta(a: MetricResult, b: MetricResult) -> MetricResult:
    """Return ``a - b`` as a new MetricResult named ``delta_<a.name>``.

    Args:
        a: Later (or first) MetricResult.
        b: Earlier (or second) MetricResult on the same grid as ``a``.

    Returns:
        MetricResult with ``array = a.array - b.array``. NaN propagates.
        Units copy from ``a``. The result name is ``f"delta_{a.name}"``.

    Raises:
        ValueError: If the two inputs differ in shape, transform, CRS, or
            window_m. No resampling is performed.
    """
    _check_aligned(a, b)
    diff = (a.array - b.array).astype(np.float32)
    logger.info(
        "metric_delta: %s - %s on %dx%d grid",
        a.name,
        b.name,
        a.array.shape[0],
        a.array.shape[1],
    )
    return make_result(
        f"delta_{a.name}",
        diff,
        transform=a.transform,
        crs=a.crs,
        window_m=a.window_m,
        params={"source": a.name, "minuend_params": dict(a.params)},
        units=a.units,
    )


def _named_delta(expected: str, a: MetricResult, b: MetricResult) -> MetricResult:
    if a.name != expected or b.name != expected:
        raise ValueError(
            f"expected both inputs to be {expected!r}; got {a.name!r} and {b.name!r}",
        )
    return metric_delta(a, b)


@register_delta("delta_crown_fraction")
def delta_crown_fraction(a: MetricResult, b: MetricResult) -> MetricResult:
    """V5 dCF: change in crown fraction between two NAIP epochs."""
    return _named_delta("crown_fraction", a, b)


@register_delta("delta_gap_fraction")
def delta_gap_fraction(a: MetricResult, b: MetricResult) -> MetricResult:
    """V5 dGF: change in gap fraction between two NAIP epochs."""
    return _named_delta("gap_fraction", a, b)


@register_delta("delta_crown_p90")
def delta_crown_p90(a: MetricResult, b: MetricResult) -> MetricResult:
    """V5 dCW90: change in crown width P90 between two NAIP epochs."""
    return _named_delta("crown_p90", a, b)


@register_delta("delta_crown_count")
def delta_crown_count(a: MetricResult, b: MetricResult) -> MetricResult:
    """V5 dCD: change in crown density/count between two NAIP epochs."""
    return _named_delta("crown_count", a, b)


@register_delta("delta_edge_density")
def delta_edge_density(a: MetricResult, b: MetricResult) -> MetricResult:
    """V5 dEdge: change in canopy edge density between two NAIP epochs."""
    return _named_delta("edge_density", a, b)


__all__ = [
    "metric_delta",
    "delta_crown_fraction",
    "delta_gap_fraction",
    "delta_crown_p90",
    "delta_crown_count",
    "delta_edge_density",
]
