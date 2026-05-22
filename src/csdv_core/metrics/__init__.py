"""csdv_core.metrics - Pure-function metric implementations.

Importing this package triggers each metric module's ``@register`` calls so
that :func:`csdv_core.metrics.registry.get_metric` can resolve them.
"""

from __future__ import annotations

from csdv_core.metrics import (  # noqa: F401  (register side effects)
    cover,
    crown,
    deltas,
    gap,
    spatial,
    texture,
)
from csdv_core.metrics._result import MetricResult, make_result
from csdv_core.metrics.registry import (
    MetricSpec,
    get_delta,
    get_metric,
    list_deltas,
    list_metrics,
    register,
    register_delta,
)

__all__ = [
    "MetricResult",
    "make_result",
    "MetricSpec",
    "get_metric",
    "list_metrics",
    "register",
    "get_delta",
    "list_deltas",
    "register_delta",
]
