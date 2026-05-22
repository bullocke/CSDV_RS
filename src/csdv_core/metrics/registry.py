"""csdv_core.metrics.registry - Decorator-based metric registry.

Each metric module decorates its public callables with :func:`register`. At
import time the registry collects ``name -> callable`` mappings; the
:class:`MetricSpec` returned by :func:`get_metric` also exposes default
parameters resolved from ``config/metrics.yaml`` (per-metric ``params``
overrides plus the global ``defaults`` block).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, Callable[..., Any]] = {}
_DELTA_REGISTRY: dict[str, Callable[..., Any]] = {}


@dataclass(frozen=True)
class MetricSpec:
    """A registered metric and its YAML-resolved default parameters."""

    name: str
    fn: Callable[..., Any]
    defaults: dict[str, Any]


def register(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Class/function decorator: register a metric callable under ``name``."""

    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in _REGISTRY and _REGISTRY[name] is not fn:
            logger.debug("metric %s re-registered (was %r)", name, _REGISTRY[name])
        _REGISTRY[name] = fn
        return fn

    return _wrap


def _resolve_defaults(name: str) -> dict[str, Any]:
    """Merge global ``defaults`` with per-metric ``params`` from metrics.yaml."""
    from csdv_core.config import load_metrics

    cfg = load_metrics()
    merged: dict[str, Any] = {
        "window_sizes_m": list(cfg.defaults.window_sizes_m),
        "chm_gap_threshold_m": cfg.defaults.chm_gap_threshold_m,
        "min_crowns_per_window": cfg.defaults.min_crowns_per_window,
    }
    entry = cfg.metrics.get(name)
    if entry is not None:
        merged.update(entry.params)
    return merged


def get_metric(name: str) -> MetricSpec:
    """Return the :class:`MetricSpec` for ``name`` or raise ``KeyError``."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown metric {name!r}. Known: {sorted(_REGISTRY)}")
    return MetricSpec(name=name, fn=_REGISTRY[name], defaults=_resolve_defaults(name))


def list_metrics() -> list[str]:
    """Return registered metric names, sorted."""
    return sorted(_REGISTRY)


def register_delta(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: register a two-input delta callable under ``name``.

    Delta callables take two :class:`MetricResult` instances and return a
    :class:`MetricResult`. They are kept in a separate registry from
    single-input metrics so their distinct call signature stays visible.
    """

    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in _DELTA_REGISTRY and _DELTA_REGISTRY[name] is not fn:
            logger.debug("delta %s re-registered (was %r)", name, _DELTA_REGISTRY[name])
        _DELTA_REGISTRY[name] = fn
        return fn

    return _wrap


def get_delta(name: str) -> Callable[..., Any]:
    """Return the registered delta callable for ``name`` or raise ``KeyError``."""
    if name not in _DELTA_REGISTRY:
        raise KeyError(f"Unknown delta {name!r}. Known: {sorted(_DELTA_REGISTRY)}")
    return _DELTA_REGISTRY[name]


def list_deltas() -> list[str]:
    """Return registered delta names, sorted."""
    return sorted(_DELTA_REGISTRY)


__all__ = [
    "register",
    "get_metric",
    "list_metrics",
    "register_delta",
    "get_delta",
    "list_deltas",
    "MetricSpec",
]
