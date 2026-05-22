"""Result type returned by all metric functions.

Frozen so metric outputs can be cached, hashed by identity, and treated as
immutable downstream. ``array`` is read-only via NumPy's ``writeable`` flag
when constructed through :func:`make_result`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MetricResult:
    """Output of a metric computation.

    Attributes:
        name: Registered metric name (e.g. ``"gap_fraction"``).
        array: 2-D float32 result, one cell per analysis window.
        transform: Affine transform of the output grid (one pixel per window).
        crs: CRS string (e.g. ``"EPSG:5070"``).
        window_m: Analysis window side length in meters.
        params: Parameters used to compute the result.
        units: Human-readable units (e.g. ``"fraction"``, ``"m"``).
    """

    name: str
    array: np.ndarray
    transform: Any
    crs: str
    window_m: float
    params: dict[str, Any] = field(default_factory=dict)
    units: str = ""


def make_result(
    name: str,
    array: np.ndarray,
    *,
    transform: Any,
    crs: str,
    window_m: float,
    params: dict[str, Any] | None = None,
    units: str = "",
) -> MetricResult:
    """Construct a :class:`MetricResult` with a read-only float32 array."""
    arr = np.asarray(array, dtype=np.float32)
    arr.setflags(write=False)
    return MetricResult(
        name=name,
        array=arr,
        transform=transform,
        crs=crs,
        window_m=float(window_m),
        params=dict(params or {}),
        units=units,
    )
