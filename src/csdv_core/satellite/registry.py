"""csdv_core.satellite.registry — the three satellite registries.

A satellite metric is assembled from three independent pieces, and each one is
a different kind of thing, so each gets its own registry:

* A **sensor** says where the archive is and how to turn its native bands into
  harmonized surface reflectance. It is fully described by data, so
  :func:`register_sensor` takes a :class:`SensorSpec` rather than decorating a
  callable. Adding Sentinel-2 or HLS is then a data change.
* An **index** is band algebra. It is a function, so it registers by decorator
  like :mod:`csdv_core.metrics.registry` does.
* An **annual metric** turns one stand's observation series into one number per
  year. Also a function, and its defaults come from ``satellite.yaml`` the same
  way metric defaults come from ``metrics.yaml``.

Keeping them apart means a call site never has to ask what kind of name it is
holding, and a new sensor cannot accidentally shadow an index.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_SENSORS: dict[str, SensorSpec] = {}
_INDICES: dict[str, IndexSpec] = {}
_ANNUAL: dict[str, AnnualMetricSpec] = {}

__all__ = [
    "AnnualMetricSpec",
    "IndexSpec",
    "SensorSpec",
    "get_annual",
    "get_index",
    "get_sensor",
    "list_annual_metrics",
    "list_indices",
    "list_sensors",
    "register_annual",
    "register_index",
    "register_sensor",
]


# --------------------------------------------------------------------------
# Specs
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SensorSpec:
    """One sensor and everything needed to harmonize its archive.

    Attributes:
        name: Short code used in config and in the ``sensor`` column, e.g.
            ``"L5"``.
        platform: Human-readable platform and instrument.
        collection_id: Earth Engine collection, e.g.
            ``"LANDSAT/LT05/C02/T1_L2"``.
        bands: Harmonized name to native band name, e.g.
            ``{"red": "SR_B3", "nir": "SR_B4"}``. The Landsat 4/5/7 to 8/9
            shift lives entirely here, which is why it is data and not code.
        reflectance_scale: Multiplier applied to the raw digital number.
        reflectance_offset: Offset added after scaling.
        qa_band: Per-pixel quality band.
        qa_rule: Dispatch key naming how ``qa_band`` is interpreted. Landsat
            Collection 2 uses ``"landsat_c2_qa_pixel"``; Sentinel-2 would use
            its own.
        saturation_band: Band carrying per-band saturation flags, or None.
        saturation_bits: Harmonized band name to its bit in
            ``saturation_band``. Stored rather than derived from the native
            band number, because that rule does not hold outside Landsat.
        cloud_property: Scene-level cloud property used for the cheap
            prefilter.
        first_year: First year the sensor acquired usable data.
        last_year: Last year, or None while it is still acquiring.
        notes: Anything a reader of the output needs to know.
    """

    name: str
    platform: str
    collection_id: str
    bands: Mapping[str, str]
    reflectance_scale: float = 2.75e-05
    reflectance_offset: float = -0.2
    qa_band: str = "QA_PIXEL"
    qa_rule: str = "landsat_c2_qa_pixel"
    saturation_band: str | None = "QA_RADSAT"
    saturation_bits: Mapping[str, int] = field(default_factory=dict)
    cloud_property: str = "CLOUD_COVER_LAND"
    first_year: int = 1984
    last_year: int | None = None
    notes: str = ""


@dataclass(frozen=True)
class IndexSpec:
    """A spectral index as pure band algebra, plus the bands it consumes.

    Attributes:
        name: Index name; also the band name it produces and the observation
            column it lands in.
        fn: ``Mapping[str, band] -> band``. Written against the band-algebra
            surface in :mod:`csdv_core.satellite.indices` so the same code runs
            on ``ee.Image`` and on NumPy.
        bands: Harmonized band names the formula reads, in the order it reads
            them. Used to fail loudly when an index is requested against a
            sensor that lacks a band.
        valid_range: Physically possible range. Values outside it are dropped
            by the observation filter.
        units: Display units.
        description: One line for ``csdv satellite list``.
    """

    name: str
    fn: Callable[[Mapping[str, Any]], Any]
    bands: tuple[str, ...]
    valid_range: tuple[float, float] = (-1.0, 1.0)
    units: str = "index"
    description: str = ""


@dataclass(frozen=True)
class AnnualMetricSpec:
    """A registered per-stand-per-year metric and its YAML-resolved defaults.

    Attributes:
        name: Metric name. Must match the spelling used in ``stages.yaml`` and
            ``trajectories.yaml``.
        fn: ``(observations, year=..., **params) -> AnnualResult``.
        index: Observation column the metric reads, e.g. ``"ndvi"``.
        defaults: ``annual_defaults`` from ``satellite.yaml`` with the
            per-metric ``params`` merged over the top.
        units: Display units.
        description: One line for ``csdv satellite list``.
    """

    name: str
    fn: Callable[..., Any]
    index: str
    defaults: dict[str, Any]
    units: str = ""
    description: str = ""


# --------------------------------------------------------------------------
# Sensors
# --------------------------------------------------------------------------
def register_sensor(spec: SensorSpec) -> SensorSpec:
    """Register a sensor and return it, so a module constant can wrap the call.

    Takes data rather than decorating a callable, which is the one deliberate
    departure from :mod:`csdv_core.metrics.registry`. Everything a sensor needs
    is in the spec, so adding one should never require writing a function.
    """
    existing = _SENSORS.get(spec.name)
    if existing is not None and existing != spec:
        logger.debug("sensor %s re-registered (was %r)", spec.name, existing)
    _SENSORS[spec.name] = spec
    return spec


def get_sensor(name: str) -> SensorSpec:
    """Return the :class:`SensorSpec` for ``name`` or raise ``KeyError``."""
    if name not in _SENSORS:
        raise KeyError(f"Unknown sensor {name!r}. Known: {sorted(_SENSORS)}")
    return _SENSORS[name]


def list_sensors() -> list[str]:
    """Return registered sensor names, sorted."""
    return sorted(_SENSORS)


# --------------------------------------------------------------------------
# Indices
# --------------------------------------------------------------------------
def register_index(
    name: str,
    *,
    bands: tuple[str, ...],
    valid_range: tuple[float, float] = (-1.0, 1.0),
    units: str = "index",
    description: str = "",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: register a band-algebra callable under ``name``."""

    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in _INDICES and _INDICES[name].fn is not fn:
            logger.debug("index %s re-registered (was %r)", name, _INDICES[name].fn)
        _INDICES[name] = IndexSpec(
            name=name,
            fn=fn,
            bands=tuple(bands),
            valid_range=valid_range,
            units=units,
            description=description or (fn.__doc__ or "").strip().split("\n")[0],
        )
        return fn

    return _wrap


def get_index(name: str) -> IndexSpec:
    """Return the :class:`IndexSpec` for ``name`` or raise ``KeyError``."""
    if name not in _INDICES:
        raise KeyError(f"Unknown index {name!r}. Known: {sorted(_INDICES)}")
    return _INDICES[name]


def list_indices() -> list[str]:
    """Return registered index names, sorted."""
    return sorted(_INDICES)


# --------------------------------------------------------------------------
# Annual metrics
# --------------------------------------------------------------------------
def _resolve_annual_defaults(name: str) -> tuple[str, dict[str, Any], str]:
    """Merge ``annual_defaults`` with per-metric ``params`` from satellite.yaml.

    Returns:
        ``(index, params, units)``. Unlike
        :func:`csdv_core.metrics.registry._resolve_defaults`, the global block
        is dumped rather than enumerated key by key, so adding a shared default
        is a YAML and model edit with no code change here.
    """
    from csdv_core.config import load_satellite

    cfg = load_satellite()
    merged: dict[str, Any] = dict(cfg.annual_defaults.model_dump())
    entry = cfg.annual_metrics.get(name)
    if entry is None:
        logger.debug("annual metric %s has no satellite.yaml entry", name)
        return "ndvi", merged, ""
    merged.update(entry.params)
    return entry.index, merged, entry.units


def register_annual(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: register a per-stand-per-year metric callable under ``name``.

    Defaults are resolved lazily in :func:`get_annual` rather than here, so the
    registry can be imported without reading configuration.
    """

    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in _ANNUAL and _ANNUAL[name].fn is not fn:
            logger.debug("annual metric %s re-registered", name)
        _ANNUAL[name] = AnnualMetricSpec(
            name=name,
            fn=fn,
            index="",
            defaults={},
            description=(fn.__doc__ or "").strip().split("\n")[0],
        )
        return fn

    return _wrap


def get_annual(name: str) -> AnnualMetricSpec:
    """Return the :class:`AnnualMetricSpec` for ``name`` or raise ``KeyError``.

    The registered entry carries the callable; the index, defaults and units
    are filled from ``satellite.yaml`` on every call, so a config reload takes
    effect without re-importing the module.
    """
    if name not in _ANNUAL:
        raise KeyError(f"Unknown annual metric {name!r}. Known: {sorted(_ANNUAL)}")
    stub = _ANNUAL[name]
    index, defaults, units = _resolve_annual_defaults(name)
    return AnnualMetricSpec(
        name=name,
        fn=stub.fn,
        index=index,
        defaults=defaults,
        units=units,
        description=stub.description,
    )


def list_annual_metrics() -> list[str]:
    """Return registered annual metric names, sorted."""
    return sorted(_ANNUAL)
