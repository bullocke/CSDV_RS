"""Pydantic models for packaged YAML configuration.

Private module. Public access is via the loader functions in
``csdv_core.config`` (``load_sites``, ``load_metrics``, ...).

All models use ``extra="forbid"`` so unknown keys in YAML fail loudly.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_Strict = ConfigDict(extra="forbid")


class _Base(BaseModel):
    model_config = _Strict


# ---------------------------------------------------------------------------
# sites.yaml
# ---------------------------------------------------------------------------


class NaipSpec(_Base):
    years: list[int] = Field(default_factory=list)
    doqq_ids: list[str] = Field(default_factory=list)


class NeonSpec(_Base):
    site_code: str
    aop_years: list[int] = Field(default_factory=list)


class SiteEntry(_Base):
    name: str
    category: Literal["neon", "managed", "secondary"]
    state: str
    crs: str = "EPSG:5070"
    bbox: tuple[float, float, float, float] | None = None
    center_lonlat: tuple[float, float] | None = None
    naip: NaipSpec = Field(default_factory=NaipSpec)
    neon: NeonSpec | None = None
    notes: str = ""


class SitesConfig(_Base):
    sites: dict[str, SiteEntry]

    def get(self, code: str) -> SiteEntry:
        """Return the site entry for ``code`` or raise ``KeyError``."""
        try:
            return self.sites[code]
        except KeyError as exc:
            raise KeyError(f"Unknown site code: {code!r}") from exc


# ---------------------------------------------------------------------------
# metrics.yaml
# ---------------------------------------------------------------------------


class MetricDefaults(_Base):
    window_sizes_m: list[float]
    chm_gap_threshold_m: float
    min_crowns_per_window: int


class MetricParams(_Base):
    params: dict[str, float | int | str | bool | None] = Field(default_factory=dict)


class ChmInferenceDefaults(_Base):
    aoi_half_km: float = 2.5
    chip_size: int = 432
    chip_overlap: float = 0.2


class MetricsConfig(_Base):
    defaults: MetricDefaults
    metrics: dict[str, MetricParams] = Field(default_factory=dict)
    chm_inference: ChmInferenceDefaults = Field(default_factory=ChmInferenceDefaults)


# ---------------------------------------------------------------------------
# site_types.yaml
# ---------------------------------------------------------------------------


class Predicate(_Base):
    var: str
    op: Literal["<", "<=", "==", "!=", ">=", ">"]
    value: float | int | str | bool | None


class SiteTypeRule(_Base):
    name: str
    group: str
    rules: list[Predicate] = Field(default_factory=list)


class SiteTypesConfig(_Base):
    site_types: dict[str, SiteTypeRule]


# ---------------------------------------------------------------------------
# stages.yaml
# ---------------------------------------------------------------------------


class Range(_Base):
    min: float | None = None
    max: float | None = None


class StageEnvelopes(_Base):
    envelopes: dict[str, dict[str, Range]] = Field(default_factory=dict)


class StagesConfig(_Base):
    stages: dict[str, StageEnvelopes]
    stage_order: list[str] = Field(default_factory=list)
    stage_codes: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# trajectories.yaml
# ---------------------------------------------------------------------------


class TrajectoryPredicate(_Base):
    """Predicate over a multi-date stage/metric cube.

    Attributes:
        dim: Which input the predicate consults. ``"stage"`` reads the stage
            cube; ``"stage_delta"`` reads the per-pixel difference between
            consecutive dates; ``"site_type"`` reads the year-invariant
            site-type raster; ``"metric"`` reads a named metric cube;
            ``"persistence"`` reads a metric cube and computes the fraction
            of dates where the metric crosses ``threshold``.
        var: Required when ``dim`` is ``"metric"`` or ``"persistence"``.
            The metric name. For ``"stage"``/``"stage_delta"`` the var is
            implicit and must be left unset.
        reducer: Time-axis reduction. ``"latest"``/``"earliest"`` pick a
            single date; ``"all"``/``"any"`` require the predicate to hold
            on all/any dates; ``"mean"``/``"min"``/``"max"`` reduce to a
            scalar per pixel; ``"scalar"`` skips reduction (for inputs that
            are already 2-D, e.g. ``site_type`` or persistence outputs).
        op: Comparison operator. ``"in"`` requires ``value`` to be a list
            and treats it as a membership test.
        value: Right-hand side. May be a list when ``op == "in"``. Stage
            values are stage abbreviation strings (resolved via
            ``stage_codes``). ``None`` means the predicate is a placeholder
            (always evaluates to False).
        threshold: For ``dim == "persistence"`` only. The metric is first
            compared to ``threshold`` per date using ``op``, producing a
            per-pixel persistence fraction in [0, 1] which is then compared
            to ``value`` with ``>=``.
    """

    dim: Literal[
        "stage",
        "stage_delta",
        "site_type",
        "metric",
        "persistence",
    ] = "metric"
    var: str | None = None
    reducer: Literal[
        "latest",
        "earliest",
        "all",
        "any",
        "mean",
        "min",
        "max",
        "scalar",
    ] = "scalar"
    op: Literal["<", "<=", "==", "!=", ">=", ">", "in"]
    value: float | int | str | bool | list[float | int | str | bool] | None
    threshold: float | int | None = None


class TrajectoryRule(_Base):
    name: str
    group: Literal["DS", "EF", "LC", "FC"]
    signature: list[TrajectoryPredicate] = Field(default_factory=list)


class TrajectoriesConfig(_Base):
    trajectories: dict[str, TrajectoryRule]
    trajectory_order: list[str] = Field(default_factory=list)
    trajectory_codes: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# satellite.yaml
# ---------------------------------------------------------------------------
#
# Sensor, index and annual-metric *names* are deliberately not validated here
# against the registries in ``csdv_core.satellite``. Doing so would make
# ``config`` import ``satellite``, which imports ``config``. Names are checked
# in ``get_sensor`` / ``get_index`` / ``get_annual``, the same place
# ``get_metric`` checks them, and a unit test asserts the YAML names and the
# registered names are the same set.


#: QA_PIXEL bit names, Landsat Collection 2. The layout is identical across
#: Landsat 4 to 9; ``cirrus`` is always 0 on the TM and ETM+ sensors.
QA_PIXEL_BIT_NAMES = (
    "fill",
    "dilated_cloud",
    "cirrus",
    "cloud",
    "cloud_shadow",
    "snow",
    "clear",
    "water",
)


class EarthEngineConfig(_Base):
    project: str = "dyce-biomass"
    high_volume: bool = True
    deadline_ms: int = Field(default=300_000, ge=1_000)
    workload_tag: str = "csdv-satellite"
    max_workers: int = Field(default=4, ge=1, le=8)


class ExtractionConfig(_Base):
    sensors: list[str]
    indices: list[str]
    start_year: int = Field(ge=1982)
    end_year: int
    scale_m: float = Field(default=30.0, gt=0)
    tile_scale: int = Field(default=2, ge=1, le=16)
    page_size: int = Field(default=5000, ge=1)
    chunk: Literal["year", "half_year", "month"] = "year"
    max_scene_cloud_cover: float = Field(default=80.0, ge=0.0, le=100.0)
    aoi_buffer_m: float = Field(default=1000.0, ge=0.0)
    simplify_tolerance_m: float = Field(default=1.0, ge=0.0)
    max_attempts: int = Field(default=4, ge=1)
    backoff_base_s: float = Field(default=2.0, gt=1.0)
    backoff_max_s: float = Field(default=60.0, gt=0.0)

    @model_validator(mode="after")
    def _years_ordered(self) -> ExtractionConfig:
        if self.end_year < self.start_year:
            raise ValueError(
                f"end_year {self.end_year} precedes start_year {self.start_year}"
            )
        return self


class QaConfig(_Base):
    mask_bits: list[Literal[QA_PIXEL_BIT_NAMES]]  # type: ignore[valid-type]
    mask_cloud_confidence_medium: bool = False
    mask_saturated_bands: bool = True
    reflectance_min: float = 0.0
    reflectance_max: float = 1.0
    index_valid_range: tuple[float, float] = (-1.0, 1.0)
    min_pixels: int = Field(default=4, ge=0)
    min_effective_pixels: float = Field(default=2.0, ge=0.0)
    min_coverage_fraction: float = Field(default=0.60, ge=0.0, le=1.0)
    min_area_m2: float = Field(default=3600.0, ge=0.0)

    @model_validator(mode="after")
    def _range_ordered(self) -> QaConfig:
        lo, hi = self.index_valid_range
        if hi <= lo:
            raise ValueError(
                f"index_valid_range must be increasing, got {self.index_valid_range}"
            )
        return self


class AnnualDefaults(_Base):
    doy_min: int = Field(default=1, ge=1, le=366)
    doy_max: int = Field(default=366, ge=1, le=366)
    min_obs: int = Field(default=6, ge=1)


class AnnualMetricEntry(_Base):
    index: str
    units: str = ""
    params: dict[str, float | int | str | bool | None] = Field(default_factory=dict)


class SatelliteConfig(_Base):
    earth_engine: EarthEngineConfig = Field(default_factory=EarthEngineConfig)
    extraction: ExtractionConfig
    qa: QaConfig
    annual_defaults: AnnualDefaults = Field(default_factory=AnnualDefaults)
    annual_metrics: dict[str, AnnualMetricEntry] = Field(default_factory=dict)
