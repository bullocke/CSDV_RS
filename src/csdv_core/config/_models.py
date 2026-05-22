"""Pydantic models for packaged YAML configuration.

Private module. Public access is via the loader functions in
``csdv_core.config`` (``load_sites``, ``load_metrics``, ...).

All models use ``extra="forbid"`` so unknown keys in YAML fail loudly.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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
