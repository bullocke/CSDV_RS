"""csdv_core.satellite.annual — observation series to one number per year.

The extraction step produces one row per stand per scene. This turns that into
one row per stand per year, which is what the classification layers consume:
the stage envelopes read a value at each NAIP date, and the trajectory rules
read the series.

Three metrics ship. ``ndvi_mean`` is the growing-season level.
``ndvi_seasonal_amplitude`` is the size of the annual cycle, from a
single-harmonic fit. ``ndvi_trend`` is the direction of travel, from a
Theil-Sen slope over a trailing window. All three are named exactly as
``stages.yaml`` and ``trajectories.yaml`` reference them.

Nothing here touches Earth Engine or the filesystem, so the part of the module
where the science lives is testable against synthetic frames. Every metric
returns NaN with a stated reason rather than raising or guessing, which is the
same contract :class:`csdv_core.zonal.record.StandMetricRecord` offers: a blank
in the output table can always be traced back to a cause.
"""

from __future__ import annotations

import calendar
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from csdv_core.satellite.registry import (
    get_annual,
    list_annual_metrics,
    register_annual,
)

logger = logging.getLogger(__name__)

#: Support columns are prefixed with this on the way into a flat row. The
#: satellite and zonal tables are joined on ``(stand_id, year)``, and the zonal
#: side already owns the bare ``support_`` prefix and the ``unavailable``
#: column, so a distinct prefix here means the join can never silently collide.
SUPPORT_PREFIX = "sat_"

#: Column carrying the joined reasons, so it does not clash with the zonal
#: table's ``unavailable``.
REASON_COLUMN = "satellite_unavailable"

#: Total sum of squares below which a series is treated as flat and ``r2`` is
#: reported undefined. An index lives in [-1, 1], so this is far below any real
#: signal and far above the floating-point noise a constant series leaves.
_FLAT_SERIES_SS_TOT = 1e-12

__all__ = [
    "REASON_COLUMN",
    "SUPPORT_PREFIX",
    "AnnualResult",
    "HarmonicFit",
    "StandYearRecord",
    "annual_table",
    "filter_observations",
    "fit_single_harmonic",
    "ndvi_mean",
    "ndvi_seasonal_amplitude",
    "ndvi_trend",
    "records_to_annual_frame",
    "stand_year_record",
    "theil_sen_slope",
]


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class HarmonicFit:
    """Single-harmonic least-squares fit of one index series in one year.

    Model ``y(t) = offset + cos_coef*cos(2*pi*t) + sin_coef*sin(2*pi*t)`` with
    ``t = doy / days_in_year``, so the fundamental period is exactly one year
    and the coefficients are directly interpretable.

    Attributes:
        offset: The harmonic mean level, beta 0.
        cos_coef: beta 1.
        sin_coef: beta 2.
        amplitude: ``2 * hypot(cos_coef, sin_coef)``, reported peak to trough.
        phase_doy: Day of year of the modelled maximum. A free sanity check: a
            southern Indiana hardwood stand peaks near day 190 and maize near
            day 215, so a fitted peak in February means the fit is junk even
            when every numeric guard passed.
        n_obs: Observations the fit used.
        doy_span: Last minus first day of year among those observations.
        condition: Condition number of the design matrix.
        rmse: Root mean squared residual.
        r2: Coefficient of determination, NaN when the series is flat.
        reason: Empty when the fit is usable, otherwise why it is not.
    """

    offset: float
    cos_coef: float
    sin_coef: float
    amplitude: float
    phase_doy: float
    n_obs: int
    doy_span: float
    condition: float
    rmse: float
    r2: float
    reason: str = ""


@dataclass(frozen=True)
class AnnualResult:
    """One annual metric for one stand, with the support to read it by."""

    name: str
    value: float
    support: Mapping[str, float] = field(default_factory=dict)
    units: str = ""
    reason: str = ""


@dataclass(frozen=True)
class StandYearRecord:
    """Satellite metrics for one stand in one year.

    Shaped like :class:`csdv_core.zonal.record.StandMetricRecord` so the two
    tables flatten the same way and join without translation.

    Attributes:
        stand_id: Stand identifier.
        year: Calendar year.
        area_m2: Stand area, carried so a reader can see how many Landsat
            pixels a value rests on.
        n_observations: Clear observations that survived quality control.
        n_observations_raw: Observations before quality control.
        sensor_mix: Comma-separated sensors contributing, e.g. ``"L5,L7"``.
            A metric that moves only in 2013 or 2021 is more likely a change of
            fleet than a change on the ground, and this is what makes that
            visible.
        metrics: Metric name to value. NaN where not computed.
        support: Auxiliary numbers, flattened with the ``sat_`` prefix.
        reasons: Metric name to why its value is NaN.
    """

    stand_id: str
    year: int
    area_m2: float
    n_observations: int
    n_observations_raw: int
    sensor_mix: str = ""
    metrics: Mapping[str, float] = field(default_factory=dict)
    support: Mapping[str, float] = field(default_factory=dict)
    reasons: Mapping[str, str] = field(default_factory=dict)

    def value(self, name: str) -> float:
        """Return a metric value, or NaN if it was not computed."""
        return float(self.metrics.get(name, float("nan")))

    def to_row(self) -> dict[str, Any]:
        """Flatten to one row, prefixing support columns with ``sat_``."""
        row: dict[str, Any] = {
            "stand_id": self.stand_id,
            "year": self.year,
            "area_m2": self.area_m2,
            f"{SUPPORT_PREFIX}n_observations": self.n_observations,
            f"{SUPPORT_PREFIX}n_observations_raw": self.n_observations_raw,
            f"{SUPPORT_PREFIX}sensor_mix": self.sensor_mix,
        }
        row.update({name: float(value) for name, value in self.metrics.items()})
        row.update({f"{SUPPORT_PREFIX}{k}": v for k, v in self.support.items()})
        row[REASON_COLUMN] = "; ".join(
            f"{name}: {why}" for name, why in sorted(self.reasons.items())
        )
        return row


# --------------------------------------------------------------------------
# Quality control
# --------------------------------------------------------------------------
def filter_observations(
    obs: pd.DataFrame,
    *,
    index: str = "ndvi",
    min_pixels: int = 4,
    min_effective_pixels: float = 2.0,
    min_coverage_fraction: float = 0.60,
    min_area_m2: float = 3600.0,
    valid_range: tuple[float, float] = (-1.0, 1.0),
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop observations that cannot carry a stand-level value.

    The gates run in a fixed order and each one is counted separately, so a
    stand-year that lost fourteen of sixteen observations can say which gate
    took them.

    ``min_coverage_fraction`` is the one doing most of the work. It is the
    share of the stand that survived masking, so it falls in proportion to area
    lost to cloud, to shadow, or to a Landsat 7 scan-line gap alike. One gate,
    three failure modes, which is why the SLC-off era is kept rather than
    excluded wholesale.

    Args:
        obs: Observation frame from the extraction step.
        index: Index column being filtered, e.g. ``"ndvi"``.
        min_pixels: Minimum touching-pixel count.
        min_effective_pixels: Minimum sum of area-coverage weights.
        min_coverage_fraction: Minimum share of the stand that survived masking.
        min_area_m2: Minimum stand area. Below roughly four Landsat pixels the
            value is dominated by the point spread of neighbouring pixels
            rather than by the stand, so it is voided rather than reported.
        valid_range: Physically possible range for the index.

    Returns:
        ``(kept, counts)``. ``counts`` names every cause and its tally.
    """
    if obs.empty:
        return obs.copy(), {}

    frame = obs
    counts: dict[str, int] = {}

    def _drop(mask: pd.Series, cause: str) -> None:
        nonlocal frame
        n = int(mask.sum())
        if n:
            counts[cause] = counts.get(cause, 0) + n
            frame = frame.loc[~mask]

    lo, hi = valid_range
    values = pd.to_numeric(frame[index], errors="coerce")
    _drop(values.isna(), "index null (every pixel masked)")

    values = pd.to_numeric(frame[index], errors="coerce")
    _drop((values < lo) | (values > hi), f"index outside {lo} to {hi}")

    _drop(
        pd.to_numeric(frame["n_pixels"], errors="coerce").fillna(0) < min_pixels,
        f"fewer than {min_pixels} pixels",
    )
    _drop(
        pd.to_numeric(frame["pixel_weight_sum"], errors="coerce").fillna(0.0)
        < min_effective_pixels,
        f"effective pixels below {min_effective_pixels}",
    )
    _drop(
        pd.to_numeric(frame["coverage_fraction"], errors="coerce").fillna(0.0)
        < min_coverage_fraction,
        f"stand coverage below {min_coverage_fraction:.2f} (cloud, shadow or SLC gap)",
    )
    _drop(
        pd.to_numeric(frame["area_m2"], errors="coerce").fillna(0.0) < min_area_m2,
        f"stand area below {min_area_m2:.0f} m2 for 30 m metrics",
    )

    if counts:
        logger.info(
            "Quality control kept %d of %d observations (%s)",
            len(frame),
            len(obs),
            ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())),
        )
    return frame.reset_index(drop=True), counts


# --------------------------------------------------------------------------
# Fitting primitives
# --------------------------------------------------------------------------
def _days_in_year(year: int) -> int:
    return 366 if calendar.isleap(int(year)) else 365


def fit_single_harmonic(
    doy: Sequence[float],
    values: Sequence[float],
    *,
    year: int,
    min_obs: int = 6,
    min_doy_span: float = 150.0,
    max_condition: float = 30.0,
    max_amplitude: float = 1.5,
) -> HarmonicFit:
    """Fit one annual harmonic to an irregularly sampled index series.

    Amplitude is reported **peak to trough**, ``2 * hypot(b1, b2)``, because
    that is the number a reader expects when they see the word: maize running
    from 0.2 bare soil to 0.9 canopy reads as 0.7. The theoretical maximum is
    2.0 since NDVI lies in [-1, 1].

    Guards are evaluated in a fixed order, and the order matters. The span check
    fires before the conditioning check so that eight observations all inside
    July report something a reader can act on rather than a bare condition
    number.

    ``r2`` is recorded but deliberately not a guard. A closed canopy has a
    genuine but small seasonal signal and a modest ``r2``, so gating on it would
    preferentially void exactly the forest stands this project is about.

    Args:
        doy: Day of year for each observation.
        values: Index value for each observation.
        year: The calendar year, which sets the period (366 days in a leap year).
        min_obs: Minimum observations. Three parameters plus three residual
            degrees of freedom.
        min_doy_span: Minimum first-to-last day span.
        max_condition: Maximum design-matrix condition number.
        max_amplitude: Maximum credible amplitude.

    Returns:
        A :class:`HarmonicFit`. When a guard fires, ``reason`` says which and
        the diagnostic fields are still populated, so the support columns show
        how far short the year fell.
    """
    doy_arr = np.asarray(doy, dtype=float)
    val_arr = np.asarray(values, dtype=float)
    finite = np.isfinite(doy_arr) & np.isfinite(val_arr)
    doy_arr, val_arr = doy_arr[finite], val_arr[finite]

    n_obs = int(doy_arr.size)
    span = float(doy_arr.max() - doy_arr.min()) if n_obs else 0.0
    empty = HarmonicFit(
        offset=float("nan"),
        cos_coef=float("nan"),
        sin_coef=float("nan"),
        amplitude=float("nan"),
        phase_doy=float("nan"),
        n_obs=n_obs,
        doy_span=span,
        condition=float("nan"),
        rmse=float("nan"),
        r2=float("nan"),
    )

    if n_obs < min_obs:
        return _with_reason(empty, f"n_obs={n_obs} < min_obs={min_obs}")
    if span < min_doy_span:
        return _with_reason(
            empty,
            f"observations span {span:.0f} days < min_doy_span={min_doy_span:.0f}",
        )

    period = float(_days_in_year(year))
    t = 2.0 * np.pi * doy_arr / period
    design = np.column_stack([np.ones_like(t), np.cos(t), np.sin(t)])
    condition = float(np.linalg.cond(design))
    if condition > max_condition:
        return _with_reason(
            HarmonicFit(**{**empty.__dict__, "condition": condition}),
            f"design matrix ill-conditioned (cond={condition:.1f} > "
            f"max_condition={max_condition:.0f})",
        )

    beta, *_ = np.linalg.lstsq(design, val_arr, rcond=None)
    offset, cos_coef, sin_coef = (float(b) for b in beta)
    semi = float(np.hypot(cos_coef, sin_coef))
    amplitude = 2.0 * semi

    fitted = design @ beta
    residual = val_arr - fitted
    rmse = float(np.sqrt(np.mean(residual**2)))
    ss_tot = float(np.sum((val_arr - val_arr.mean()) ** 2))
    # A flat series has no variance to explain, so r2 is undefined. The floor is
    # absolute rather than a test against zero because a genuinely constant
    # series still leaves floating-point noise in ss_tot, and dividing one piece
    # of noise by another returns a number that looks like a fit statistic.
    r2 = (
        float(1.0 - np.sum(residual**2) / ss_tot)
        if ss_tot > _FLAT_SERIES_SS_TOT
        else float("nan")
    )

    # model = offset + semi * cos(2*pi*doy/period - phase), maximal at phase.
    phase = float(np.arctan2(sin_coef, cos_coef))
    phase_doy = float((phase / (2.0 * np.pi)) * period % period)

    fit = HarmonicFit(
        offset=offset,
        cos_coef=cos_coef,
        sin_coef=sin_coef,
        amplitude=amplitude,
        phase_doy=phase_doy,
        n_obs=n_obs,
        doy_span=span,
        condition=condition,
        rmse=rmse,
        r2=r2,
    )
    if amplitude > max_amplitude:
        return _with_reason(
            fit,
            f"fitted amplitude {amplitude:.2f} exceeds max_amplitude={max_amplitude}",
        )
    return fit


def _with_reason(fit: HarmonicFit, reason: str) -> HarmonicFit:
    return HarmonicFit(**{**fit.__dict__, "reason": reason})


def theil_sen_slope(
    x: Sequence[float], y: Sequence[float]
) -> tuple[float, float, float]:
    """Return the Theil-Sen slope, and the low and high ends of its interval.

    The median of all pairwise slopes, so one bad year cannot set the sign the
    way it can with least squares. That matters here because a single year with
    an undetected cloud is exactly the failure mode a recovery trend has to
    survive.

    Returns:
        ``(slope, lo, hi)``, all NaN when fewer than three finite pairs or when
        every point shares an x value.
    """
    from scipy.stats import theilslopes

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[finite], y_arr[finite]
    if x_arr.size < 3 or np.unique(x_arr).size < 2:
        return float("nan"), float("nan"), float("nan")
    slope, _intercept, lo, hi = theilslopes(y_arr, x_arr)
    return float(slope), float(lo), float(hi)


# --------------------------------------------------------------------------
# The registered annual metrics
# --------------------------------------------------------------------------
def _in_year(obs: pd.DataFrame, year: int) -> pd.DataFrame:
    return obs.loc[pd.to_numeric(obs["year"], errors="coerce") == int(year)]


def _in_window(obs: pd.DataFrame, doy_min: int, doy_max: int) -> pd.DataFrame:
    doy = pd.to_numeric(obs["doy"], errors="coerce")
    return obs.loc[(doy >= int(doy_min)) & (doy <= int(doy_max))]


@register_annual("ndvi_mean")
def ndvi_mean(
    obs: pd.DataFrame,
    *,
    year: int,
    index: str = "ndvi",
    doy_min: int = 152,
    doy_max: int = 258,
    min_obs: int = 3,
    **_ignored: Any,
) -> AnnualResult:
    """Growing-season mean index level for one stand in one year.

    The window defaults to 1 June through 15 September. At 39 degrees north
    green-up finishes near day 130 and senescence starts near day 270, so this
    sits inside the plateau and a two-week difference in acquisition date does
    not move the value more than the stand does.

    The mean is unweighted across observations. Weighting by pixel count would
    over-weight the fully clear scenes against the partly masked ones, which
    biases the answer toward clear-sky conditions rather than toward the stand.
    """
    window = _in_window(_in_year(obs, year), doy_min, doy_max)
    values = pd.to_numeric(window[index], errors="coerce").dropna()
    support = {
        "ndvi_mean_n_obs": float(len(values)),
        "ndvi_mean_median": float(values.median()) if len(values) else float("nan"),
    }
    if len(values) < min_obs:
        return AnnualResult(
            "ndvi_mean",
            float("nan"),
            support,
            "index",
            f"n_obs={len(values)} in DOY {doy_min}-{doy_max} < min_obs={min_obs}",
        )
    return AnnualResult("ndvi_mean", float(values.mean()), support, "index")


@register_annual("ndvi_seasonal_amplitude")
def ndvi_seasonal_amplitude(
    obs: pd.DataFrame,
    *,
    year: int,
    index: str = "ndvi",
    doy_min: int = 1,
    doy_max: int = 366,
    min_obs: int = 6,
    min_doy_span: float = 150.0,
    max_condition: float = 30.0,
    max_amplitude: float = 1.5,
    **_ignored: Any,
) -> AnnualResult:
    """Peak-to-trough size of the annual cycle, from a single-harmonic fit.

    Uses the whole calendar year rather than the growing season, because the
    winter trough is half the signal.
    """
    window = _in_window(_in_year(obs, year), doy_min, doy_max)
    fit = fit_single_harmonic(
        pd.to_numeric(window["doy"], errors="coerce"),
        pd.to_numeric(window[index], errors="coerce"),
        year=year,
        min_obs=min_obs,
        min_doy_span=min_doy_span,
        max_condition=max_condition,
        max_amplitude=max_amplitude,
    )
    support = {
        "amplitude_n_obs": float(fit.n_obs),
        "amplitude_doy_span": fit.doy_span,
        "amplitude_condition": fit.condition,
        "amplitude_rmse": fit.rmse,
        "amplitude_r2": fit.r2,
        "amplitude_phase_doy": fit.phase_doy,
        "amplitude_offset": fit.offset,
    }
    if fit.reason:
        return AnnualResult(
            "ndvi_seasonal_amplitude",
            float("nan"),
            support,
            "index (peak to trough)",
            fit.reason,
        )
    return AnnualResult(
        "ndvi_seasonal_amplitude", fit.amplitude, support, "index (peak to trough)"
    )


@register_annual("ndvi_trend")
def ndvi_trend(
    obs: pd.DataFrame,
    *,
    year: int,
    index: str = "ndvi",
    window_years: int = 5,
    min_years: int = 4,
    **_ignored: Any,
) -> AnnualResult:
    """Theil-Sen slope of the growing-season mean over a trailing window.

    Reported in index units per year at the last year of the window, so the
    value at 2020 describes 2016 through 2020. Recovery reads positive,
    chronic decline negative.

    The underlying yearly levels come from :func:`ndvi_mean` with its own
    configured parameters, so the trend and the level can never be computed
    over different seasons.
    """
    level_spec = get_annual("ndvi_mean")
    level_params = {
        k: v
        for k, v in level_spec.defaults.items()
        if k in {"doy_min", "doy_max", "min_obs"}
    }
    years = list(range(int(year) - int(window_years) + 1, int(year) + 1))
    levels = [
        (y, level_spec.fn(obs, year=y, index=index, **level_params).value)
        for y in years
    ]
    usable = [(y, v) for y, v in levels if np.isfinite(v)]
    support = {
        "trend_n_years": float(len(usable)),
        "trend_window_years": float(window_years),
    }
    if len(usable) < min_years:
        return AnnualResult(
            "ndvi_trend",
            float("nan"),
            support,
            "index per year",
            f"{len(usable)} usable years in {years[0]}-{years[-1]} < min_years={min_years}",
        )
    slope, lo, hi = theil_sen_slope([y for y, _ in usable], [v for _, v in usable])
    support["trend_slope_lo"] = lo
    support["trend_slope_hi"] = hi
    if not np.isfinite(slope):
        return AnnualResult(
            "ndvi_trend", float("nan"), support, "index per year", "slope undetermined"
        )
    return AnnualResult("ndvi_trend", slope, support, "index per year")


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def stand_year_record(
    obs: pd.DataFrame,
    stand_id: str,
    year: int,
    *,
    area_m2: float,
    n_raw: int,
    metrics: Sequence[str],
    qc_counts: Mapping[str, int] | None = None,
) -> StandYearRecord:
    """Compute every requested metric for one stand in one year.

    Args:
        obs: One stand's quality-controlled observations, **all years**. Metrics
            slice what they need; ``ndvi_trend`` looks back across years.
        stand_id: Stand identifier.
        year: Calendar year.
        area_m2: Stand area.
        n_raw: Observations in this year before quality control.
        metrics: Annual metric names to compute.
        qc_counts: Per-cause drop tallies for this stand, recorded as support.

    Returns:
        A :class:`StandYearRecord`, emitted even when no observation survived.
    """
    this_year = _in_year(obs, year)
    values: dict[str, float] = {}
    support: dict[str, float] = {}
    reasons: dict[str, str] = {}

    for name in metrics:
        spec = get_annual(name)
        result = spec.fn(obs, year=year, index=spec.index, **spec.defaults)
        values[name] = result.value
        support.update(result.support)
        if result.reason:
            reasons[name] = result.reason

    if not this_year.empty:
        coverage = pd.to_numeric(this_year["coverage_fraction"], errors="coerce")
        pixels = pd.to_numeric(this_year["n_pixels"], errors="coerce")
        sensors = sorted(set(this_year["sensor"].astype(str)))
        support["median_coverage_fraction"] = float(coverage.median())
        support["median_n_pixels"] = float(pixels.median())
        support["min_n_pixels"] = float(pixels.min())
        support["n_sensors"] = float(len(sensors))
        support["frac_l7"] = float((this_year["sensor"].astype(str) == "L7").mean())
        support["n_paths"] = float(this_year["wrs_path"].nunique())
    else:
        sensors = []
        for key in (
            "median_coverage_fraction",
            "median_n_pixels",
            "min_n_pixels",
            "n_sensors",
            "frac_l7",
            "n_paths",
        ):
            support[key] = float("nan")

    for cause, n in (qc_counts or {}).items():
        support[f"dropped_{_slug(cause)}"] = float(n)

    return StandYearRecord(
        stand_id=str(stand_id),
        year=int(year),
        area_m2=float(area_m2),
        n_observations=int(len(this_year)),
        n_observations_raw=int(n_raw),
        sensor_mix=",".join(sensors),
        metrics=values,
        support=support,
        reasons=reasons,
    )


def _slug(text: str) -> str:
    keep = [c if c.isalnum() else "_" for c in text.lower()]
    return "".join(keep).strip("_").replace("__", "_")[:40]


def records_to_annual_frame(records: Sequence[StandYearRecord]) -> pd.DataFrame:
    """Stack records into a tidy frame sorted by stand and year."""
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame([record.to_row() for record in records])
    return frame.sort_values(["stand_id", "year"]).reset_index(drop=True)


def annual_table(
    obs: pd.DataFrame,
    *,
    metrics: Sequence[str] | None = None,
    cfg: Any = None,
    years: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Derive per-stand-per-year metrics from an observation table.

    A row is emitted for every stand and every year in range, including years
    where no observation survived quality control. A gap in the record is then
    visible as a row that says why, rather than as a row that is not there,
    which is what lets the table join cleanly against the stand metrics.

    Args:
        obs: Observation frame from the extraction step, any number of stands.
        metrics: Annual metric names. Defaults to every registered metric.
        cfg: Loaded ``satellite.yaml``. Read from config when omitted.
        years: Years to emit. Defaults to the span present in ``obs``.

    Returns:
        A tidy frame, one row per stand per year.
    """
    from csdv_core.config import load_satellite

    cfg = cfg if cfg is not None else load_satellite()
    metrics = list(metrics) if metrics is not None else list_annual_metrics()
    if obs.empty:
        logger.warning("No observations supplied; annual table is empty")
        return pd.DataFrame()

    obs = obs.copy()
    obs["year"] = pd.to_numeric(obs["year"], errors="coerce").astype("int64")
    if years is None:
        years = list(range(int(obs["year"].min()), int(obs["year"].max()) + 1))
    years = [int(y) for y in years]

    # Stand identity and area come from the raw frame, so a stand voided
    # wholesale by the minimum-area gate still gets rows saying so.
    areas = (
        obs.groupby("stand_id")["area_m2"]
        .apply(lambda s: float(pd.to_numeric(s, errors="coerce").median()))
        .to_dict()
    )
    raw_counts = obs.groupby(["stand_id", "year"]).size().to_dict()

    records: list[StandYearRecord] = []
    for stand_id, raw_group in obs.groupby("stand_id", sort=True):
        kept, qc_counts = filter_observations(
            raw_group,
            index=cfg.annual_metrics[metrics[0]].index
            if cfg.annual_metrics
            else "ndvi",
            min_pixels=cfg.qa.min_pixels,
            min_effective_pixels=cfg.qa.min_effective_pixels,
            min_coverage_fraction=cfg.qa.min_coverage_fraction,
            min_area_m2=cfg.qa.min_area_m2,
            valid_range=tuple(cfg.qa.index_valid_range),
        )
        for year in years:
            records.append(
                stand_year_record(
                    kept,
                    str(stand_id),
                    year,
                    area_m2=areas.get(stand_id, float("nan")),
                    n_raw=int(raw_counts.get((stand_id, year), 0)),
                    metrics=metrics,
                    qc_counts=qc_counts,
                )
            )

    frame = records_to_annual_frame(records)
    computed = {
        name: int(frame[name].notna().sum()) for name in metrics if name in frame
    }
    logger.info(
        "Derived %d stand-year records over %d stands and %d years (%s)",
        len(frame),
        obs["stand_id"].nunique(),
        len(years),
        ", ".join(f"{k}: {v}" for k, v in sorted(computed.items())),
    )
    return frame
