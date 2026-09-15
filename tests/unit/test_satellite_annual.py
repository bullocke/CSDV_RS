"""Tests for csdv_core.satellite.annual.

This is where the science lives, so it carries the largest test file. Every
metric is exercised against synthetic frames with hand-computable answers, and
every guard is exercised for the reason string it emits, because a NaN whose
cause cannot be read is no better than a crash.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from csdv_core.satellite.annual import (
    REASON_COLUMN,
    SUPPORT_PREFIX,
    annual_table,
    filter_observations,
    fit_single_harmonic,
    ndvi_mean,
    ndvi_seasonal_amplitude,
    ndvi_trend,
    stand_year_record,
    theil_sen_slope,
)


def _series(doy, values, *, year=2019, stand_id="S1", sensor="L8", **overrides):
    """A minimal observation frame with everything the filters read."""
    n = len(doy)
    frame = pd.DataFrame(
        {
            "stand_id": [stand_id] * n,
            "sensor": [sensor] * n,
            "year": [year] * n,
            "doy": list(doy),
            "ndvi": list(values),
            "n_pixels": [40] * n,
            "pixel_weight_sum": [36.0] * n,
            "coverage_fraction": [1.0] * n,
            "area_m2": [32400.0] * n,
            "wrs_path": [21] * n,
        }
    )
    for key, value in overrides.items():
        frame[key] = value
    return frame


def _harmonic(doy, *, offset=0.5, cos_coef=0.15, sin_coef=0.20, period=365.0):
    t = 2.0 * np.pi * np.asarray(doy, dtype=float) / period
    return offset + cos_coef * np.cos(t) + sin_coef * np.sin(t)


# --------------------------------------------------------------------------
# The harmonic fit
# --------------------------------------------------------------------------
def test_harmonic_recovers_a_noiseless_series_exactly() -> None:
    doy = np.linspace(10, 360, 24)
    fit = fit_single_harmonic(doy, _harmonic(doy), year=2019)
    assert fit.reason == ""
    assert fit.offset == pytest.approx(0.5, abs=1e-9)
    assert fit.amplitude == pytest.approx(2 * np.hypot(0.15, 0.20), abs=1e-9)
    assert fit.r2 == pytest.approx(1.0, abs=1e-9)
    assert fit.rmse == pytest.approx(0.0, abs=1e-9)
    assert fit.n_obs == 24


def test_phase_doy_lands_on_the_analytic_maximum() -> None:
    doy = np.linspace(1, 365, 40)
    values = _harmonic(doy)
    fit = fit_single_harmonic(doy, values, year=2019)
    fine = np.arange(1.0, 366.0, 0.25)
    expected = fine[int(np.argmax(_harmonic(fine)))]
    assert abs(fit.phase_doy - expected) < 2.0


def test_amplitude_is_reported_peak_to_trough() -> None:
    # Semi-amplitude 0.25 -> the reported number is 0.50.
    doy = np.linspace(5, 360, 30)
    fit = fit_single_harmonic(
        doy, _harmonic(doy, cos_coef=0.25, sin_coef=0.0), year=2019
    )
    assert fit.amplitude == pytest.approx(0.50, abs=1e-9)


def test_amplitude_is_invariant_to_offset_and_linear_in_gain() -> None:
    # Two properties that pin the definition against a future "simplification".
    doy = np.linspace(5, 360, 30)
    base = _harmonic(doy)
    plain = fit_single_harmonic(doy, base, year=2019).amplitude
    shifted = fit_single_harmonic(doy, base + 0.3, year=2019).amplitude
    scaled = fit_single_harmonic(doy, base * 2.0, year=2019).amplitude
    assert shifted == pytest.approx(plain, abs=1e-9)
    assert scaled == pytest.approx(2.0 * plain, abs=1e-9)


def test_flat_series_has_zero_amplitude_and_undefined_r2() -> None:
    doy = np.linspace(5, 360, 12)
    fit = fit_single_harmonic(doy, np.full(12, 0.88), year=2019)
    assert fit.amplitude == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(fit.r2)  # SS_tot is zero, so r2 is undefined, not 0 or 1
    assert fit.reason == ""


def test_too_few_observations_reports_the_count_and_keeps_it_in_support() -> None:
    doy = np.linspace(5, 360, 5)
    fit = fit_single_harmonic(doy, _harmonic(doy), year=2019, min_obs=6)
    assert np.isnan(fit.amplitude)
    assert "min_obs" in fit.reason
    assert fit.n_obs == 5  # support still says how far short the year fell


def test_clustered_observations_report_the_span_not_the_condition_number() -> None:
    # Guard ordering. Eight observations all inside July are ill-conditioned
    # too, but "spans 40 days" is the reason a reader can act on.
    doy = np.linspace(180, 220, 8)
    fit = fit_single_harmonic(doy, _harmonic(doy), year=2019)
    assert np.isnan(fit.amplitude)
    assert "min_doy_span" in fit.reason
    assert "condition" not in fit.reason


def test_two_tight_clusters_trip_the_condition_guard() -> None:
    doy = np.array([1.0, 2.0, 3.0, 4.0, 183.0, 184.0, 185.0, 186.0])
    fit = fit_single_harmonic(doy, _harmonic(doy), year=2019, min_doy_span=150.0)
    assert np.isnan(fit.amplitude)
    assert "ill-conditioned" in fit.reason
    assert np.isfinite(fit.condition)


def test_leap_year_uses_a_366_day_period() -> None:
    doy = np.linspace(1, 366, 40)
    leap = fit_single_harmonic(doy, _harmonic(doy, period=366.0), year=2020)
    assert leap.r2 == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------
# Theil-Sen
# --------------------------------------------------------------------------
def test_theil_sen_recovers_a_known_slope() -> None:
    x = np.arange(2015, 2021, dtype=float)
    slope, lo, hi = theil_sen_slope(x, 0.5 + 0.02 * (x - 2015))
    assert slope == pytest.approx(0.02, abs=1e-9)
    assert lo <= slope <= hi


def test_theil_sen_ignores_a_single_bad_year() -> None:
    # The reason a robust slope is used: one undetected cloud year must not
    # dominate a recovery trend.
    x = np.arange(2015, 2021, dtype=float)
    y = 0.5 + 0.02 * (x - 2015)
    y[3] = 0.05
    assert theil_sen_slope(x, y)[0] == pytest.approx(0.02, abs=1e-9)
    # Least squares survives with the right sign here but is attenuated more
    # than fourfold by the one bad year.
    assert np.polyfit(x, y, 1)[0] < 0.02 / 3.0


def test_theil_sen_needs_three_points() -> None:
    assert np.isnan(theil_sen_slope([2015.0, 2016.0], [0.5, 0.6])[0])


# --------------------------------------------------------------------------
# The registered metrics
# --------------------------------------------------------------------------
def test_ndvi_mean_uses_only_the_growing_season_window() -> None:
    obs = _series([100, 160, 200, 250, 300], [0.30, 0.85, 0.90, 0.86, 0.40])
    result = ndvi_mean(obs, year=2019)
    # DOY 100 and 300 sit outside 152-258 and must not pull the level down.
    assert result.value == pytest.approx(np.mean([0.85, 0.90, 0.86]))
    assert result.support["ndvi_mean_n_obs"] == 3.0
    assert result.reason == ""


def test_ndvi_mean_below_min_obs_is_nan_with_a_reason() -> None:
    obs = _series([160, 200], [0.85, 0.90])
    result = ndvi_mean(obs, year=2019, min_obs=3)
    assert np.isnan(result.value)
    assert "min_obs=3" in result.reason


def test_ndvi_mean_ignores_other_years() -> None:
    obs = pd.concat(
        [
            _series([160, 200, 240], [0.85, 0.90, 0.88], year=2019),
            _series([160, 200, 240], [0.20, 0.25, 0.22], year=2020),
        ],
        ignore_index=True,
    )
    assert ndvi_mean(obs, year=2019).value == pytest.approx(np.mean([0.85, 0.90, 0.88]))


def test_ndvi_seasonal_amplitude_wraps_the_fit_and_carries_its_support() -> None:
    doy = np.linspace(10, 360, 20)
    obs = _series(doy, _harmonic(doy))
    result = ndvi_seasonal_amplitude(obs, year=2019)
    assert result.value == pytest.approx(2 * np.hypot(0.15, 0.20), abs=1e-9)
    assert result.support["amplitude_n_obs"] == 20.0
    assert result.support["amplitude_r2"] == pytest.approx(1.0, abs=1e-9)
    assert np.isfinite(result.support["amplitude_phase_doy"])


def test_ndvi_trend_recovers_a_recovery_slope_across_years() -> None:
    frames = []
    for offset, year in enumerate(range(2015, 2021)):
        level = 0.50 + 0.04 * offset
        frames.append(_series([160, 190, 220, 250], [level] * 4, year=year))
    obs = pd.concat(frames, ignore_index=True)
    result = ndvi_trend(obs, year=2020, window_years=5)
    assert result.value == pytest.approx(0.04, abs=1e-9)
    assert result.support["trend_n_years"] == 5.0


def test_ndvi_trend_needs_enough_usable_years() -> None:
    obs = pd.concat(
        [_series([160, 190, 220], [0.8] * 3, year=y) for y in (2019, 2020)],
        ignore_index=True,
    )
    result = ndvi_trend(obs, year=2020, window_years=5, min_years=4)
    assert np.isnan(result.value)
    assert "min_years=4" in result.reason


# --------------------------------------------------------------------------
# Quality control
# --------------------------------------------------------------------------
def test_filter_observations_names_every_cause_it_drops_for() -> None:
    obs = _series([100, 120, 140, 160, 180, 200], [0.8] * 6)
    obs.loc[0, "ndvi"] = np.nan
    obs.loc[1, "ndvi"] = 4.2
    obs.loc[2, "n_pixels"] = 1
    obs.loc[3, "pixel_weight_sum"] = 0.5
    obs.loc[4, "coverage_fraction"] = 0.30
    kept, counts = filter_observations(obs)
    assert len(kept) == 1
    assert sum(counts.values()) == 5
    assert len(counts) == 5  # one named cause per dropped row


def test_partial_slc_off_coverage_is_dropped_by_the_coverage_gate() -> None:
    obs = _series([100, 130, 160, 190], [0.85] * 4, sensor="L7")
    obs.loc[[1, 2], "coverage_fraction"] = 0.28
    kept, counts = filter_observations(obs)
    assert len(kept) == 2
    assert any("coverage" in cause for cause in counts)


def test_a_stand_smaller_than_four_pixels_is_voided_entirely() -> None:
    obs = _series([100, 130, 160], [0.85] * 3, area_m2=2000.0)
    kept, counts = filter_observations(obs)
    assert kept.empty
    assert any("stand area" in cause for cause in counts)


def test_filter_observations_on_an_empty_frame_is_a_no_op() -> None:
    kept, counts = filter_observations(_series([], []))
    assert kept.empty
    assert counts == {}


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def test_annual_table_emits_a_row_per_stand_year_including_empty_ones() -> None:
    doy = np.linspace(10, 360, 18)
    frames = [
        _series(doy, _harmonic(doy), year=year, stand_id=stand)
        for stand in ("S1", "S2")
        for year in (2018, 2020)  # 2019 is deliberately absent
    ]
    table = annual_table(pd.concat(frames, ignore_index=True))
    assert len(table) == 6  # 2 stands x 3 years, gap included
    assert list(table["year"].unique()) == [2018, 2019, 2020]
    gap = table[(table["stand_id"] == "S1") & (table["year"] == 2019)].iloc[0]
    assert np.isnan(gap["ndvi_seasonal_amplitude"])
    assert "ndvi_seasonal_amplitude" in gap[REASON_COLUMN]
    assert gap[f"{SUPPORT_PREFIX}n_observations"] == 0


def test_annual_table_carries_support_and_the_sensor_mix() -> None:
    doy = np.linspace(10, 360, 18)
    obs = pd.concat(
        [
            _series(doy[:9], _harmonic(doy[:9]), sensor="L7"),
            _series(doy[9:], _harmonic(doy[9:]), sensor="L8"),
        ],
        ignore_index=True,
    )
    row = annual_table(obs).iloc[0]
    assert row[f"{SUPPORT_PREFIX}sensor_mix"] == "L7,L8"
    assert row[f"{SUPPORT_PREFIX}n_sensors"] == 2.0
    assert row[f"{SUPPORT_PREFIX}frac_l7"] == pytest.approx(0.5)
    assert row[f"{SUPPORT_PREFIX}median_coverage_fraction"] == pytest.approx(1.0)


def test_stand_year_record_flattens_without_colliding_with_the_zonal_table() -> None:
    from csdv_core.zonal.record import StandMetricRecord

    doy = np.linspace(10, 360, 18)
    record = stand_year_record(
        _series(doy, _harmonic(doy)),
        "S1",
        2019,
        area_m2=32400.0,
        n_raw=18,
        metrics=["ndvi_mean", "ndvi_seasonal_amplitude"],
    )
    row = record.to_row()
    zonal_row = StandMetricRecord(
        stand_id="S1",
        date="2019-07-04",
        year=2019,
        native_res_m=0.6,
        area_m2=32400.0,
        n_pixels=10,
        bbox_fill_fraction=0.8,
    ).to_row()

    assert REASON_COLUMN in row and "unavailable" in zonal_row
    assert REASON_COLUMN != "unavailable"
    shared = (set(row) & set(zonal_row)) - {"stand_id", "year", "area_m2"}
    assert shared == set(), f"satellite and zonal rows collide on {shared}"
