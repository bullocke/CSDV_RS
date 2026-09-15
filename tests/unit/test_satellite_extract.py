"""Tests for csdv_core.satellite.extract.

No network. What is exercised is everything that decides whether the numbers
coming back are trustworthy: the schema normalisation, the chunk arithmetic and
the retry classification. The Earth Engine call itself is covered by the
integration tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from csdv_core.satellite.extract import (
    OBSERVATION_COLUMNS,
    PIXEL_AREA_M2,
    _is_transient,
    _normalize_observations,
    _subdivide,
    _with_retry,
    check_mask_propagation,
    date_chunks,
)


# --------------------------------------------------------------------------
# Schema normalisation
# --------------------------------------------------------------------------
def _raw_page(**overrides) -> pd.DataFrame:
    """A page shaped the way Earth Engine actually returns one."""
    frame = pd.DataFrame(
        {
            "stand_id": ["S1", "S2"],
            "image_id": ["LT05_021033_20050704", "LT05_021033_20050704"],
            "sensor": ["L5", "L5"],
            "time_ms": [1120435200000, 1120435200000],
            "wrs_path": [21, 21],
            "wrs_row": [33, 33],
            "scene_cloud_cover": [4.0, 4.0],
            # The reducer names its outputs after the band, so an ndvi band
            # produces a column called ndvi_mean, which is also the name of an
            # annual metric.
            "ndvi_mean": [0.88, 0.42],
            "ndvi_count": [36, 9],
            "valid_count": [36, 9],
            "valid_sum": [32.4, 8.1],
        }
    )
    for key, value in overrides.items():
        frame[key] = value
    return frame


_AREAS = {"S1": 32400.0, "S2": 8100.0}


def test_index_column_is_renamed_off_the_annual_metric_name() -> None:
    out = _normalize_observations(_raw_page(), "ndvi", _AREAS)
    # ndvi_mean is the per-year metric; the per-scene value must not share it.
    assert "ndvi" in out and "ndvi_mean" not in out
    assert "ndvi_n_pixels" in out
    assert out["ndvi"].tolist() == pytest.approx([0.88, 0.42])


def test_dates_are_derived_from_the_timestamp() -> None:
    out = _normalize_observations(_raw_page(), "ndvi", _AREAS)
    assert out["date"].tolist() == ["2005-07-04", "2005-07-04"]
    assert out["year"].tolist() == [2005, 2005]
    assert out["doy"].tolist() == [185, 185]


def test_coverage_fraction_is_weight_over_area_in_pixels() -> None:
    out = _normalize_observations(_raw_page(), "ndvi", _AREAS)
    assert out["expected_pixels"].tolist() == pytest.approx(
        [32400.0 / PIXEL_AREA_M2, 8100.0 / PIXEL_AREA_M2]
    )
    assert out["coverage_fraction"].tolist() == pytest.approx([0.9, 0.9])


def test_a_fully_masked_page_has_no_index_column_and_is_filled_not_raised() -> None:
    # Earth Engine returns the union of properties actually present, so a page
    # in which every stand was cloud-masked comes back without the index at all.
    # The coverage band inherits the same mask, so its counts go to zero too.
    page = _raw_page(valid_count=0, valid_sum=0.0).drop(
        columns=["ndvi_mean", "ndvi_count"]
    )
    out = _normalize_observations(page, "ndvi", _AREAS)
    assert out["ndvi"].isna().all()
    assert list(out.columns)[: len(OBSERVATION_COLUMNS)] == list(OBSERVATION_COLUMNS)


def test_declared_dtypes_survive_normalisation() -> None:
    out = _normalize_observations(_raw_page(), "ndvi", _AREAS)
    for name, dtype in OBSERVATION_COLUMNS.items():
        assert (
            str(out[name].dtype) == dtype
        ), f"{name} is {out[name].dtype}, expected {dtype}"


def test_mask_propagation_check_catches_an_unmasked_coverage_band() -> None:
    # The first real Earth Engine run had exactly this: a fully cloud-masked
    # scene reporting the stand's full area, which made every quality gate
    # inert. The check is what stops it coming back silently.
    # A correctly masked scene zeroes the valid band along with the index,
    # because the valid band is derived from the index and carries its mask.
    # Zeroing only the index would be the bug, not the control case.
    page = _raw_page()
    page.loc[0, ["ndvi_mean", "ndvi_count", "valid_count", "valid_sum"]] = [
        np.nan,
        0,
        0,
        0.0,
    ]
    good = _normalize_observations(page, "ndvi", _AREAS)
    assert good.loc[0, "n_pixels"] == 0
    assert check_mask_propagation(good, "ndvi") == 0

    broken = good.copy()
    broken.loc[0, "n_pixels"] = 36  # masked scene still claiming a full stand
    assert check_mask_propagation(broken, "ndvi") == 1


def test_area_comes_from_the_geometry_not_the_server() -> None:
    out = _normalize_observations(_raw_page(), "ndvi", _AREAS)
    assert out["area_m2"].tolist() == pytest.approx([32400.0, 8100.0])


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def test_year_chunks_are_one_per_year_end_exclusive() -> None:
    chunks = date_chunks(1985, 1987)
    assert chunks == [
        ("1985", "1985-01-01", "1986-01-01"),
        ("1986", "1986-01-01", "1987-01-01"),
        ("1987", "1987-01-01", "1988-01-01"),
    ]


def test_finer_granularities() -> None:
    assert len(date_chunks(1985, 1985, granularity="half_year")) == 2
    months = date_chunks(1985, 1985, granularity="month")
    assert len(months) == 12
    assert months[-1] == ("1985-12", "1985-12-01", "1986-01-01")


def test_chunk_ladder_bottoms_out_at_month() -> None:
    assert _subdivide("year") == "half_year"
    assert _subdivide("half_year") == "month"
    assert _subdivide("month") is None


def test_date_chunks_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="precedes"):
        date_chunks(2000, 1999)
    with pytest.raises(ValueError, match="Unknown granularity"):
        date_chunks(2000, 2001, granularity="fortnight")


# --------------------------------------------------------------------------
# Retry classification
# --------------------------------------------------------------------------
def test_transient_and_permanent_failures_are_told_apart() -> None:
    assert _is_transient("Computation timed out.")
    assert _is_transient("User memory limit exceeded.")
    assert _is_transient("Too many concurrent aggregations")
    assert _is_transient("HttpError 503 Service Unavailable")
    # Retrying these only burns four deadlines each.
    assert not _is_transient("Image.load: asset not found")
    assert not _is_transient("Caller does not have permission")
    assert not _is_transient("not signed up for Earth Engine")


def test_retry_backs_off_and_eventually_succeeds() -> None:
    attempts = {"n": 0}
    delays: list[float] = []

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("Computation timed out.")
        return "ok"

    result = _with_retry(flaky, label="1995", sleep=delays.append)
    assert result == "ok"
    assert attempts["n"] == 3
    assert len(delays) == 2
    assert delays[1] > delays[0]  # backoff grows


def test_retry_does_not_retry_a_permanent_failure() -> None:
    attempts = {"n": 0}

    def broken() -> None:
        attempts["n"] += 1
        raise RuntimeError("not signed up for Earth Engine")

    with pytest.raises(RuntimeError, match="not signed up"):
        _with_retry(broken, label="1995", sleep=lambda _s: None)
    assert attempts["n"] == 1


def test_retry_gives_up_after_max_attempts_and_raises_the_last_error() -> None:
    def always_slow() -> None:
        raise RuntimeError("Computation timed out.")

    with pytest.raises(RuntimeError, match="timed out"):
        _with_retry(always_slow, label="1995", max_attempts=2, sleep=lambda _s: None)
