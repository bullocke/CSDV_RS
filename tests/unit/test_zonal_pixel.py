"""Tests for canopy height metrics computed inside a stand polygon."""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box

from csdv_core.io.grids import GridSpec
from csdv_core.metrics.gap import gap_fraction as windowed_gap_fraction
from csdv_core.zonal.mask import stand_window
from csdv_core.zonal.pixel import (
    crown_fraction,
    gap_fraction,
    gap_persistence,
    height_band_fraction,
    height_stats,
    n_valid,
    shrub_fraction,
    small_tree_fraction,
)

TRANSFORM = from_origin(west=0.0, north=30.0, xsize=1.0, ysize=1.0)
SHAPE = (30, 30)


@pytest.fixture()
def quarter_gap() -> tuple[np.ndarray, np.ndarray]:
    """A 10 x 10 m stand where exactly 25 of its 100 pixels are gap."""
    chm = np.full(SHAPE, 15.0, dtype=np.float32)
    # Rows 15..19, cols 5..9 in raster space is the top-left quarter of a
    # stand covering x 5..15, y 5..15.
    chm[15:20, 5:10] = 0.5
    sw = stand_window(box(5.0, 5.0, 15.0, 15.0), TRANSFORM, SHAPE)
    return (
        chm[
            sw.window.row_off : sw.window.row_off + sw.window.height,
            sw.window.col_off : sw.window.col_off + sw.window.width,
        ],
        sw.mask,
    )


def test_gap_fraction_is_exact(quarter_gap):
    chm, mask = quarter_gap
    assert gap_fraction(chm, mask) == pytest.approx(0.25)


def test_crown_fraction_is_the_complement(quarter_gap):
    chm, mask = quarter_gap
    assert crown_fraction(chm, mask) == pytest.approx(1.0 - gap_fraction(chm, mask))


def test_pixels_outside_the_stand_do_not_change_the_result():
    """Changing a pixel just outside the boundary leaves every metric alone."""
    chm = np.full(SHAPE, 15.0, dtype=np.float32)
    sw = stand_window(box(10.0, 10.0, 20.0, 20.0), TRANSFORM, SHAPE, pad_px=3)
    block = chm[
        sw.window.row_off : sw.window.row_off + sw.window.height,
        sw.window.col_off : sw.window.col_off + sw.window.width,
    ].copy()
    before = gap_fraction(block, sw.mask)
    outside = np.flatnonzero(~sw.mask.ravel())
    block.ravel()[outside] = 0.0
    assert gap_fraction(block, sw.mask) == pytest.approx(before)


def test_nan_pixels_leave_both_numerator_and_denominator():
    chm = np.full(SHAPE, 15.0, dtype=np.float32)
    chm[15:20, 5:10] = np.nan
    sw = stand_window(box(5.0, 5.0, 15.0, 15.0), TRANSFORM, SHAPE)
    block = chm[
        sw.window.row_off : sw.window.row_off + sw.window.height,
        sw.window.col_off : sw.window.col_off + sw.window.width,
    ]
    assert n_valid(block, sw.mask) == 75
    assert gap_fraction(block, sw.mask) == pytest.approx(0.0)


def test_all_nan_stand_returns_nan():
    chm = np.full((10, 10), np.nan, dtype=np.float32)
    mask = np.ones((10, 10), dtype=bool)
    assert np.isnan(gap_fraction(chm, mask))
    assert np.isnan(crown_fraction(chm, mask))
    assert np.isnan(shrub_fraction(chm, mask))


def test_height_bands_are_half_open_and_partition_the_range():
    chm = np.array(
        [[0.2, 0.5, 1.9, 2.0, 9.9, 10.0, 19.9, 20.0, 50.0]], dtype=np.float32
    )
    mask = np.ones_like(chm, dtype=bool)
    assert shrub_fraction(chm, mask) == pytest.approx(2 / 9)  # 0.5 and 1.9
    assert small_tree_fraction(chm, mask) == pytest.approx(2 / 9)  # 2.0 and 9.9
    assert height_band_fraction(chm, mask, lo_m=10.0, hi_m=20.0) == pytest.approx(2 / 9)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="does not match mask"):
        gap_fraction(np.zeros((4, 4), dtype=np.float32), np.ones((5, 5), dtype=bool))


def test_height_stats_use_canopy_pixels_only():
    chm = np.concatenate(
        [np.full(50, 0.5, dtype=np.float32), np.full(50, 20.0, dtype=np.float32)]
    ).reshape(10, 10)
    mask = np.ones((10, 10), dtype=bool)
    stats = height_stats(chm, mask)
    assert stats["height_mean"] == pytest.approx(20.0)
    assert stats["height_cv"] == pytest.approx(0.0)


def test_height_stats_need_enough_canopy_pixels():
    chm = np.full((10, 10), 0.5, dtype=np.float32)
    chm[0, 0:3] = 25.0  # only three canopy pixels
    mask = np.ones((10, 10), dtype=bool)
    stats = height_stats(chm, mask)
    assert all(np.isnan(v) for v in stats.values())


def test_gap_persistence_counts_pixels_that_are_gap_at_both_dates():
    a = np.full((10, 10), 15.0, dtype=np.float32)
    b = np.full((10, 10), 15.0, dtype=np.float32)
    a[0:5, :] = 0.5  # 50 pixels gap at date a
    b[0:2, :] = 0.5  # 20 pixels gap at date b, all inside a's gap
    mask = np.ones((10, 10), dtype=bool)
    assert gap_persistence(a, b, mask) == pytest.approx(0.20)


def test_gap_persistence_denominator_excludes_pixels_invalid_at_either_date():
    a = np.full((10, 10), 0.5, dtype=np.float32)
    b = np.full((10, 10), 0.5, dtype=np.float32)
    b[0, :] = np.nan  # 10 pixels invalid at date b
    mask = np.ones((10, 10), dtype=bool)
    # Every pixel valid at both dates is gap at both dates.
    assert gap_persistence(a, b, mask) == pytest.approx(1.0)


def test_gap_persistence_rejects_mismatched_grids():
    mask = np.ones((10, 10), dtype=bool)
    with pytest.raises(ValueError, match="shape mismatch"):
        gap_persistence(
            np.zeros((10, 10), dtype=np.float32),
            np.zeros((10, 12), dtype=np.float32),
            mask,
        )


def test_gap_persistence_is_nan_without_a_commonly_valid_pixel():
    a = np.full((4, 4), np.nan, dtype=np.float32)
    b = np.full((4, 4), 1.0, dtype=np.float32)
    assert np.isnan(gap_persistence(a, b, np.ones((4, 4), dtype=bool)))


def test_matches_the_windowed_implementation_on_a_full_tile():
    """A rectangular stand covering exactly one window must agree with the
    windowed metric, which is what makes the reimplementation trustworthy."""
    rng = np.random.default_rng(seed=7)
    chm = rng.uniform(0.0, 25.0, size=SHAPE).astype(np.float32)
    grid = GridSpec(transform=TRANSFORM, crs="EPSG:26916", pixel_size_m=1.0)
    windowed = windowed_gap_fraction(chm, grid, window_m=10.0, height_threshold_m=2.0)

    # Window (0, 0) covers x 0..10, y 20..30.
    sw = stand_window(box(0.0, 20.0, 10.0, 30.0), TRANSFORM, SHAPE)
    block = chm[
        sw.window.row_off : sw.window.row_off + sw.window.height,
        sw.window.col_off : sw.window.col_off + sw.window.width,
    ]
    assert sw.n_inside == 100
    assert gap_fraction(block, sw.mask) == pytest.approx(
        float(windowed.array[0, 0]), abs=1e-6
    )


def test_stand_value_lies_within_the_spread_of_its_windows():
    """Section 2 proposes comparing a stand value against the distribution of
    window values inside it as a uniformity diagnostic. The stand value must
    fall inside that range."""
    rng = np.random.default_rng(seed=11)
    chm = rng.uniform(0.0, 25.0, size=SHAPE).astype(np.float32)
    grid = GridSpec(transform=TRANSFORM, crs="EPSG:26916", pixel_size_m=1.0)
    windowed = windowed_gap_fraction(chm, grid, window_m=10.0, height_threshold_m=2.0)

    sw = stand_window(box(0.0, 0.0, 30.0, 30.0), TRANSFORM, SHAPE)
    whole = gap_fraction(chm, sw.mask)
    cells = windowed.array[np.isfinite(windowed.array)]
    assert cells.size == 9
    assert cells.min() <= whole <= cells.max()
    assert whole == pytest.approx(float(cells.mean()), abs=1e-6)
