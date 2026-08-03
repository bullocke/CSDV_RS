"""Tests for the grid and provenance logic of the Planetary Computer fetch.

The network calls are not exercised here. What is exercised is the part that
determines whether two dates end up comparable: the snapped grid, and the
single date tag a multi-quad mosaic is given.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from csdv_core.download.naip_pc import (
    grid_from_bounds,
    median_date_tag,
    snap_bounds,
)


def test_snap_expands_outward_to_the_snap_multiple():
    bounds = snap_bounds((559427.2, 4323937.2, 564887.0, 4330915.8), snap_m=3.0)
    assert bounds == pytest.approx((559425.0, 4323936.0, 564888.0, 4330917.0))
    # Snapping never shrinks the extent.
    assert bounds[0] <= 559427.2 and bounds[2] >= 564887.0


def test_snap_applies_padding_before_snapping():
    plain = snap_bounds((100.0, 100.0, 200.0, 200.0), snap_m=3.0)
    padded = snap_bounds((100.0, 100.0, 200.0, 200.0), snap_m=3.0, pad_m=150.0)
    assert padded[0] < plain[0] - 140.0
    assert padded[2] > plain[2] + 140.0


def test_snapped_bounds_are_a_whole_number_of_pixels_at_both_resolutions():
    """0.6 m and 1.0 m grids must nest, so a coarse date and a fine date share
    pixel corners and one warp is enough to align them."""
    minx, miny, maxx, maxy = snap_bounds((0.7, 0.3, 1000.2, 800.9), snap_m=3.0)
    for resolution in (0.6, 1.0):
        assert (maxx - minx) / resolution == pytest.approx(
            round((maxx - minx) / resolution)
        )
        assert (maxy - miny) / resolution == pytest.approx(
            round((maxy - miny) / resolution)
        )


def test_snap_rejects_a_non_positive_multiple():
    with pytest.raises(ValueError, match="snap_m must be positive"):
        snap_bounds((0.0, 0.0, 1.0, 1.0), snap_m=0.0)


def test_grid_places_the_origin_at_the_upper_left():
    grid = grid_from_bounds((0.0, 0.0, 60.0, 30.0), 0.6)
    assert grid.width == 100
    assert grid.height == 50
    assert grid.transform.c == pytest.approx(0.0)
    assert grid.transform.f == pytest.approx(30.0)
    assert grid.transform.a == pytest.approx(0.6)
    assert grid.transform.e == pytest.approx(-0.6)


def test_grids_at_the_two_naip_resolutions_share_a_corner():
    bounds = snap_bounds((559427.2, 4323937.2, 564887.0, 4330915.8), snap_m=3.0)
    fine = grid_from_bounds(bounds, 0.6)
    coarse = grid_from_bounds(bounds, 1.0)
    assert fine.transform.c == pytest.approx(coarse.transform.c)
    assert fine.transform.f == pytest.approx(coarse.transform.f)


def test_grid_rejects_empty_bounds():
    with pytest.raises(ValueError, match="empty grid"):
        grid_from_bounds((10.0, 10.0, 10.0, 20.0), 0.6)


def test_median_date_tag_picks_the_middle_acquisition():
    tag = median_date_tag(
        [
            "2018-08-12T00:00:00Z",
            "2018-09-04T00:00:00Z",
            "2018-08-12T00:00:00Z",
        ]
    )
    assert tag == "20180812"


def test_median_date_tag_accepts_datetimes():
    tag = median_date_tag(
        [
            datetime(2022, 6, 19, tzinfo=timezone.utc),
            datetime(2022, 6, 21, tzinfo=timezone.utc),
            datetime(2022, 6, 21, tzinfo=timezone.utc),
        ]
    )
    assert tag == "20220621"


def test_median_date_tag_is_eight_digits_for_the_model_filename():
    """The canopy height model parses day of year from an eight-digit token in
    the filename, so the tag has to be exactly that."""
    tag = median_date_tag(["2016-09-04T00:00:00Z"])
    assert len(tag) == 8
    assert tag.isdigit()


def test_median_date_tag_rejects_an_empty_set():
    with pytest.raises(ValueError, match="No acquisition dates"):
        median_date_tag([])
