"""Tests for spatial pattern metrics inside a stand polygon."""

from __future__ import annotations

import numpy as np
import pytest

from csdv_core.zonal.spatial import (
    interior_edge_mask,
    stand_edge_density,
    stand_linearity,
    stand_row_directionality,
    stand_spatial_metrics,
)


def test_solid_canopy_has_no_interior_edge():
    """The polygon boundary is not a canopy edge, so a stand that is canopy
    everywhere must report zero, not its own perimeter."""
    stand = np.zeros((40, 40), dtype=bool)
    stand[10:30, 10:30] = True
    canopy = stand.copy()
    result = stand_edge_density(canopy, stand, pixel_size_m=1.0)
    assert result.value == pytest.approx(0.0)


def test_a_clearing_inside_the_stand_does_produce_edge():
    stand = np.zeros((40, 40), dtype=bool)
    stand[5:35, 5:35] = True
    canopy = stand.copy()
    canopy[15:25, 15:25] = False  # a gap in the middle
    result = stand_edge_density(canopy, stand, pixel_size_m=1.0)
    assert result.value > 0.0


def test_edge_density_is_normalised_by_stand_area_not_bounding_box():
    """The same canopy pattern in a wider bounding box gives the same density."""
    tight_stand = np.zeros((20, 20), dtype=bool)
    tight_stand[:] = True
    tight_canopy = tight_stand.copy()
    tight_canopy[8:12, :] = False
    tight = stand_edge_density(tight_canopy, tight_stand, pixel_size_m=1.0)

    wide_stand = np.zeros((40, 40), dtype=bool)
    wide_stand[10:30, 10:30] = True
    wide_canopy = wide_stand.copy()
    wide_canopy[18:22, 10:30] = False
    wide = stand_edge_density(wide_canopy, wide_stand, pixel_size_m=1.0)

    assert wide.value == pytest.approx(tight.value, rel=0.15)
    assert wide.support_fraction < tight.support_fraction


def test_edge_density_scales_with_pixel_size():
    stand = np.ones((30, 30), dtype=bool)
    canopy = stand.copy()
    canopy[14:16, :] = False
    coarse = stand_edge_density(canopy, stand, pixel_size_m=1.0)
    fine = stand_edge_density(canopy, stand, pixel_size_m=0.5)
    # Halving the pixel halves the edge length and quarters the area, so the
    # density doubles.
    assert fine.value == pytest.approx(2.0 * coarse.value)


def test_interior_edge_mask_drops_the_boundary():
    stand = np.zeros((20, 20), dtype=bool)
    stand[5:15, 5:15] = True
    edges = interior_edge_mask(stand, stand)
    assert not edges.any()


def test_linearity_is_higher_for_a_striped_gap_pattern():
    stand = np.ones((60, 60), dtype=bool)
    striped = np.zeros((60, 60), dtype=bool)
    striped[:, 20:26] = True  # one straight corridor

    rng = np.random.default_rng(seed=2)
    scattered = rng.random((60, 60)) < 0.10

    linear = stand_linearity(striped, stand)
    blotchy = stand_linearity(scattered, stand)
    assert linear.value > blotchy.value


def test_linearity_reports_a_reason_when_there_are_no_gaps():
    stand = np.ones((30, 30), dtype=bool)
    result = stand_linearity(np.zeros((30, 30), dtype=bool), stand)
    assert np.isnan(result.value)
    assert "no gap edges" in result.reason


def test_row_directionality_is_withheld_on_thin_support():
    image = np.zeros((40, 40), dtype=np.float32)
    stand = np.zeros((40, 40), dtype=bool)
    stand[:, 0:8] = True  # fills 0.2 of the bounding box
    result = stand_row_directionality(image, stand)
    assert np.isnan(result.value)
    assert "min_support" in result.reason
    assert result.support_fraction == pytest.approx(0.2)


def test_row_directionality_is_higher_for_a_row_pattern():
    stand = np.ones((64, 64), dtype=bool)
    rows = np.zeros((64, 64), dtype=np.float32)
    rows[:, ::4] = 20.0  # regular vertical rows

    rng = np.random.default_rng(seed=4)
    noise = rng.uniform(0.0, 20.0, size=(64, 64)).astype(np.float32)

    assert (
        stand_row_directionality(rows, stand).value
        > stand_row_directionality(noise, stand).value
    )


def test_row_directionality_rejects_a_shape_mismatch():
    with pytest.raises(ValueError, match="does not match mask"):
        stand_row_directionality(
            np.zeros((10, 10), dtype=np.float32), np.ones((12, 12), dtype=bool)
        )


def test_stand_spatial_metrics_returns_registry_names_and_reasons():
    chm = np.full((40, 40), 15.0, dtype=np.float32)
    chm[18:22, :] = 0.5
    stand = np.zeros((40, 40), dtype=bool)
    stand[5:35, 5:35] = True
    metrics, support, reasons = stand_spatial_metrics(chm, stand, pixel_size_m=0.6)
    assert set(metrics) == {"edge_density", "linearity_index", "row_directionality"}
    assert np.isfinite(metrics["edge_density"])
    assert support["support_fraction"] == pytest.approx(900 / 1600)
    # Support is 0.5625, just above the directionality threshold.
    assert "row_directionality" not in reasons
