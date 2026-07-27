"""Tests for mask-aware GLCM texture inside a stand.

Two properties matter and are tested here. On a fully valid rectangle the
implementation must reproduce the windowed metric exactly, which is what makes
it a faithful replacement. And its result must not change when the invalid
padding around a stand changes, which is what the windowed metric gets wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import from_origin
from skimage.feature import graycomatrix

from csdv_core.io.grids import GridSpec
from csdv_core.metrics.texture import glcm_texture as windowed_glcm_texture
from csdv_core.zonal.texture import (
    DEFAULT_ANGLES,
    masked_glcm,
    quantize_masked,
    texture_entropy,
)

LEVELS = 16


@pytest.fixture()
def patchy() -> np.ndarray:
    """A 40 x 40 image with structure, so entropy is well away from its bounds."""
    rng = np.random.default_rng(seed=3)
    base = rng.uniform(0.0, 200.0, size=(40, 40)).astype(np.float32)
    base[10:20, 10:20] += 400.0  # a bright block, so tone is not uniform noise
    return base


def test_matches_skimage_on_a_fully_valid_rectangle(patchy):
    valid = np.ones(patchy.shape, dtype=bool)
    quant, _, _ = quantize_masked(patchy, valid, levels=LEVELS)
    mine = masked_glcm(quant, valid, levels=LEVELS, angles=DEFAULT_ANGLES)
    theirs = graycomatrix(
        quant,
        distances=[1],
        angles=list(DEFAULT_ANGLES),
        levels=LEVELS,
        symmetric=True,
        normed=True,
    )
    assert mine.shape == theirs.shape
    np.testing.assert_allclose(mine, theirs, atol=1e-12)


def test_entropy_matches_the_windowed_metric_on_a_full_tile(patchy):
    """The whole 40 x 40 image is one 40 m window at 1 m pixels."""
    grid = GridSpec(
        transform=from_origin(0.0, 40.0, 1.0, 1.0), crs="EPSG:26916", pixel_size_m=1.0
    )
    windowed = windowed_glcm_texture(patchy, grid, window_m=40.0, levels=LEVELS)
    mine = texture_entropy(patchy, np.ones(patchy.shape, dtype=bool), levels=LEVELS)
    assert mine.reason == ""
    assert mine.entropy_bits == pytest.approx(float(windowed.array[0, 0]), abs=1e-5)


def test_entropy_is_invariant_to_the_surrounding_padding(patchy):
    """Widening the bounding box around an unchanged stand must not move the
    number. This is the property the windowed metric does not have."""
    stand = patchy[8:32, 8:32]
    tight_mask = np.ones(stand.shape, dtype=bool)
    tight = texture_entropy(stand, tight_mask, levels=LEVELS)

    padded = np.full((40, 40), np.nan, dtype=np.float32)
    padded[8:32, 8:32] = stand
    padded_mask = np.zeros((40, 40), dtype=bool)
    padded_mask[8:32, 8:32] = True
    wide = texture_entropy(padded, padded_mask, levels=LEVELS)

    assert wide.entropy_bits == pytest.approx(tight.entropy_bits, abs=1e-9)
    assert wide.support_fraction < tight.support_fraction


def test_windowed_metric_changes_with_padding(patchy):
    """Documents the defect the mask-aware version fixes.

    Setting out-of-stand pixels to grey level 0 puts them all in one bin, so the
    co-occurrence matrix gains a spike whose size depends on the fill fraction.
    If csdv_core.metrics.texture is ever made mask-aware, this test should be
    removed along with the workaround.
    """
    grid_24 = GridSpec(
        transform=from_origin(0.0, 24.0, 1.0, 1.0), crs="EPSG:26916", pixel_size_m=1.0
    )
    grid_32 = GridSpec(
        transform=from_origin(0.0, 32.0, 1.0, 1.0), crs="EPSG:26916", pixel_size_m=1.0
    )
    stand = patchy[8:32, 8:32]
    tight = windowed_glcm_texture(stand, grid_24, window_m=24.0, levels=LEVELS)

    # Same stand, 32 x 32 box, so the fill fraction drops to 0.5625.
    padded = np.full((32, 32), np.nan, dtype=np.float32)
    padded[4:28, 4:28] = stand
    loose = windowed_glcm_texture(padded, grid_32, window_m=32.0, levels=LEVELS)

    assert abs(float(loose.array[0, 0]) - float(tight.array[0, 0])) > 0.5


def test_pairs_crossing_the_boundary_are_excluded():
    """A constant stand surrounded by different constant values gives zero
    entropy, because no pair mixes the two tones."""
    image = np.full((30, 30), 100.0, dtype=np.float32)
    image[5:25, 5:25] = 10.0
    mask = np.zeros((30, 30), dtype=bool)
    mask[5:25, 5:25] = True
    result = texture_entropy(image, mask, levels=LEVELS)
    # Only one grey level survives inside the stand, so there is no variation.
    assert result.reason == "no tonal variation inside the stand"
    assert np.isnan(result.entropy_bits)


def test_entropy_of_a_two_tone_checkerboard_is_one_bit():
    board = np.indices((40, 40)).sum(axis=0) % 2
    image = board.astype(np.float32) * 100.0
    mask = np.ones(image.shape, dtype=bool)
    result = texture_entropy(image, mask, levels=2)
    # Horizontal and vertical neighbours always differ; the diagonals never do.
    # Averaged over the four angles that is (1 + 0 + 1 + 0) / 2 bits... the two
    # diagonal planes each hold two equally likely same-tone pairs, so 1 bit.
    assert 0.9 <= result.entropy_bits <= 1.1


def test_small_stand_reports_a_reason_not_a_number():
    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    mask = np.ones((10, 10), dtype=bool)
    result = texture_entropy(image, mask, min_valid_pixels=256)
    assert np.isnan(result.entropy_bits)
    assert "min_valid_pixels" in result.reason
    assert result.n_valid == 100


def test_non_finite_pixels_inside_the_stand_are_excluded():
    rng = np.random.default_rng(seed=5)
    image = rng.uniform(0, 100, size=(40, 40)).astype(np.float32)
    mask = np.ones(image.shape, dtype=bool)
    holed = image.copy()
    holed[0:4, 0:4] = np.nan
    result = texture_entropy(holed, mask, levels=LEVELS)
    assert result.n_valid == 40 * 40 - 16
    assert np.isfinite(result.entropy_bits)


def test_support_fraction_reports_the_bounding_box_fill():
    image = np.zeros((20, 20), dtype=np.float32)
    mask = np.zeros((20, 20), dtype=bool)
    mask[:10, :] = True
    result = texture_entropy(image, mask, levels=LEVELS)
    assert result.support_fraction == pytest.approx(0.5)


def test_explicit_stretch_bounds_are_used_and_reported(patchy):
    mask = np.ones(patchy.shape, dtype=bool)
    result = texture_entropy(patchy, mask, levels=LEVELS, vmin=0.0, vmax=1000.0)
    assert result.vmin == pytest.approx(0.0)
    assert result.vmax == pytest.approx(1000.0)
    # A wider stretch packs the data into fewer grey levels, lowering entropy.
    default = texture_entropy(patchy, mask, levels=LEVELS)
    assert result.entropy_bits < default.entropy_bits


def test_entropy_cannot_exceed_its_theoretical_maximum(patchy):
    mask = np.ones(patchy.shape, dtype=bool)
    result = texture_entropy(patchy, mask, levels=LEVELS)
    assert 0.0 <= result.entropy_bits <= 2 * np.log2(LEVELS)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="does not match mask"):
        texture_entropy(np.zeros((4, 4), dtype=np.float32), np.ones((5, 5), dtype=bool))
