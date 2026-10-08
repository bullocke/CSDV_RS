"""Tests for residual canopy versus gap components inside a stand."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from csdv_core.zonal.components import (
    NEW_GAP,
    OUTSIDE,
    PREEXISTING_GAP,
    RESIDUAL,
    WHOLE,
    classify_components,
    component_stats,
    crowns_by_component,
    distance_to_residual,
    gap_patches,
    height_by_distance,
    patch_table,
    variance_decomposition,
)

SIZE = 12


def _mask() -> np.ndarray:
    mask = np.ones((SIZE, SIZE), dtype=bool)
    mask[0, :] = False  # one row outside the stand
    return mask


def _reference() -> np.ndarray:
    """Tall canopy everywhere except a pre-existing opening in the top left."""
    ref = np.full((SIZE, SIZE), 20.0, dtype=np.float32)
    ref[1:4, 0:3] = 0.5
    return ref


def _post() -> np.ndarray:
    """The east half is cut to the ground, the west half survives."""
    post = _reference().copy()
    post[:, 6:] = 0.3
    post[5, 5] = np.nan
    return post


def _expected(with_reference: bool) -> np.ndarray:
    expected = np.full((SIZE, SIZE), NEW_GAP, dtype=np.uint8)
    expected[:, 0:6] = RESIDUAL
    expected[1:4, 0:3] = PREEXISTING_GAP if with_reference else NEW_GAP
    expected[0, :] = OUTSIDE  # outside the stand
    expected[5, 5] = OUTSIDE  # NaN height
    return expected


def test_codes_with_reference():
    codes = classify_components(_post(), _mask(), chm_ref=_reference())
    assert codes.dtype == np.uint8
    np.testing.assert_array_equal(codes, _expected(with_reference=True))


def test_codes_without_reference_have_no_preexisting_gap():
    codes = classify_components(_post(), _mask())
    assert not np.any(codes == PREEXISTING_GAP)
    np.testing.assert_array_equal(codes, _expected(with_reference=False))


def test_component_pixels_partition_the_valid_stand():
    mask = _mask()
    post = _post()
    codes = classify_components(post, mask, chm_ref=_reference())
    valid = mask & np.isfinite(post)
    assert int((codes > OUTSIDE).sum()) == int(valid.sum())


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        classify_components(_post(), _mask()[:5])


def test_component_stats_track_gap_fill():
    codes = classify_components(_post(), _mask(), chm_ref=_reference())
    later = _post().copy()
    later[1:, 6:9] = 5.0  # half the gap columns grow past the threshold
    before = component_stats(_post(), codes, NEW_GAP)
    after = component_stats(later, codes, NEW_GAP)
    assert before["canopy_fraction"] == 0.0
    assert after["canopy_fraction"] == pytest.approx(0.5)
    assert after["height_median"] > before["height_median"]
    assert np.isnan(before["canopy_height_p90"])
    assert after["canopy_height_p90"] == pytest.approx(5.0)


def test_whole_stats_match_pixel_module():
    from csdv_core.zonal.pixel import crown_fraction, height_stats

    mask = _mask()
    post = _post()
    codes = classify_components(post, mask, chm_ref=_reference())
    whole = component_stats(post, codes, WHOLE)
    assert whole["canopy_fraction"] == pytest.approx(crown_fraction(post, mask))
    expected = height_stats(post, mask)
    assert whole["canopy_height_p90"] == pytest.approx(expected["height_p90"])
    assert whole["canopy_height_cv"] == pytest.approx(expected["height_cv"])


def test_component_stats_below_min_pixels_is_nan():
    codes = classify_components(_post(), _mask(), chm_ref=_reference())
    stats = component_stats(_post(), codes, PREEXISTING_GAP, min_pixels=50)
    assert stats["n_pixels"] == 9
    assert np.isnan(stats["height_p90"])


def test_variance_parts_sum_to_total():
    codes = classify_components(_post(), _mask(), chm_ref=_reference())
    parts = variance_decomposition(_post(), codes)
    within = sum(v for k, v in parts.items() if k.startswith("within_"))
    assert within + parts["between"] == pytest.approx(parts["total"], rel=1e-5)
    # Two flat components differ only between themselves.
    assert parts["between"] > 0.99 * parts["total"]


def test_distance_zero_on_residual_and_grows_eastward():
    codes = classify_components(_post(), _mask(), chm_ref=_reference())
    dist = distance_to_residual(codes, pixel_size_m=0.6)
    assert (dist[codes == RESIDUAL] == 0).all()
    assert dist[6, 6] == pytest.approx(0.6)
    assert dist[6, 11] == pytest.approx(6 * 0.6)


def test_distance_without_residual_is_infinite():
    codes = np.full((4, 4), NEW_GAP, dtype=np.uint8)
    assert np.isinf(distance_to_residual(codes, pixel_size_m=1.0)).all()


def test_height_by_distance_bins():
    codes = classify_components(_post(), _mask(), chm_ref=_reference())
    later = _post().copy()
    later[1:, 6:8] = 6.0  # only the two columns nearest the residual grow
    dist = distance_to_residual(codes, pixel_size_m=1.0)
    table = height_by_distance(later, codes, dist, bins_m=[0, 2.5, 10], min_pixels=1)
    assert len(table) == 2
    assert table.loc[0, "canopy_fraction"] == pytest.approx(1.0)
    assert table.loc[1, "canopy_fraction"] == pytest.approx(0.0)


def test_gap_patches_and_table():
    codes = np.zeros((6, 6), dtype=np.uint8)
    codes[0:2, 0:2] = NEW_GAP  # four pixels
    codes[4, 5] = NEW_GAP  # one isolated pixel
    labels = gap_patches(codes)
    assert labels.max() == 2
    kept = gap_patches(codes, min_area_px=2)
    assert kept.max() == 1
    assert kept[4, 5] == 0
    chm = np.zeros((6, 6), dtype=np.float32)
    chm[0, 0] = 3.0
    table = patch_table(kept, {2020: chm}, pixel_size_m=0.5)
    assert len(table) == 1
    assert table.loc[0, "area_m2"] == pytest.approx(4 * 0.25)
    assert table.loc[0, "canopy_fraction"] == pytest.approx(0.25)


def test_crowns_by_component_samples_centroid():
    import geopandas as gpd
    from shapely.geometry import box

    codes = np.zeros((4, 4), dtype=np.uint8)
    codes[:, :2] = RESIDUAL
    codes[:, 2:] = NEW_GAP
    transform = Affine(1.0, 0.0, 100.0, 0.0, -1.0, 50.0)
    crowns = gpd.GeoDataFrame(
        geometry=[
            box(100.2, 49.2, 100.8, 49.8),  # column 0
            box(103.2, 46.2, 103.8, 46.8),  # column 3
            box(110.0, 40.0, 111.0, 41.0),  # beyond the block
        ]
    )
    out = crowns_by_component(crowns, codes, transform)
    assert list(out) == [RESIDUAL, NEW_GAP, OUTSIDE]
