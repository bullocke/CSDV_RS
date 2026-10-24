"""Tests for the per-crown flood bound, the power window and parameter keys."""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import from_origin
from scipy import ndimage as ndi

from csdv_core.segmentation.chm_watershed import locate_seeds, segment_crowns
from csdv_core.segmentation.flood import (
    crown_floors,
    grow_crowns,
    grow_crowns_reference,
)
from csdv_core.segmentation.params import (
    OPTIMIZED_POST_TRIM_PARAMS,
    TALLO_BIOMES,
    WINDOW_FUNCTIONS,
    SegmentationParams,
    WindowFunction,
    tallo_params,
    tallo_window,
)

PX = 0.6
TRANSFORM = from_origin(500_000, 4_000_000, PX, PX)
CRS = "EPSG:32617"


def _dome(shape, centre, height, radius_px):
    rr, cc = np.indices(shape, dtype="float64")
    d2 = (rr - centre[0]) ** 2 + (cc - centre[1]) ** 2
    return height * np.exp(-d2 / (2.0 * (radius_px / 2.0) ** 2))


def _random_canopy(seed: int, shape=(160, 160)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = ndi.gaussian_filter(rng.random(shape), 3) * 70.0
    h -= np.nanmin(h) - 1.0
    h[h < 8.0] = np.nan
    return h


def test_shorter_neighbour_claims_the_flank_post_trim_drops():
    shape = (80, 120)
    h = np.maximum(_dome(shape, (40, 45), 25.0, 30), _dome(shape, (40, 75), 16.0, 30))
    base = dict(window=WINDOW_FUNCTIONS["fixed_5m"], smooth_radius_m=0.0, th_cr=0.7)
    old = segment_crowns(
        h, TRANSFORM, CRS, params=SegmentationParams(bound_mode="post_trim", **base)
    )
    new = segment_crowns(h, TRANSFORM, CRS, params=SegmentationParams(**base))
    assert len(old) == len(new) == 2
    short_old = old.loc[old["apex_h_m"].idxmin(), "area_m2"]
    short_new = new.loc[new["apex_h_m"].idxmin(), "area_m2"]
    assert short_new > 1.2 * short_old
    assert new["area_m2"].sum() > old["area_m2"].sum()


def test_every_pixel_passes_its_crown_floor_and_touches_its_seed():
    h = _random_canopy(3)
    valid = ~np.isnan(h)
    rows, cols = locate_seeds(h, valid, PX, WINDOW_FUNCTIONS["fixed_5m"])
    apex = h[rows, cols]
    labels, _ = grow_crowns(h, rows, cols, apex, th_cr=0.6, max_radius_px=20)
    floor = np.concatenate([[np.inf], 0.6 * apex])
    lab = labels > 0
    assert np.all(h[lab] >= floor[labels[lab]])
    for i in range(1, rows.size + 1):
        comp, n = ndi.label(labels == i)
        assert n == 1
        assert comp[rows[i - 1], cols[i - 1]] == 1


def test_isolated_dome_keeps_exactly_the_pixels_above_its_floor():
    h = _dome((61, 61), (30, 30), 20.0, 24)
    h[h < 2.0] = np.nan
    labels, _ = grow_crowns(
        h, np.array([30]), np.array([30]), np.array([20.0]), th_cr=0.55
    )
    np.testing.assert_array_equal(labels > 0, np.nan_to_num(h) >= 0.55 * 20.0)


def test_radius_cap_is_a_circle_and_is_reported():
    h = 10.0 - 1e-3 * _dome((61, 61), (30, 30), -1.0, 200)  # near-flat, peak at centre
    labels, capped = grow_crowns(
        h,
        np.array([30]),
        np.array([30]),
        np.array([h[30, 30]]),
        th_cr=0.0,
        max_radius_px=10,
    )
    rr, cc = np.indices(h.shape)
    np.testing.assert_array_equal(labels > 0, (rr - 30) ** 2 + (cc - 30) ** 2 <= 100)
    assert capped[0] == pytest.approx(1.0)


def test_min_depth_lowers_the_floor_of_a_short_crown_only():
    floors = crown_floors(np.array([6.0, 30.0]), th_cr=0.7, min_crown_depth_m=3.0)
    np.testing.assert_allclose(floors, [3.0, 21.0])
    floors = crown_floors(np.array([6.0, 30.0]), th_cr=0.5, max_crown_depth_m=10.0)
    np.testing.assert_allclose(floors, [3.0, 20.0])
    floors = crown_floors(np.array([10.0]), th_cr=0.3, th_seed=0.45)
    np.testing.assert_allclose(floors, [4.5])


@pytest.mark.parametrize("rule", ["apex", "mean"])
def test_compiled_kernel_matches_reference(rule):
    h = _random_canopy(7)
    valid = ~np.isnan(h)
    rows, cols = locate_seeds(h, valid, PX, WINDOW_FUNCTIONS["fixed_5m"])
    kw = dict(th_cr=0.6, th_seed=0.45, max_radius_px=15, rule=rule)
    a, fa = grow_crowns(h, rows, cols, h[rows, cols], **kw)
    b, fb = grow_crowns_reference(h, rows, cols, h[rows, cols], **kw)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(fa, fb)
    again, _ = grow_crowns(h, rows, cols, h[rows, cols], **kw)
    np.testing.assert_array_equal(a, again)


def test_mean_rule_grows_further_down_the_flank_than_the_apex_rule():
    h = _dome((61, 61), (30, 30), 20.0, 24)
    h[h < 2.0] = np.nan
    seed = (np.array([30]), np.array([30]), np.array([20.0]))
    apex_rule, _ = grow_crowns(h, *seed, th_cr=0.7)
    mean_rule, _ = grow_crowns(h, *seed, th_cr=0.7, th_seed=0.45, rule="mean")
    assert (mean_rule > 0).sum() > (apex_rule > 0).sum()


def _brute_force_tops(h, valid, window, px, metric):
    """Every pixel with no higher pixel in its own window, by definition."""
    exact = window(np.where(valid, h, 0.0)) / 2.0 / px
    if metric == "pixel":
        radius = np.clip(np.rint(exact), max(1, int(round(1.0 / px))), None)
    else:
        radius = np.clip(exact, max(1.0, 1.0 / px), None)
    out = []
    nr, nc = h.shape
    hh = np.where(valid, h, -np.inf)
    for r, c in zip(*np.nonzero(valid), strict=True):
        rad = float(radius[r, c])
        reach = int(np.floor(rad + 1e-9))
        r0, r1 = max(0, r - reach), min(nr, r + reach + 1)
        c0, c1 = max(0, c - reach), min(nc, c + reach + 1)
        yy, xx = np.ogrid[r0 - r : r1 - r, c0 - c : c1 - c]
        disk = yy * yy + xx * xx <= rad * rad + 1e-9
        if hh[r, c] >= hh[r0:r1, c0:c1][disk].max():
            out.append((r, c))
    return out


@pytest.mark.parametrize("metric", ["exact", "pixel"])
@pytest.mark.parametrize("px", [0.6, 1.0])
def test_seed_candidates_match_the_definition(metric, px):
    h = np.round(_random_canopy(11, shape=(90, 90)), 1)  # rounding makes plateaus
    valid = ~np.isnan(h)
    window = tallo_window("Temperate Broadleaf & Mixed Forests", lo=3.0, hi=12.0)
    tops = set(_brute_force_tops(h, valid, window, px, metric))
    rows, cols = locate_seeds(h, valid, px, window, window_metric=metric)
    # Plateaus collapse to one seed, which must be one of the tied tops.
    assert all((r, c) in tops for r, c in zip(rows, cols, strict=True))
    mask = np.zeros(h.shape, dtype=bool)
    mask[tuple(np.array(sorted(tops)).T)] = True
    _, n = ndi.label(mask)
    assert rows.size == n


def test_exact_window_separates_what_rounding_merged():
    # At 1 m np.rint takes a 3 m window's radius (1.5 px) up to 2 and a 5 m
    # window's (2.5 px) down to 2, so both become the same disk. The exact
    # metric keeps them apart.
    h = np.zeros((9, 9))
    h[4, 2], h[4, 4] = 10.0, 9.0  # 2 m apart: two tops under a 3 m window only
    valid = h > 0
    w3 = WindowFunction(a=3.0, lo=3.0, hi=3.0)
    w5 = WindowFunction(a=5.0, lo=5.0, hi=5.0)
    n = {
        (m, name): locate_seeds(h, valid, 1.0, w, window_metric=m)[0].size
        for m in ("exact", "pixel")
        for name, w in (("3m", w3), ("5m", w5))
    }
    assert n[("pixel", "3m")] == n[("pixel", "5m")] == 1
    assert n[("exact", "3m")] == 2 and n[("exact", "5m")] == 1


def test_power_window_and_neutral_keys():
    w = WindowFunction(a=0.9618, b=0.6295, k=0.5, lo=3.0, hi=25.0, form="power")
    np.testing.assert_allclose(
        w([0.0, 10.0, 30.0]), [3.0, 3.0, 0.5 * 0.9618 * 30**0.6295]
    )
    assert "form" not in WINDOW_FUNCTIONS["fixed_5m"].as_dict()
    assert w.as_dict()["form"] == "power" and w.as_dict()["k"] == 0.5
    # The study's chosen set keeps the key recorded before the new fields existed.
    assert OPTIMIZED_POST_TRIM_PARAMS.key == "68368c948c"
    assert "bound_mode" not in OPTIMIZED_POST_TRIM_PARAMS.as_dict()
    assert "window_metric" not in OPTIMIZED_POST_TRIM_PARAMS.as_dict()
    assert SegmentationParams().as_dict()["bound_mode"] == "flood"
    with pytest.raises(ValueError):
        SegmentationParams(bound_mode="post_trim", th_seed=0.45)


def test_tallo_presets():
    assert len(TALLO_BIOMES) >= 5
    p = tallo_params("Temperate Conifer Forests")
    assert (p.min_height_m, p.th_cr, p.th_seed, p.max_crown_radius_m) == (
        1.0,
        0.55,
        0.45,
        15.0,
    )
    assert p.window.form == "power" and p.window.k == 0.5
    assert p.window(40.0) == pytest.approx(
        0.5 * 0.9053028822 * 40**0.5453393609, rel=1e-6
    )
    with pytest.raises(KeyError):
        tallo_window("Atlantis")
