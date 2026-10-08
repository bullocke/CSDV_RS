"""Tests for csdv_core.change.refine."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from csdv_core.change import refine as rf

T = Affine(0.6, 0, 500000, 0, -0.6, 4300000)


def _forest(n: int = 200, h: float = 25.0) -> np.ndarray:
    return np.full((n, n), h, "float32")


def test_square_clearcut_gives_square_envelope():
    h0 = _forest()
    h1 = h0.copy()
    h1[50:150, 50:150] = 1.0  # 60 m x 60 m = 0.36 ha
    res = rf.refine(h0, h1, T)
    assert res.stats["envelope_ha"] == pytest.approx(0.36, rel=0.05)
    minx, miny, maxx, maxy = res.envelope_geom.bounds
    assert maxx - minx == pytest.approx(60, abs=2)
    assert maxy - miny == pytest.approx(60, abs=2)


def test_misregistration_is_not_loss():
    h0 = np.zeros((60, 60), "float32")
    h0[20:40, 20:40] = 25.0  # one big crown
    h1 = np.roll(h0, 1, axis=1)  # shifted one pixel
    assert not rf.loss_pixels(h0, h1).any()


def test_scattered_gaps_merge_under_closing():
    h0 = _forest()
    h1 = h0.copy()
    for r in range(40, 160, 25):  # 8 x 8 px gaps (4.8 m) 15 m apart
        for c in range(40, 160, 25):
            h1[r : r + 8, c : c + 8] = 0.5
    tight = rf.refine(h0, h1, T, params=rf.RefineParams(close_radius_m=2))
    wide = rf.refine(h0, h1, T, params=rf.RefineParams(close_radius_m=10))
    assert tight.envelope_geom.area == pytest.approx(0, abs=1)  # below min area
    assert wide.envelope_geom.geom_type == "Polygon"
    assert wide.stats["opening_share"] < 0.5


def test_seed_keeps_only_touching_components():
    h0 = _forest()
    h1 = h0.copy()
    h1[10:60, 10:60] = 0.0
    h1[120:180, 120:180] = 0.0
    seed = np.zeros(h0.shape, bool)
    seed[130:140, 130:140] = True
    res = rf.refine(h0, h1, T, seed=seed)
    assert res.envelope[150, 150] and not res.envelope[30, 30]


def test_search_mask_limits_loss():
    h0 = _forest()
    h1 = h0.copy()
    h1[50:150, 50:150] = 0.0
    search = np.zeros(h0.shape, bool)
    search[:, :100] = True
    res = rf.refine(h0, h1, T, search=search)
    assert not res.envelope[:, 100:].any()
    assert res.envelope[100, 70]


def test_small_tree_gap_below_min_opening_is_dropped():
    h0 = _forest(80)
    h1 = h0.copy()
    h1[40:44, 40:44] = 0.0  # 2.4 m gap, 5.8 m2
    res = rf.refine(h0, h1, T)
    assert not res.openings.any()
    assert res.envelope_geom.is_empty


def test_nodata_never_counts_as_loss():
    h0 = _forest(40)
    h1 = np.full_like(h0, np.nan)
    assert not rf.loss_pixels(h0, h1).any()
