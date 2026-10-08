"""Tests for csdv_core.change.di."""

from __future__ import annotations

import numpy as np
import pytest

from csdv_core.change import di


def _field(n: int = 200, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(10.0, 2.0, (n, n)).astype("float32")


def test_scene_z_is_standard_normal_over_forest():
    v = _field()
    pop = np.ones(v.shape, bool)
    z = di.scene_z(v, pop)
    assert np.median(z) == pytest.approx(0.0, abs=0.02)
    assert np.std(z) == pytest.approx(1.0, abs=0.03)


def test_scene_z_robust_to_disturbance():
    v = _field()
    v[:40, :40] = -20.0  # 4% of the image is a deep loss
    pop = np.ones(v.shape, bool)
    z = di.scene_z(v, pop)
    z_mean = di.scene_z(v, pop, robust=False)
    assert np.median(z[60:, 60:]) == pytest.approx(0.0, abs=0.1)
    # Mean and SD are pulled by the loss, which also shrinks its own z.
    assert np.median(z_mean[60:, 60:]) > 0.15
    assert z_mean[:40, :40].max() > -6
    assert z[:40, :40].max() < -10


def test_local_z_removes_gradient():
    v = _field()
    v += np.linspace(0, 20, v.shape[1])[None, :]  # strong east-west stripe
    pop = np.ones(v.shape, bool)
    zs, zl = di.scene_z(v, pop), di.local_z(v, pop, 31)
    assert abs(np.median(zs[:, :20])) > 1.0
    assert abs(np.median(zl[:, 20:-20])) < 0.1
    assert np.std(zl[20:-20, 20:-20]) == pytest.approx(1.0, abs=0.1)


def test_local_z_small_population_falls_back_to_scene():
    v = _field()
    pop = np.zeros(v.shape, bool)
    pop[:, :100] = True
    zl = di.local_z(v, pop, 21)
    assert np.isfinite(zl).all()


def test_pixmean_removes_fixed_pattern():
    rng = np.random.default_rng(1)
    pattern = rng.normal(0, 5, (100, 100))
    hist = di.PixelHistory(pattern.shape)
    pop = np.ones(pattern.shape, bool)
    for _ in range(3):
        hist.add((pattern + rng.normal(0, 0.5, pattern.shape)).astype("float32"), pop)
    x = (pattern + rng.normal(0, 0.5, pattern.shape)).astype("float32")
    x[10:20, 10:20] -= 5
    z = di.standardize(x, pop, "pixmean", hist=hist)
    assert np.median(z[10:20, 10:20]) < -5
    assert np.std(z[30:, 30:]) == pytest.approx(1.0, abs=0.1)


def test_pixz_needs_history():
    hist = di.PixelHistory((50, 50))
    pop = np.ones((50, 50), bool)
    hist.add(_field(50), pop)
    z = di.standardize(_field(50, 1), pop, "pixz", hist=hist)
    assert np.isnan(z).all()


def test_shrunk_sd_pulls_toward_pooled():
    hist = di.PixelHistory((100, 100))
    pop = np.ones((100, 100), bool)
    for s in range(3):
        hist.add(_field(100, s), pop)
    raw = np.sqrt(hist.var(2))
    shrunk = di.shrunk_sd(hist, pop, min_n=2, prior_df=10.0)
    assert np.std(shrunk) < np.std(raw)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        di.standardize(_field(10), np.ones((10, 10), bool), "bogus")  # type: ignore


def test_stouffer_equal_weights():
    a = np.full((2, 2), 2.0, "float32")
    assert di.stouffer([a, a])[0, 0] == pytest.approx(2 * 2 / np.sqrt(2))


def test_scene_z_saturated_layer_falls_back_to_mean_sd():
    v = np.ones((100, 100), "float32")
    v[:20] = np.linspace(0.5, 0.99, 100)[None, :]
    z = di.scene_z(v, np.ones(v.shape, bool))
    assert np.isfinite(z).all()
    assert z[:20, 0].max() < 0 < z[50, 50]
