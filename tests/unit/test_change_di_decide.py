"""Tests for csdv_core.change.di_decide."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from csdv_core.change import di_decide as dd


def _null_field(n: int = 400, seed: int = 0, smooth: float = 0.0) -> np.ndarray:
    z = np.random.default_rng(seed).normal(0, 1, (n, n))
    if smooth:
        z = ndimage.gaussian_filter(z, smooth)
        z /= z.std()
    return z.astype("float32")


def test_fit_null_recovers_shifted_null():
    z = (_null_field() * 1.3 + 0.4).ravel()
    z[:4000] = np.random.default_rng(1).normal(-6, 1, 4000)  # 2.5% losses
    null = dd.fit_null(z)
    assert null.mu0 == pytest.approx(0.4, abs=0.05)
    assert null.sigma0 == pytest.approx(1.3, abs=0.08)
    assert null.p0 == pytest.approx(0.975, abs=0.03)


def test_inserted_patch_gets_high_probability():
    z = _null_field()
    z[100:130, 100:130] = -6.0
    pop = np.ones(z.shape, bool)
    model = dd.lfdr_model(z, pop)
    p = model.prob(z)
    assert p[100:130, 100:130].min() > 0.95
    assert (p[200:, 200:] > 0.5).mean() < 0.001


def test_lfdr_is_one_above_centre_and_monotone():
    z = _null_field()
    model = dd.lfdr_model(z)
    assert model.lfdr(np.array([0.5, 3.0], "float32")).tolist() == [1.0, 1.0]
    assert np.all(np.diff(model.lfdr_curve) >= 0)


def test_cluster_probability_null_field_rejects_most():
    z = _null_field(smooth=1.5)
    pop = np.ones(z.shape, bool)
    res = dd.cluster_probability(z, dd.lfdr_model(z, pop), pop, p_form=0.0)
    # With no real losses, clusters should rarely look more real than the mirror.
    probs = res.prob[1:]
    assert probs.size > 0
    assert (probs > 0.5).mean() < 0.2


def test_cluster_probability_finds_patch():
    z = _null_field(smooth=1.0)
    z[50:80, 50:80] -= 5.0
    pop = np.ones(z.shape, bool)
    res = dd.cluster_probability(z, dd.lfdr_model(z, pop), pop)
    assert res.prob_map()[60, 60] > 0.9


def test_bh_reject_controls_null():
    z = _null_field()
    null = dd.EmpiricalNull(0.0, 1.0, 1.0, "known")
    assert dd.bh_reject(z, null, np.ones(z.shape, bool)).mean() < 1e-4


def test_accumulate_persistent_vs_transient():
    # Two pixels, three dates: weak evidence twice versus once then normal.
    llr = np.array([[[2.0, 2.0]], [[2.5, -3.0]], [[0.0, 0.0]]], "float32")
    res = dd.accumulate(llr, prior=0.02)
    assert res.prob[1, 0, 0] > 0.5  # repeated weak loss builds up
    assert res.prob[1, 0, 1] < 0.05  # one dip followed by normal resets
    assert res.onset[1, 0, 0] == 0
    first = res.first(np.ones(llr.shape, bool))
    assert first.tolist() == [[0, -1]]


def test_accumulate_single_date_matches_lfdr():
    z = _null_field(100)
    z[:10, :10] = -4
    model = dd.lfdr_model(z)
    res = dd.accumulate(model.llr(z)[None], prior=1 - model.null.p0)
    p = model.prob(z)
    assert np.allclose(res.prob[0][:10, :10], p[:10, :10], atol=1e-3)


def test_confirm_next_requires_follow_up():
    llr = np.array([[[3.0, 3.0]], [[3.0, -3.0]]], "float32")
    p = dd.confirm_next(llr, prior=0.02)
    assert p[0, 0, 0] > 0.5 > p[0, 0, 1]
