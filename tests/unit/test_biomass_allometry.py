"""Unit tests for csdv_core.biomass.allometry (Jucker et al. 2017)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from csdv_core.biomass.allometry import jucker_agb_kg, jucker_diameter_cm


def test_angiosperm_agb_matches_equation_7() -> None:
    # H x CD = 200: 0.016 * 200**2.013 * exp(0.204**2 / 2)
    expected = 0.016 * 200.0**2.013 * math.exp(0.204**2 / 2)
    assert jucker_agb_kg(20.0, 10.0) == pytest.approx(expected)
    assert expected == pytest.approx(700.05, abs=0.1)


def test_gymnosperm_uses_combined_constants() -> None:
    # The paper reports the combined gymnosperm values as 0.109 and 1.790.
    expected = 0.109 * 200.0**1.790 * math.exp(0.204**2 / 2)
    assert jucker_agb_kg(20.0, 10.0, gymnosperm=True) == pytest.approx(expected)


def test_softwood_fraction_blends_linearly() -> None:
    a = jucker_agb_kg(20.0, 10.0, gymnosperm=0.0)
    g = jucker_agb_kg(20.0, 10.0, gymnosperm=1.0)
    half = jucker_agb_kg(20.0, 10.0, gymnosperm=0.5)
    assert half == pytest.approx(0.5 * (a + g))


def test_merging_equal_crowns_roughly_conserves_angiosperm_agb() -> None:
    # Two 8 m crowns at 20 m against one crown of the same total area.
    single = jucker_agb_kg(20.0, 8.0)
    merged = jucker_agb_kg(20.0, 8.0 * math.sqrt(2.0))
    assert merged / (2.0 * single) == pytest.approx(1.0, abs=0.02)


def test_invalid_inputs_give_nan() -> None:
    out = jucker_agb_kg(
        np.array([20.0, 0.0, np.nan, 15.0]), np.array([5.0, 5.0, 5.0, -1.0])
    )
    assert np.isfinite(out[0])
    assert np.isnan(out[1:]).all()


def test_global_diameter_model() -> None:
    expected = 0.557 * 200.0**0.809 * math.exp(0.056**2 / 2)
    assert jucker_diameter_cm(20.0, 10.0) == pytest.approx(expected)


def test_tallo_model_matches_hand_calculation():
    from csdv_core.biomass.allometry import tallo_biomes, tallo_model

    assert "Temperate Broadleaf & Mixed Forests" in tallo_biomes()
    m = tallo_model("Temperate Broadleaf & Mixed Forests")
    # refAGB Tallo_allometric_lookup_2.csv, BART row.
    a, b, c, s = (
        -3.002564858783528,
        2.1777384617970745,
        1.2442899315884226,
        1.137493818906883,
    )
    expected = s * np.exp(a + b * np.log(25.0) + c * np.log(10.0))
    np.testing.assert_allclose(m.agb_kg(25.0, 10.0), expected, rtol=1e-8)
    assert np.isnan(m.agb_kg(0.0, 10.0))
    assert np.allclose(m.cov, m.cov.T) and np.all(np.linalg.eigvalsh(m.cov) > 0)


def test_tallo_draws_center_on_fit():
    from csdv_core.biomass.allometry import tallo_model

    m = tallo_model("Temperate Conifer Forests")
    k = m.draws(4000, seed=1)
    np.testing.assert_allclose(k.mean(axis=0), [m.a, m.b, m.c], atol=0.01)
    agb = m.agb_kg(np.array([10.0, 20.0]), np.array([4.0, 8.0]), coefs=k[:5])
    assert agb.shape == (5, 2)
    with pytest.raises(KeyError):
        tallo_model("Mediterranean Forests, Woodlands & Scrub")
