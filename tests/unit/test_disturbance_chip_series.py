"""Tests for the DI series of the chip figures (csdv_core.disturbance._chip_panels)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from csdv_core.change.di_decide import EmpiricalNull, LfdrModel
from csdv_core.config import load_disturbance
from csdv_core.config._models import DisturbanceConfig
from csdv_core.disturbance._chip_panels import decision_series, probability_cutoff

#: Side of the synthetic grid, in cells.
GRID_SIDE = 120
#: Rows and columns of the planted loss patch on the last date.
PATCH = (slice(50, 60), slice(50, 60))
#: Raw z of the planted patch, above the per-cell cut of -3.
PATCH_Z = -2.5


def _params(rule: str) -> DisturbanceConfig:
    """Default parameters with another decision rule.

    Args:
        rule: Decision rule.

    Returns:
        The parameters.
    """
    params = load_disturbance()
    decide_params = params.decide.model_copy(update={"rule": rule})
    return params.model_copy(update={"decide": decide_params})


def _saved_run(rule: str) -> SimpleNamespace:
    """A stand-in saved run with noise z and one coherent moderate loss patch.

    Args:
        rule: Decision rule of the run.

    Returns:
        Object with the attributes :func:`decision_series` reads.
    """
    rng = np.random.default_rng(0)
    shape = (3, GRID_SIDE, GRID_SIDE)
    z = rng.normal(0.0, 1.0, shape).astype("float32")
    z[0] = np.nan
    z[2][PATCH] = PATCH_Z
    forest = np.ones(shape, dtype=bool)
    layers = SimpleNamespace(forest=forest)
    grid = SimpleNamespace(cell_m=19.2)
    return SimpleNamespace(
        params=_params(rule),
        grid=grid,
        layers=layers,
        z=z,
        years=(2014, 2016, 2018),
    )


def test_probability_cutoff_finds_boundary_crossing() -> None:
    """The cutoff is the highest z whose probability exceeds the boundary."""
    null = EmpiricalNull(mu0=0.0, sigma0=1.0, p0=1.0, method="central")
    centres = np.linspace(-5.0, 0.0, 501)
    lfdr_curve = centres / -5.0
    lfdr_curve = 1.0 - lfdr_curve
    model = LfdrModel(null, centres, lfdr_curve)
    cutoff = probability_cutoff(model, 0.5)
    assert abs(cutoff - -2.5) < 0.02


def test_probability_cutoff_nan_when_nothing_passes() -> None:
    """A flat lfdr of one never passes the boundary."""
    null = EmpiricalNull(mu0=0.0, sigma0=1.0, p0=1.0, method="central")
    centres = np.linspace(-5.0, 0.0, 51)
    lfdr_curve = np.ones_like(centres)
    model = LfdrModel(null, centres, lfdr_curve)
    assert np.isnan(probability_cutoff(model, 0.5))


def test_spatial_rule_series_is_smoothed_and_patch_passes() -> None:
    """D6 shows smoothed z, and a coherent patch above -3 lies below the cutoff."""
    saved = _saved_run("D6")
    series = decision_series(saved)
    assert "25 m" in series.title
    assert np.isnan(series.cutoffs[0])
    cutoff = series.cutoffs[2]
    assert -3.0 < cutoff < 0.0
    patch_values = series.values[2][PATCH]
    assert np.median(patch_values) < cutoff
    background_sd = np.nanstd(series.values[2][:40, :40])
    assert background_sd < 0.8


def test_d1_series_uses_fixed_cut() -> None:
    """D1 shows the raw z with the fixed -baseline_z cut on fitted dates."""
    saved = _saved_run("D1")
    series = decision_series(saved)
    assert series.values is saved.z
    assert np.isnan(series.cutoffs[0])
    assert series.cutoffs[1:] == (-3.0, -3.0)


def test_d2_series_cutoff_below_smoothed_cutoff() -> None:
    """D2's raw cutoff lies further from zero than the smoothed D6 cutoff."""
    raw_series = decision_series(_saved_run("D2"))
    smoothed_series = decision_series(_saved_run("D6"))
    assert raw_series.cutoffs[1] < smoothed_series.cutoffs[1]
