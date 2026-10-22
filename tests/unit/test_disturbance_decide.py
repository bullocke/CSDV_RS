"""Unit tests for csdv_core.disturbance.decide on synthetic z stacks."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from csdv_core.config._models import DisturbanceConfig
from csdv_core.disturbance import decide
from csdv_core.disturbance.params import resolve_params

#: Cell side (m) of the synthetic grids.
CELL_M = 19.2
#: Side (cells) of the synthetic grids.
GRID_SIDE = 120
#: Gaussian SD (cells) that gives the background noise spatial correlation.
NOISE_SIGMA_CELLS = 1.0
#: NAIP years of the synthetic stacks.
YEARS = (2012, 2014, 2016, 2018)
#: Planted 6 x 6 loss patch (row and column slices).
PATCH = (slice(30, 36), slice(30, 36))
#: Isolated cells planted below the D1 threshold.
LONE_CELLS = ((10, 10), (10, 60), (60, 10), (65, 65))


def _params(rule: str) -> DisturbanceConfig:
    """Packaged defaults with the decision rule replaced.

    Args:
        rule: Decision rule name.

    Returns:
        Validated run parameters.
    """
    return resolve_params(None, {"decide.rule": rule})


def _noise_stack(seed: int) -> np.ndarray:
    """Spatially correlated unit-SD z stack, ``(len(YEARS), GRID_SIDE, GRID_SIDE)``.

    White noise is smoothed with :data:`NOISE_SIGMA_CELLS` and rescaled to unit
    SD, which resembles real z fields better than white noise does.

    Args:
        seed: Random seed.

    Returns:
        float32 z stack with date 0 set to NaN (as for a change layer).
    """
    generator = np.random.default_rng(seed)
    shape = (len(YEARS), GRID_SIDE, GRID_SIDE)
    white = generator.normal(0.0, 1.0, shape)
    stack = np.full(shape, np.nan, dtype="float32")
    for date_index in range(1, len(YEARS)):
        smoothed = ndimage.gaussian_filter(white[date_index], NOISE_SIGMA_CELLS)
        stack[date_index] = smoothed / smoothed.std()
    return stack


def _monitored() -> np.ndarray:
    """Monitored mask: every cell on dates 1 and later, none on date 0.

    Returns:
        bool ``(len(YEARS), GRID_SIDE, GRID_SIDE)``.
    """
    monitored = np.ones((len(YEARS), GRID_SIDE, GRID_SIDE), dtype=bool)
    monitored[0] = False
    return monitored


@pytest.fixture
def planted_z() -> np.ndarray:
    """Noise with a 6 x 6 loss patch and isolated deep cells on date 1."""
    z = _noise_stack(8)
    z[1][PATCH] = -6.0
    for row, col in LONE_CELLS:
        z[1, row, col] = -4.0
    return z


def test_d1_threshold_is_inclusive() -> None:
    """D1 flags z equal to -baseline_z but not NaN or unmonitored cells."""
    z = np.array([[[-3.0, -2.99, np.nan, -5.0]]], dtype="float32")
    monitored = np.array([[[True, True, True, False]]])
    flags = decide.rule_d1(z, monitored, _params("D1"))
    np.testing.assert_array_equal(flags, [[[True, False, False, False]]])


def test_fit_date_models_skips_date_zero(planted_z: np.ndarray) -> None:
    """Date 0 is skipped, later dates get a fit labelled with their year."""
    models, fits = decide.fit_date_models(planted_z, _monitored(), _params("D2"), YEARS)
    assert models[0] is None
    assert fits[0].skipped
    assert fits[0].method == decide.SKIPPED_METHOD
    assert all(model is not None for model in models[1:])
    fit_years = [fit.year for fit in fits]
    assert fit_years == list(YEARS)
    assert not fits[1].skipped
    assert fits[1].n_cells == GRID_SIDE * GRID_SIDE


def test_d2_flags_patch(planted_z: np.ndarray) -> None:
    """D2 flags the planted patch and returns probability 0 on skipped dates."""
    monitored = _monitored()
    params = _params("D2")
    models, _ = decide.fit_date_models(planted_z, monitored, params, YEARS)
    flags, probability = decide.rule_d2(planted_z, monitored, models, params)
    assert flags[1][PATCH].all()
    assert (probability[0] == 0).all()
    flagged_probability = probability[flags]
    assert (flagged_probability > params.decide.p_boundary).all()


def test_d5_flags_patch_but_not_lone_cells(planted_z: np.ndarray) -> None:
    """Spatial support keeps the patch and drops isolated deep cells."""
    monitored = _monitored()
    d1_flags = decide.rule_d1(planted_z, monitored, _params("D1"))
    params = _params("D5")
    models, _ = decide.fit_date_models(planted_z, monitored, params, YEARS)
    flags, _ = decide.rule_spatial(
        planted_z, monitored, models, params, CELL_M, persistence=False, strict=False
    )
    patch_share = float(flags[1][PATCH].mean())
    assert patch_share > 0.9
    for row, col in LONE_CELLS:
        assert d1_flags[1, row, col]
        assert not flags[1, row, col]


def test_d6_suppresses_rebound(planted_z: np.ndarray) -> None:
    """A loss followed by a gain at the next date is dropped by D6 only."""
    rebound_z = planted_z.copy()
    rebound_z[2][PATCH] = 6.0
    monitored = _monitored()
    params = _params("D6")
    models, _ = decide.fit_date_models(rebound_z, monitored, params, YEARS)
    d5_flags, _ = decide.rule_spatial(
        rebound_z, monitored, models, params, CELL_M, persistence=False, strict=False
    )
    d6_flags, _ = decide.rule_spatial(
        rebound_z, monitored, models, params, CELL_M, persistence=True, strict=False
    )
    assert d5_flags[1][PATCH].mean() > 0.9
    assert not d6_flags[1][PATCH].any()


def test_d6_last_date_equals_d5(planted_z: np.ndarray) -> None:
    """The last date has no next date, so D6 keeps the D5 probability."""
    last_z = planted_z.copy()
    last_z[-1][PATCH] = -6.0
    monitored = _monitored()
    params = _params("D6")
    models, _ = decide.fit_date_models(last_z, monitored, params, YEARS)
    loss = decide.spatial_loss_probability(last_z, monitored, models, params, CELL_M)
    persistent = decide.persistence_probability(
        last_z, monitored, models, loss, params, CELL_M
    )
    np.testing.assert_array_equal(persistent[-1], loss[-1])
    assert (persistent[-1][PATCH] > params.decide.p_boundary).all()


@pytest.mark.parametrize("persistence", [False, True])
def test_strict_variant_is_subset(planted_z: np.ndarray, persistence: bool) -> None:
    """D5s flags are a subset of D5, and D6s of D6."""
    monitored = _monitored()
    params = _params("D5")
    models, _ = decide.fit_date_models(planted_z, monitored, params, YEARS)
    loose, _ = decide.rule_spatial(
        planted_z,
        monitored,
        models,
        params,
        CELL_M,
        persistence=persistence,
        strict=False,
    )
    strict, _ = decide.rule_spatial(
        planted_z,
        monitored,
        models,
        params,
        CELL_M,
        persistence=persistence,
        strict=True,
    )
    assert strict.any()
    strict_outside_loose = strict & ~loose
    assert not strict_outside_loose.any()


def test_first_onset_takes_earliest_date() -> None:
    """The onset is the first flagged date, -1 where none."""
    flags = np.zeros((3, 1, 3), dtype=bool)
    flags[1, 0, 0] = True
    flags[2, 0, 0] = True
    flags[2, 0, 1] = True
    onset = decide.first_onset(flags)
    assert onset.dtype == np.int8
    np.testing.assert_array_equal(onset, [[1, 2, decide.NO_ONSET]])


def test_apply_mmu_drops_small_and_keeps_diagonal() -> None:
    """Objects below the MMU go, diagonal chains count as one object."""
    onset = np.full((6, 6), decide.NO_ONSET, dtype="int8")
    onset[0, 0] = 1
    onset[1, 1] = 1
    onset[2, 2] = 1
    onset[4, 4] = 1
    onset[4, 5] = 1
    onset[0, 5] = 2
    onset[1, 5] = 2
    onset[2, 4] = 2
    filtered = decide.apply_mmu(onset, cell_ha=0.1, mmu_ha=0.3)
    assert filtered.dtype == np.int8
    assert (filtered[[0, 1, 2], [0, 1, 2]] == 1).all()
    assert (filtered[4, 4:] == decide.NO_ONSET).all()
    assert (filtered[[0, 1, 2], [5, 5, 4]] == 2).all()
    unchanged = decide.apply_mmu(onset, cell_ha=0.1, mmu_ha=0.0)
    np.testing.assert_array_equal(unchanged, onset)


def test_mmu_cells_and_growth_cap() -> None:
    """1 acre is 11 cells at 19.2 m, 0.11 ha is 3, and growth never exceeds the MMU."""
    cell_ha = 19.2**2 / 10_000
    assert decide.mmu_cells(0.405, cell_ha) == 11
    assert decide.mmu_cells(0.11, cell_ha) == 3
    assert decide.mmu_cells(0.0, cell_ha) == 0
    assert decide.growth_min_cells(0.405, 0.11, cell_ha) == 3
    assert decide.growth_min_cells(0.05, 0.11, cell_ha) == 2


def test_interval_detections_keeps_second_detection() -> None:
    """A cell kept at date 1 is kept again at date 2 when flagged in a large object."""
    flags = np.zeros((3, 8, 8), dtype=bool)
    flags[1, 0:3, 0:3] = True
    flags[2, 0:4, 0:4] = True
    detected = decide.interval_detections(
        flags, cell_ha=0.1, mmu_ha=0.9, growth_mmu_ha=0.9
    )
    assert detected[1].sum() == 9
    assert detected[2].sum() == 16
    assert detected[2, 0, 0] and detected[1, 0, 0]
    first = decide.first_onset(detected)
    assert first[0, 0] == 1
    assert first[3, 3] == 2


def test_interval_detections_growth_rule() -> None:
    """Small objects touching an earlier detection survive, isolated ones do not."""
    flags = np.zeros((3, 10, 10), dtype=bool)
    flags[1, 0:3, 0:3] = True
    flags[2, 3, 0:2] = True
    flags[2, 8, 8:10] = True
    flags[2, 0, 5] = True
    detected = decide.interval_detections(
        flags, cell_ha=0.1, mmu_ha=0.9, growth_mmu_ha=0.2
    )
    assert detected[1].sum() == 9
    assert detected[2, 3, 0:2].all()
    assert not detected[2, 8, 8:10].any()
    assert not detected[2, 0, 5]
    without_growth = decide.interval_detections(
        flags, cell_ha=0.1, mmu_ha=0.9, growth_mmu_ha=0.9
    )
    assert not without_growth[2].any()
    any_size = decide.interval_detections(
        flags, cell_ha=0.1, mmu_ha=0.9, growth_mmu_ha=0.0
    )
    assert any_size[2].sum() == 2


def test_interval_detections_frees_cells_of_discarded_objects() -> None:
    """A flag removed by the MMU does not block a later detection of the cell."""
    flags = np.zeros((3, 8, 8), dtype=bool)
    flags[1, 0, 0] = True
    flags[2, 0:3, 0:3] = True
    detected = decide.interval_detections(
        flags, cell_ha=0.1, mmu_ha=0.9, growth_mmu_ha=0.9
    )
    assert not detected[1].any()
    assert detected[2, 0, 0]
    onset_mmu = decide.apply_mmu(decide.first_onset(flags), cell_ha=0.1, mmu_ha=0.9)
    assert onset_mmu[0, 0] == decide.NO_ONSET


def test_onset_stack_is_one_hot() -> None:
    onset = np.array([[1, -1], [2, 1]], dtype="int8")
    stack = decide.onset_stack(onset, 3)
    assert stack.shape == (3, 2, 2)
    assert not stack[0].any()
    np.testing.assert_array_equal(stack[1], [[True, False], [False, True]])
    np.testing.assert_array_equal(decide.first_onset(stack), onset)


def test_skipped_date_has_no_detections(planted_z: np.ndarray) -> None:
    """A date with too few monitored cells is skipped and detects nothing."""
    monitored = _monitored()
    monitored[1] = False
    monitored[1][PATCH] = True
    result = decide.decide(planted_z, monitored, _params("D6"), CELL_M, years=YEARS)
    assert result.fits[1].skipped
    assert np.isnan(result.fits[1].mu0)
    assert result.fits[1].n_cells == 36
    assert (result.onset_index != 1).all()
    assert np.isnan(result.probability[0]).all()
    assert np.isnan(result.probability[1]).all()
    unmonitored = ~monitored[2]
    assert np.isnan(result.probability[2][unmonitored]).all()


def test_decide_d6_maps_patch_and_years(planted_z: np.ndarray) -> None:
    """decide() finds the patch at date 1 and converts indices to years."""
    monitored = _monitored()
    monitored[:, 0, 0] = False
    result = decide.decide(planted_z, monitored, _params("D6"), CELL_M, years=YEARS)
    assert result.rule == "D6"
    assert len(result.fits) == len(YEARS)
    patch_onset = result.onset_index_mmu[PATCH]
    patch_detected_share = float((patch_onset == 1).mean())
    assert patch_detected_share > 0.9
    onset_year = decide.onset_index_to_year(result.onset_index_mmu, YEARS, monitored)
    assert onset_year.dtype == np.int16
    patch_years = onset_year[PATCH]
    detected_in_patch = patch_onset == 1
    assert (patch_years[detected_in_patch] == YEARS[1]).all()
    assert onset_year[0, 0] == decide.ONSET_YEAR_UNMONITORED
    assert onset_year[0, 1] == decide.ONSET_YEAR_NONE
    assert result.flags.shape == planted_z.shape
    assert result.detected.shape == planted_z.shape
    assert not (result.detected & ~result.flags).any()
    np.testing.assert_array_equal(
        result.onset_index_mmu, decide.first_onset(result.detected)
    )


def test_decide_d1_has_no_probability(planted_z: np.ndarray) -> None:
    """D1 returns no probability stack but still lists the fits."""
    result = decide.decide(planted_z, _monitored(), _params("D1"), CELL_M, years=YEARS)
    assert result.probability is None
    assert len(result.fits) == len(YEARS)


def test_decide_rejects_shape_mismatch(planted_z: np.ndarray) -> None:
    """A monitored mask of another shape raises ValueError."""
    monitored = np.ones((2, GRID_SIDE, GRID_SIDE), dtype=bool)
    with pytest.raises(ValueError):
        decide.decide(planted_z, monitored, _params("D6"), CELL_M, years=YEARS)
