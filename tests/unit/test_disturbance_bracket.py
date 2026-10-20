"""Unit tests for the bracketed rules D7 and D7s on synthetic CHM stacks.

The synthetic site is uniform forest with spatially correlated height noise
and three planted patches, all with a 10 m drop in the interval ending at
date index 2: a persistent step down, a spike in the earlier image that
returns, and a dip in the later image that recovers.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from csdv_core.config._models import DisturbanceConfig
from csdv_core.disturbance import decide
from csdv_core.disturbance.layers import (
    CellLayers,
    bracket_changes,
    change_layer,
    population_mask,
)
from csdv_core.disturbance.params import resolve_params
from csdv_core.disturbance.standardize import (
    quantize_z,
    standardize_brackets,
    standardize_stack,
)

#: Cell side (m) of the synthetic grids.
CELL_M = 19.2
#: Side (cells) of the synthetic grids.
GRID_SIDE = 90
#: NAIP years of the synthetic stacks.
YEARS = (2012, 2014, 2016, 2018)
#: Mean canopy height (m).
BASE_HEIGHT_M = 20.0
#: Height noise SD (m) after smoothing.
NOISE_SD_M = 1.0
#: Height change (m) of the planted patches.
DROP_M = 10.0
#: Date index of the planted drop.
DROP_INDEX = 2
#: Persistent step down: lower from the drop date on.
STEP_PATCH = (slice(15, 22), slice(15, 22))
#: Spike: the earlier image of the drop interval is high, then returns.
SPIKE_PATCH = (slice(15, 22), slice(60, 67))
#: Dip: the later image of the drop interval is low, then recovers.
DIP_PATCH = (slice(60, 67), slice(15, 22))
#: Persistent step down in the last interval.
LAST_PATCH = (slice(60, 67), slice(60, 67))


def _params(rule: str) -> DisturbanceConfig:
    """Packaged defaults with the decision rule replaced.

    Args:
        rule: Decision rule name.

    Returns:
        Validated run parameters.
    """
    return resolve_params(None, {"decide.rule": rule})


def _layers(height: np.ndarray) -> CellLayers:
    """Cell layers of a uniform forest with the given heights.

    Args:
        height: float32 ``(T, rows, cols)`` heights.

    Returns:
        Layers with full cover and every cell forest.
    """
    cover = np.ones(height.shape, dtype="float32")
    forest = np.ones(height.shape, dtype=bool)
    return CellLayers(
        years=YEARS,
        chm_height=height,
        chm_cover=cover,
        valid_share=cover,
        forest=forest,
    )


def _heights(seed: int = 3) -> np.ndarray:
    """Noisy forest heights with the four planted patches.

    Args:
        seed: Random seed.

    Returns:
        float32 ``(len(YEARS), GRID_SIDE, GRID_SIDE)`` heights (m).
    """
    generator = np.random.default_rng(seed)
    shape = (len(YEARS), GRID_SIDE, GRID_SIDE)
    white = generator.normal(0.0, 1.0, shape)
    height = np.empty(shape, dtype="float32")
    for date_index in range(len(YEARS)):
        smoothed = ndimage.gaussian_filter(white[date_index], 1.0)
        scaled = smoothed / smoothed.std() * NOISE_SD_M
        height[date_index] = BASE_HEIGHT_M + scaled
    height[DROP_INDEX:][(slice(None), *STEP_PATCH)] -= DROP_M
    height[DROP_INDEX - 1][SPIKE_PATCH] += DROP_M
    height[DROP_INDEX][DIP_PATCH] -= DROP_M
    last_index = len(YEARS) - 1
    height[last_index][LAST_PATCH] -= DROP_M
    return height


@pytest.fixture(scope="module")
def stacks() -> dict[str, object]:
    """Quantized main and bracket z, monitored mask and fitted models."""
    params = _params("D7")
    height = _heights()
    layers = _layers(height)
    population = population_mask(layers.forest)
    change = change_layer(layers, "d_chm_h")
    z_unrounded = standardize_stack(change, population, params, CELL_M)
    z = quantize_z(z_unrounded)
    back_change, forward_change = bracket_changes(layers, "d_chm_h")
    bracket_z = standardize_brackets(
        back_change, forward_change, population, params, CELL_M
    )
    monitored = population.copy()
    monitored[0] = False
    models, _ = decide.fit_date_models(z, monitored, params, YEARS)
    return {
        "z": z,
        "bracket_z": bracket_z,
        "monitored": monitored,
        "models": models,
    }


def _patch_share(flags: np.ndarray, patch: tuple[slice, slice]) -> float:
    """Share of a patch's cells flagged.

    Args:
        flags: bool ``(rows, cols)``.
        patch: Row and column slices.

    Returns:
        Flagged share in [0, 1].
    """
    patch_flags = flags[patch]
    return float(patch_flags.mean())


def _flags(stacks: dict[str, object], rule: str) -> np.ndarray:
    """Flags of a spatial or bracketed rule on the fixture stacks.

    Args:
        stacks: The fixture.
        rule: ``D5``, ``D7`` or ``D7s``.

    Returns:
        bool ``(T, rows, cols)``.
    """
    params = _params(rule)
    if rule == "D5":
        flags, _ = decide.rule_spatial(
            stacks["z"],
            stacks["monitored"],
            stacks["models"],
            params,
            CELL_M,
            persistence=False,
            strict=False,
        )
        return flags
    flags, _ = decide.rule_bracketed(
        stacks["z"],
        stacks["monitored"],
        stacks["models"],
        stacks["bracket_z"],
        params,
        CELL_M,
        YEARS,
        strict=rule.endswith("s"),
    )
    return flags


def test_bracket_changes_values() -> None:
    state = np.arange(4, dtype="float32")[:, np.newaxis, np.newaxis] ** 2
    height = np.broadcast_to(state, (4, 2, 2)).copy()
    height[0, 0, 0] = np.nan
    layers = _layers(height)
    back, forward = bracket_changes(layers, "d_chm_h")
    assert back.dtype == np.float32
    assert np.isnan(back[:2]).all()
    assert back[2, 1, 1] == pytest.approx(4.0)
    assert back[3, 1, 1] == pytest.approx(8.0)
    assert np.isnan(back[2, 0, 0])
    assert np.isnan(forward[0]).all()
    assert np.isnan(forward[3]).all()
    assert forward[1, 1, 1] == pytest.approx(4.0)
    assert forward[2, 1, 1] == pytest.approx(8.0)
    assert np.isnan(forward[1, 0, 0])


def test_bracket_changes_rejects_missing_ndvi() -> None:
    height = np.zeros((3, 2, 2), dtype="float32")
    layers = _layers(height)
    with pytest.raises(ValueError, match="ndvi"):
        bracket_changes(layers, "d_ndvi")


def test_d5_flags_all_planted_drops(stacks: dict[str, object]) -> None:
    flags = _flags(stacks, "D5")
    drop_flags = flags[DROP_INDEX]
    for patch in (STEP_PATCH, SPIKE_PATCH, DIP_PATCH):
        assert _patch_share(drop_flags, patch) > 0.8


def test_d7_keeps_step_and_drops_spike_and_dip(stacks: dict[str, object]) -> None:
    flags = _flags(stacks, "D7")
    drop_flags = flags[DROP_INDEX]
    assert _patch_share(drop_flags, STEP_PATCH) > 0.8
    assert _patch_share(drop_flags, SPIKE_PATCH) == 0.0
    assert _patch_share(drop_flags, DIP_PATCH) == 0.0
    any_date = flags.any(axis=0)
    assert _patch_share(any_date, SPIKE_PATCH) == 0.0
    assert _patch_share(any_date, DIP_PATCH) == 0.0


def test_d7_last_date_uses_back_check_only(stacks: dict[str, object]) -> None:
    params = _params("D7")
    z = stacks["z"]
    monitored = stacks["monitored"]
    back_z, forward_z = stacks["bracket_z"]
    probability = decide.bracketed_probability(
        z, monitored, stacks["models"], stacks["bracket_z"], params, CELL_M, YEARS
    )
    main = decide.spatial_loss_probability(
        z, monitored, stacks["models"], params, CELL_M
    )
    back = decide.check_probability(back_z, monitored, params, CELL_M, YEARS)
    forward = decide.check_probability(forward_z, monitored, params, CELL_M, YEARS)
    last_index = len(YEARS) - 1
    assert np.isnan(forward[last_index]).all()
    expected_last = np.fmin(main[last_index], back[last_index])
    np.testing.assert_array_equal(probability[last_index], expected_last)
    assert np.isnan(back[1]).all()
    expected_first = np.fmin(main[1], forward[1])
    np.testing.assert_array_equal(probability[1], expected_first)
    flags = _flags(stacks, "D7")
    assert _patch_share(flags[last_index], LAST_PATCH) > 0.8


def test_d7s_is_subset_of_d7(stacks: dict[str, object]) -> None:
    plain = _flags(stacks, "D7")
    strict = _flags(stacks, "D7s")
    assert not (strict & ~plain).any()
    assert strict.sum() > 0


def test_decide_d7_needs_bracket_z(stacks: dict[str, object]) -> None:
    params = _params("D7")
    with pytest.raises(ValueError, match="bracket_z"):
        decide.decide(stacks["z"], stacks["monitored"], params, CELL_M, years=YEARS)


def test_decide_d7_maps_step_onset(stacks: dict[str, object]) -> None:
    params = _params("D7")
    result = decide.decide(
        stacks["z"],
        stacks["monitored"],
        params,
        CELL_M,
        years=YEARS,
        bracket_z=stacks["bracket_z"],
    )
    assert result.rule == "D7"
    step_onset = result.onset_index_mmu[STEP_PATCH]
    assert (step_onset == DROP_INDEX).mean() > 0.8
    assert (result.onset_index_mmu[SPIKE_PATCH] == decide.NO_ONSET).all()
    assert (result.onset_index_mmu[DIP_PATCH] == decide.NO_ONSET).all()
    assert np.isnan(result.probability[0]).all()
