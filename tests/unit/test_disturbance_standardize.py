"""Unit tests for csdv_core.disturbance.standardize on synthetic fields."""

from __future__ import annotations

import numpy as np
import pytest

from csdv_core.change import di
from csdv_core.config._models import DisturbanceConfig
from csdv_core.disturbance import standardize
from csdv_core.disturbance.params import resolve_params

#: Cell side (m) of the synthetic grids.
CELL_M = 19.2
#: Side (cells) of the synthetic grids.
GRID_SIDE = 80


def _params(method: str) -> DisturbanceConfig:
    """Packaged defaults with the standardization method replaced.

    Args:
        method: Standardization method name.

    Returns:
        Validated run parameters.
    """
    return resolve_params(None, {"standardize.method": method})


def _noise_field(seed: int) -> np.ndarray:
    """Standard normal float32 field of shape ``(GRID_SIDE, GRID_SIDE)``.

    Args:
        seed: Random seed.

    Returns:
        float32 noise field.
    """
    generator = np.random.default_rng(seed)
    field = generator.normal(0.0, 1.0, (GRID_SIDE, GRID_SIDE))
    return field.astype("float32")


def test_window_cells_rounds_to_odd() -> None:
    """150 m at 19.2 m is 9 cells, and every result is odd."""
    assert standardize.window_cells(150, 19.2) == 9
    for window_m in (10.0, 40.0, 100.0, 150.0, 500.0):
        side = standardize.window_cells(window_m, 19.2)
        assert side % 2 == 1
        assert side >= 1


def test_window_cells_rejects_non_positive() -> None:
    """Zero or negative sizes raise ValueError."""
    with pytest.raises(ValueError):
        standardize.window_cells(0.0, 19.2)
    with pytest.raises(ValueError):
        standardize.window_cells(150.0, -1.0)


@pytest.mark.parametrize(("method", "robust"), [("scene", True), ("scenemean", False)])
def test_scene_methods_match_scene_z(method: str, robust: bool) -> None:
    """scene and scenemean equal di.scene_z with the matching robust flag."""
    values = _noise_field(1)
    population = np.ones(values.shape, dtype=bool)
    params = _params(method)
    result = standardize.standardize_date(values, population, params, CELL_M)
    expected = di.scene_z(values, population, robust=robust)
    np.testing.assert_array_equal(result, expected)


def test_twopass_lowers_z_on_planted_patch() -> None:
    """The second pass drops the patch from its own window reference."""
    values = _noise_field(2)
    values[30:42, 30:42] -= 2.5
    population = np.ones(values.shape, dtype=bool)
    single = standardize.standardize_date(values, population, _params("local"), CELL_M)
    twopass = standardize.standardize_date(
        values, population, _params("local_twopass"), CELL_M
    )
    single_patch_mean = float(single[30:42, 30:42].mean())
    twopass_patch_mean = float(twopass[30:42, 30:42].mean())
    assert twopass_patch_mean < single_patch_mean


def test_standardize_stack_leaves_empty_dates_nan() -> None:
    """Date 0 of a change stack (all NaN) stays NaN, later dates are finite."""
    later = _noise_field(3)
    change = np.stack([np.full(later.shape, np.nan, dtype="float32"), later])
    population = np.ones(change.shape, dtype=bool)
    z = standardize.standardize_stack(change, population, _params("scene"), CELL_M)
    assert z.dtype == np.float32
    assert np.isnan(z[0]).all()
    assert np.isfinite(z[1]).all()


def test_standardize_stack_rejects_shape_mismatch() -> None:
    """Different stack shapes raise ValueError."""
    change = np.zeros((2, 4, 4), dtype="float32")
    population = np.ones((2, 4, 5), dtype=bool)
    with pytest.raises(ValueError):
        standardize.standardize_stack(change, population, _params("scene"), CELL_M)


def test_encode_decode_round_trip_and_nodata() -> None:
    """Encoding rounds to 0.01, clips, and maps NaN to the nodata value."""
    z = np.array([1.234, -3.456, np.nan, 500.0, -500.0], dtype="float32")
    encoded = standardize.encode_z(z)
    assert encoded.dtype == np.int16
    expected_codes = np.array(
        [123, -346, standardize.Z_NODATA, 32767, -32767], dtype="int16"
    )
    np.testing.assert_array_equal(encoded, expected_codes)
    decoded = standardize.decode_z(encoded)
    assert decoded.dtype == np.float32
    assert np.isnan(decoded[2])
    np.testing.assert_allclose(decoded[:2], [1.23, -3.46], atol=1e-6)


def test_quantize_is_idempotent() -> None:
    """Quantizing twice changes nothing and equals decode(encode(z))."""
    z = _noise_field(4)
    z[0, 0] = np.nan
    quantized = standardize.quantize_z(z)
    encoded = standardize.encode_z(z)
    decoded = standardize.decode_z(encoded)
    np.testing.assert_array_equal(quantized, decoded)
    requantized = standardize.quantize_z(quantized)
    np.testing.assert_array_equal(requantized, quantized)
