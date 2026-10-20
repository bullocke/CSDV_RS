"""csdv_core.disturbance.standardize: turn a change layer into z stacks.

Each date of the change layer is standardized against the forest population
of that date with :mod:`csdv_core.change.di`: whole-scene median/MAD
(``scene``), scene mean/SD (``scenemean``), a square moving window
(``local``), or the window applied twice (``local_twopass``). In the second
pass, cells whose spatially supported loss probability exceeds
``decide.p_boundary`` leave the window population, so a large disturbance
does not set its own reference (``02_standardize._local`` with the ``i``
suffix). z is then quantized to 0.01, the precision it is stored at, so a
run reloaded with ``--from-run`` decides on exactly the same values.

Inputs: change stack and population mask ``(T, rows, cols)``.
Outputs: float32 z stacks of the same shape, NaN where undefined.
"""

from __future__ import annotations

import logging

import numpy as np

from csdv_core.change import di, di_decide
from csdv_core.config._models import DisturbanceConfig

logger = logging.getLogger(__name__)

#: z is stored as ``round(z * Z_SCALE)`` in int16.
Z_SCALE = 100
#: int16 nodata value of stored z.
Z_NODATA = -32768
#: Largest stored magnitude, ``|round(z * Z_SCALE)| <= Z_CLIP``.
Z_CLIP = 32767
#: Methods that use a moving window (and carry ``window_m`` in file names).
LOCAL_METHODS: tuple[str, ...] = ("local", "local_twopass")


def window_cells(window_m: float, cell_m: float) -> int:
    """Window side in cells, rounded to the nearest integer and forced odd.

    ``n = round(window_m / cell_m)``, then ``n + 1`` if ``n`` is even, as in
    ``02_standardize._window_px``. ``window_cells(150, 19.2) == 9``.

    Args:
        window_m: Requested window side (m).
        cell_m: Cell side (m).

    Returns:
        Odd window side in cells (at least 1).

    Raises:
        ValueError: If either argument is not positive.
    """
    if window_m <= 0 or cell_m <= 0:
        raise ValueError(
            f"window_m and cell_m must be positive, got {window_m} and {cell_m}"
        )
    ratio = window_m / cell_m
    rounded = round(ratio)
    side = int(rounded)
    return side + 1 - side % 2


def sigmas_in_cells(params: DisturbanceConfig, cell_m: float) -> list[float]:
    """Spatial smoothing scales in cells, ``decide.spatial_sigma_m / cell_m``.

    Args:
        params: Run parameters.
        cell_m: Cell side (m).

    Returns:
        One Gaussian SD (cells) per configured scale.
    """
    return [sigma_m / cell_m for sigma_m in params.decide.spatial_sigma_m]


def _local_z(
    values: np.ndarray,
    population: np.ndarray,
    side: int,
    params: DisturbanceConfig,
) -> np.ndarray:
    """One moving-window pass of :func:`csdv_core.change.di.local_z`.

    Args:
        values: float32 ``(rows, cols)`` change values.
        population: bool ``(rows, cols)`` window population.
        side: Odd window side in cells.
        params: Run parameters (``standardize`` section).

    Returns:
        float32 ``(rows, cols)`` z.
    """
    return di.local_z(
        values,
        population,
        side,
        min_frac=params.standardize.min_frac,
        trim_z=params.standardize.trim_z,
        sd_floor=params.standardize.sd_floor,
    )


def standardize_date(
    values: np.ndarray,
    population: np.ndarray,
    params: DisturbanceConfig,
    cell_m: float,
) -> np.ndarray:
    """Standardize one date of the change layer.

    ``scene`` and ``scenemean`` use :func:`csdv_core.change.di.scene_z`
    (median/MAD and mean/SD). ``local`` uses
    :func:`csdv_core.change.di.local_z` on a :func:`window_cells` window.
    ``local_twopass`` repeats the local z after removing cells whose spatial
    loss probability exceeds ``decide.p_boundary`` from the population.

    Args:
        values: float32 ``(rows, cols)`` change values, NaN = missing.
        population: bool ``(rows, cols)`` population for this date.
        params: Run parameters (``standardize`` and ``decide`` sections).
        cell_m: Cell side (m).

    Returns:
        float32 ``(rows, cols)`` z, NaN where ``values`` is NaN or the
        population is too small for statistics (all NaN in that case).
    """
    method = params.standardize.method
    if method == "scene":
        return di.scene_z(values, population, robust=True)
    if method == "scenemean":
        return di.scene_z(values, population, robust=False)
    side = window_cells(params.standardize.window_m, cell_m)
    first_pass = _local_z(values, population, side, params)
    if method == "local":
        return first_pass
    sigmas_cells = sigmas_in_cells(params, cell_m)
    finite_first = np.isfinite(first_pass)
    support = population & finite_first
    loss_probability = di_decide.spatial_prob(
        first_pass,
        support,
        sigmas_cells,
        bin_w=params.decide.bin_w,
        smooth_bins=params.decide.smooth_bins,
    )
    supported_loss = loss_probability > params.decide.p_boundary
    reduced_population = population & ~supported_loss
    return _local_z(values, reduced_population, side, params)


def standardize_stack(
    change: np.ndarray,
    population: np.ndarray,
    params: DisturbanceConfig,
    cell_m: float,
) -> np.ndarray:
    """Standardize every date of a change stack.

    Dates with no finite value (always index 0 of a change layer) stay NaN.

    Args:
        change: float32 ``(T, rows, cols)`` from
            :func:`csdv_core.disturbance.layers.change_layer`.
        population: bool ``(T, rows, cols)`` from
            :func:`csdv_core.disturbance.layers.population_mask`.
        params: Run parameters.
        cell_m: Cell side (m).

    Returns:
        float32 ``(T, rows, cols)`` z (not yet quantized).

    Raises:
        ValueError: If the two stacks differ in shape.
    """
    if change.shape != population.shape:
        raise ValueError(
            f"change {change.shape} and population {population.shape} differ in shape"
        )
    z = np.full(change.shape, np.nan, dtype="float32")
    for date_index in range(change.shape[0]):
        values = change[date_index]
        if not np.isfinite(values).any():
            continue
        z[date_index] = standardize_date(values, population[date_index], params, cell_m)
        logger.debug("Standardized date index %d", date_index)
    return z


def encode_z(z: np.ndarray) -> np.ndarray:
    """Encode z as stored: ``int16(clip(round(z * 100), -32767, 32767))``.

    Args:
        z: float z array, any shape, NaN = missing.

    Returns:
        int16 array of the same shape with :data:`Z_NODATA` where z is NaN.
    """
    scaled = z * Z_SCALE
    rounded = np.round(scaled)
    clipped = np.clip(rounded, -Z_CLIP, Z_CLIP)
    finite = np.isfinite(z)
    encoded = np.where(finite, clipped, Z_NODATA)
    return encoded.astype("int16")


def decode_z(encoded: np.ndarray) -> np.ndarray:
    """Decode stored z: ``float32(encoded) / 100`` with :data:`Z_NODATA` as NaN.

    Args:
        encoded: int16 array from :func:`encode_z` or a z GeoTIFF.

    Returns:
        float32 array of the same shape.
    """
    decoded = encoded.astype("float32")
    decoded[encoded == Z_NODATA] = np.nan
    return decoded / float(Z_SCALE)


def quantize_z(z: np.ndarray) -> np.ndarray:
    """Round z to the stored precision, ``decode_z(encode_z(z))``.

    The result is bit-identical to what :func:`csdv_core.disturbance.store.read_z`
    returns for the same z, so decisions match between a fresh run and a
    ``--from-run`` rerun.

    Args:
        z: float z array, any shape, NaN = missing.

    Returns:
        float32 array of the same shape.
    """
    encoded = encode_z(z)
    return decode_z(encoded)


__all__ = [
    "LOCAL_METHODS",
    "Z_NODATA",
    "Z_SCALE",
    "decode_z",
    "encode_z",
    "quantize_z",
    "sigmas_in_cells",
    "standardize_date",
    "standardize_stack",
    "window_cells",
]
