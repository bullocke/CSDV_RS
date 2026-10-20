"""csdv_core.disturbance.decide: decision rules, onset date and minimum mapping unit.

A port of the research ``_lib.fit_models``, ``run_variant`` (rules D1 and
D2), ``_spatial`` (D5, D5s, D6, D6s) and ``apply_mmu`` onto
:mod:`csdv_core.change.di_decide`:

- **D1**: ``z <= -baseline_z`` on a monitored cell.
- **D2**: lfdr posterior ``P > p_boundary``.
- **D5**: spatial support. The loss probability is the maximum over
  smoothing scales of the calibrated probability of the smoothed z.
- **D6**: D5 times ``1 - P_gain`` of the next date, so a loss that rebounds
  (a CHM failure that recovers) is suppressed. The last date has no next
  date and keeps its D5 probability.
- The ``s`` variants require the cell's own lfdr probability to pass
  ``p_boundary``. The plain variants only require it to exceed the date's
  prior (``1 - p0``, floored at ``min_prior``), so smoothing cannot grow
  the footprint.

Every date ``t`` gets its own empirical null with ``seed = t``. Dates with
fewer than ``min_fit_cells`` monitored valid cells are skipped.

Inputs: quantized z and the monitored mask, ``(T, rows, cols)``.
Outputs: :class:`DecisionResult` with onset indices before and after the MMU.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from csdv_core.change import di_decide
from csdv_core.change.di_decide import LfdrModel
from csdv_core.config._models import DisturbanceConfig
from csdv_core.disturbance.inputs import M2_PER_HA
from csdv_core.disturbance.standardize import sigmas_in_cells

logger = logging.getLogger(__name__)

#: Onset index of a cell without detection.
NO_ONSET = -1
#: ``onset_year`` raster value of a monitored cell without detection.
ONSET_YEAR_NONE = 0
#: ``onset_year`` raster value of a cell never monitored (never forest at t-1).
ONSET_YEAR_UNMONITORED = -1
#: ``DateFit.method`` of a skipped date.
SKIPPED_METHOD = "skipped"
#: 8-connected structuring element for objects.
EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class DateFit:
    """Empirical null fitted to one date's monitored z.

    Attributes:
        year: NAIP year of the date (the later image of the interval).
        n_cells: Monitored cells with finite z on that date.
        mu0: Null centre (NaN when skipped).
        sigma0: Null standard deviation (NaN when skipped).
        p0: Share of cells following the null, in (0, 1] (NaN when skipped).
        method: ``"central"`` (Efron central matching), ``"iqr"`` (median/IQR
            fallback) or :data:`SKIPPED_METHOD`.
        skipped: True when ``n_cells < decide.min_fit_cells`` (always True at
            index 0, which is not monitored). No cell is detected on a skipped
            date.
    """

    year: int
    n_cells: int
    mu0: float
    sigma0: float
    p0: float
    method: str
    skipped: bool


@dataclass(frozen=True)
class DecisionResult:
    """Outcome of one decision rule on a z stack.

    Attributes:
        rule: Rule name (``D1``, ``D2``, ``D5``, ``D5s``, ``D6``, ``D6s``).
        onset_index: int8 ``(rows, cols)``, index ``t`` of the first date with
            a detection, :data:`NO_ONSET` (-1) where none. Before the MMU.
        onset_index_mmu: int8 ``(rows, cols)``, ``onset_index`` with objects
            smaller than ``mmu.mmu_ha`` set to -1.
        probability: float32 ``(T, rows, cols)`` decision probability (D2:
            lfdr posterior, D5/D5s: spatial loss probability, D6/D6s: with
            persistence). NaN where a date was skipped or the cell is not
            monitored on that date (index 0 is all NaN). None for D1.
        fits: One :class:`DateFit` per date, length ``T``.
    """

    rule: str
    onset_index: np.ndarray
    onset_index_mmu: np.ndarray
    probability: np.ndarray | None
    fits: list[DateFit]


def skipped_fit(year: int, n_cells: int) -> DateFit:
    """Return the :class:`DateFit` of a date without a null fit.

    Args:
        year: NAIP year of the date.
        n_cells: Monitored cells with finite z on that date.

    Returns:
        A fit with NaN parameters and method :data:`SKIPPED_METHOD`.
    """
    return DateFit(
        year=year,
        n_cells=n_cells,
        mu0=float("nan"),
        sigma0=float("nan"),
        p0=float("nan"),
        method=SKIPPED_METHOD,
        skipped=True,
    )


def fit_date_models(
    z: np.ndarray,
    monitored: np.ndarray,
    params: DisturbanceConfig,
    years: Sequence[int],
) -> tuple[list[LfdrModel | None], list[DateFit]]:
    """Fit the empirical null and lfdr curve of every date.

    Date ``t`` uses the cells ``monitored[t] & isfinite(z[t])`` and
    ``seed = t``. Dates with fewer than ``decide.min_fit_cells`` such cells
    are skipped.

    Args:
        z: float32 ``(T, rows, cols)`` quantized z.
        monitored: bool ``(T, rows, cols)``.
        params: Run parameters.
        years: NAIP years, length ``T``.

    Returns:
        ``(models, fits)``: one model per date (None when skipped) and one
        :class:`DateFit` per date.

    Raises:
        ValueError: If shapes or lengths disagree.
    """
    if z.shape != monitored.shape:
        raise ValueError(f"z {z.shape} and monitored {monitored.shape} differ")
    if len(years) != z.shape[0]:
        raise ValueError(f"{len(years)} years for {z.shape[0]} dates")
    decide_params = params.decide
    models: list[LfdrModel | None] = []
    fits: list[DateFit] = []
    for date_index, year in enumerate(years):
        finite = np.isfinite(z[date_index])
        cells = monitored[date_index] & finite
        cell_count = cells.sum()
        n_cells = int(cell_count)
        if n_cells < decide_params.min_fit_cells:
            models.append(None)
            date_fit = skipped_fit(int(year), n_cells)
            fits.append(date_fit)
            if date_index > 0:
                logger.warning(
                    "Skipping %s: %d monitored cells < %d",
                    year,
                    n_cells,
                    decide_params.min_fit_cells,
                )
            continue
        null = di_decide.fit_null(
            z[date_index],
            cells,
            pct0=decide_params.pct0,
            bin_w=decide_params.bin_w,
            seed=date_index,
        )
        model = di_decide.lfdr_model(
            z[date_index],
            cells,
            null=null,
            bin_w=decide_params.bin_w,
            smooth_bins=decide_params.smooth_bins,
            seed=date_index,
        )
        models.append(model)
        date_fit = DateFit(
            year=int(year),
            n_cells=n_cells,
            mu0=null.mu0,
            sigma0=null.sigma0,
            p0=null.p0,
            method=null.method,
            skipped=False,
        )
        fits.append(date_fit)
    return models, fits


def rule_d1(
    z: np.ndarray, monitored: np.ndarray, params: DisturbanceConfig
) -> np.ndarray:
    """D1 flags: ``monitored & (z <= -decide.baseline_z)``.

    Args:
        z: float32 ``(T, rows, cols)``.
        monitored: bool ``(T, rows, cols)``.
        params: Run parameters.

    Returns:
        bool ``(T, rows, cols)`` (NaN z is never flagged).
    """
    threshold = -params.decide.baseline_z
    below = z <= threshold
    return monitored & below


def rule_d2(
    z: np.ndarray,
    monitored: np.ndarray,
    models: Sequence[LfdrModel | None],
    params: DisturbanceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """D2 flags: ``monitored & (lfdr posterior > decide.p_boundary)``.

    Args:
        z: float32 ``(T, rows, cols)``.
        monitored: bool ``(T, rows, cols)``.
        models: From :func:`fit_date_models`.
        params: Run parameters.

    Returns:
        ``(flags, probability)``: bool and float32 ``(T, rows, cols)``.
        Probability is ``nan_to_num(model.prob(z[t]))``, 0 on skipped dates.
    """
    flags = np.zeros(z.shape, dtype=bool)
    probability = np.zeros(z.shape, dtype="float32")
    for date_index, model in enumerate(models):
        if model is None:
            continue
        posterior = model.prob(z[date_index])
        passes = posterior > params.decide.p_boundary
        flags[date_index] = monitored[date_index] & passes
        probability[date_index] = np.nan_to_num(posterior)
    return flags, probability


def spatial_loss_probability(
    z: np.ndarray,
    monitored: np.ndarray,
    models: Sequence[LfdrModel | None],
    params: DisturbanceConfig,
    cell_m: float,
) -> np.ndarray:
    """D5 probability: spatially supported loss per date.

    For each non-skipped date, :func:`csdv_core.change.di_decide.spatial_prob`
    on ``z[t]`` over ``monitored[t] & isfinite(z[t])`` with sigmas
    ``decide.spatial_sigma_m / cell_m`` cells, ``bin_w`` and ``smooth_bins``.

    Args:
        z: float32 ``(T, rows, cols)``.
        monitored: bool ``(T, rows, cols)``.
        models: From :func:`fit_date_models` (only used to find skipped dates).
        params: Run parameters.
        cell_m: Cell side (m).

    Returns:
        float32 ``(T, rows, cols)`` in [0, 1], 0 on skipped dates.
    """
    sigmas_cells = sigmas_in_cells(params, cell_m)
    loss_probability = np.zeros(z.shape, dtype="float32")
    for date_index, model in enumerate(models):
        if model is None:
            continue
        finite = np.isfinite(z[date_index])
        where = monitored[date_index] & finite
        loss_probability[date_index] = di_decide.spatial_prob(
            z[date_index],
            where,
            sigmas_cells,
            bin_w=params.decide.bin_w,
            smooth_bins=params.decide.smooth_bins,
        )
    return loss_probability


def persistence_probability(
    z: np.ndarray,
    monitored: np.ndarray,
    models: Sequence[LfdrModel | None],
    loss_probability: np.ndarray,
    params: DisturbanceConfig,
    cell_m: float,
) -> np.ndarray:
    """D6 probability: loss at ``t`` that does not rebound at ``t + 1``.

    ``out[t] = loss_probability[t] * (1 - P_gain[t + 1])``, where
    ``P_gain`` is the largest gain probability of the smoothed next-date z
    over the smoothing scales, computed on cells monitored at ``t``. All
    inputs are change layers, so a rebound is a gain at ``t + 1``. The last
    date, skipped dates and dates with fewer than ``decide.min_fit_cells``
    usable next-date cells keep ``loss_probability[t]``.

    Args:
        z: float32 ``(T, rows, cols)``.
        monitored: bool ``(T, rows, cols)``.
        models: From :func:`fit_date_models`.
        loss_probability: From :func:`spatial_loss_probability`.
        params: Run parameters.
        cell_m: Cell side (m).

    Returns:
        float32 ``(T, rows, cols)`` in [0, 1].
    """
    sigmas_cells = sigmas_in_cells(params, cell_m)
    probability = loss_probability.copy()
    date_count = z.shape[0]
    for date_index in range(date_count - 1):
        if models[date_index] is None:
            continue
        next_z = z[date_index + 1]
        next_finite = np.isfinite(next_z)
        where = monitored[date_index] & next_finite
        if where.sum() < params.decide.min_fit_cells:
            continue
        gain_probability = np.zeros(z.shape[1:], dtype="float32")
        for sigma_cells in sigmas_cells:
            smoothed = di_decide.smooth_z(next_z, where, sigma_cells)
            scale_gain = di_decide.gain_prob(
                smoothed,
                where,
                bin_w=params.decide.bin_w,
                smooth_bins=params.decide.smooth_bins,
            )
            gain_probability = np.fmax(gain_probability, scale_gain)
        no_rebound = 1.0 - gain_probability
        probability[date_index] = loss_probability[date_index] * no_rebound
    return probability


def cell_lean_mask(
    z: np.ndarray,
    models: Sequence[LfdrModel | None],
    params: DisturbanceConfig,
    *,
    strict: bool,
) -> np.ndarray:
    """Cells whose own evidence leans toward loss.

    ``model.prob(z[t]) > floor`` with ``floor = decide.p_boundary`` when
    ``strict`` else ``model.prior(decide.min_prior)``. False on skipped dates.

    Args:
        z: float32 ``(T, rows, cols)``.
        models: From :func:`fit_date_models`.
        params: Run parameters.
        strict: Use the decision boundary (D5s, D6s) instead of the prior.

    Returns:
        bool ``(T, rows, cols)``.
    """
    lean = np.zeros(z.shape, dtype=bool)
    for date_index, model in enumerate(models):
        if model is None:
            continue
        if strict:
            floor = params.decide.p_boundary
        else:
            floor = model.prior(params.decide.min_prior)
        posterior = model.prob(z[date_index])
        lean[date_index] = posterior > floor
    return lean


def rule_spatial(
    z: np.ndarray,
    monitored: np.ndarray,
    models: Sequence[LfdrModel | None],
    params: DisturbanceConfig,
    cell_m: float,
    *,
    persistence: bool,
    strict: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """D5, D5s, D6 and D6s flags.

    ``flags = monitored & cell_lean_mask(strict) & (probability >
    decide.p_boundary)`` with probability from
    :func:`spatial_loss_probability`, passed through
    :func:`persistence_probability` when ``persistence`` is True.

    Args:
        z: float32 ``(T, rows, cols)``.
        monitored: bool ``(T, rows, cols)``.
        models: From :func:`fit_date_models`.
        params: Run parameters.
        cell_m: Cell side (m).
        persistence: D6 (True) or D5 (False).
        strict: The ``s`` variant.

    Returns:
        ``(flags, probability)``: bool and float32 ``(T, rows, cols)``.
    """
    probability = spatial_loss_probability(z, monitored, models, params, cell_m)
    if persistence:
        probability = persistence_probability(
            z, monitored, models, probability, params, cell_m
        )
    lean = cell_lean_mask(z, models, params, strict=strict)
    supported = probability > params.decide.p_boundary
    flags = monitored & lean & supported
    return flags, probability


def first_onset(flags: np.ndarray) -> np.ndarray:
    """Index of the first flagged date per cell.

    Args:
        flags: bool ``(T, rows, cols)``.

    Returns:
        int8 ``(rows, cols)``, :data:`NO_ONSET` where no date is flagged.
    """
    flagged_any = flags.any(axis=0)
    first_flagged = flags.argmax(axis=0)
    onset = np.where(flagged_any, first_flagged, NO_ONSET)
    return onset.astype("int8")


def apply_mmu(onset_index: np.ndarray, cell_ha: float, mmu_ha: float) -> np.ndarray:
    """Remove objects smaller than the minimum mapping unit.

    An object is a set of 8-connected cells sharing one onset index. Objects
    with fewer than ``ceil(mmu_ha / cell_ha - 1e-9)`` cells become
    :data:`NO_ONSET`. ``mmu_ha = 0`` keeps everything.

    Args:
        onset_index: int8 ``(rows, cols)`` from :func:`first_onset`.
        cell_ha: Cell area (ha).
        mmu_ha: Minimum mapping unit (ha).

    Returns:
        int8 ``(rows, cols)``, a filtered copy.
    """
    filtered = onset_index.copy()
    min_cells_float = np.ceil(mmu_ha / cell_ha - 1e-9)
    min_cells = int(min_cells_float)
    detected_values = onset_index[onset_index >= 0]
    onset_values = np.unique(detected_values)
    for onset_value in onset_values:
        same_onset = onset_index == onset_value
        labels, _ = ndimage.label(same_onset, EIGHT_CONNECTED)
        flat_labels = labels.ravel()
        object_sizes = np.bincount(flat_labels)
        small_objects = object_sizes < min_cells
        small_objects[0] = False
        small_cells = small_objects[labels]
        filtered[small_cells] = NO_ONSET
    return filtered


def mask_probability(
    probability: np.ndarray, monitored: np.ndarray, fits: Sequence[DateFit]
) -> np.ndarray:
    """Set the probability to NaN where no decision was made.

    A cell gets NaN on a date where it is not monitored or where the date was
    skipped (:attr:`DateFit.skipped`, always true at index 0). The flags are
    computed before this step, so the decisions do not change.

    Args:
        probability: float32 ``(T, rows, cols)`` from a rule.
        monitored: bool ``(T, rows, cols)``.
        fits: One :class:`DateFit` per date.

    Returns:
        float32 ``(T, rows, cols)``, a masked copy.
    """
    skipped_dates = np.array([date_fit.skipped for date_fit in fits], dtype=bool)
    skipped_stack = skipped_dates[:, np.newaxis, np.newaxis]
    undecided = ~monitored | skipped_stack
    masked = probability.astype("float32", copy=True)
    masked[undecided] = np.nan
    return masked


def decide(
    z: np.ndarray,
    monitored: np.ndarray,
    params: DisturbanceConfig,
    cell_m: float,
    *,
    years: Sequence[int],
) -> DecisionResult:
    """Run the configured rule (``decide.rule``), onset and MMU.

    Null models are fitted for every rule, D1 included, so the report always
    lists the per-date fits. For D1 the fits do not affect the decision and
    a skipped date is still evaluated.

    Args:
        z: float32 ``(T, rows, cols)`` quantized z.
        monitored: bool ``(T, rows, cols)`` from
            :func:`csdv_core.disturbance.layers.monitored_mask`.
        params: Run parameters (``decide`` and ``mmu`` sections).
        cell_m: Cell side (m).
        years: NAIP years, length ``T`` (labels the fits).

    Returns:
        The decision result.

    Raises:
        ValueError: If shapes or lengths disagree, or the rule is unknown.
    """
    rule = params.decide.rule
    models, fits = fit_date_models(z, monitored, params, years)
    probability: np.ndarray | None = None
    if rule == "D1":
        flags = rule_d1(z, monitored, params)
    elif rule == "D2":
        flags, probability = rule_d2(z, monitored, models, params)
    elif rule in ("D5", "D5s", "D6", "D6s"):
        flags, probability = rule_spatial(
            z,
            monitored,
            models,
            params,
            cell_m,
            persistence=rule.startswith("D6"),
            strict=rule.endswith("s"),
        )
    else:
        raise ValueError(f"Unknown decision rule {rule!r}")
    if probability is not None:
        probability = mask_probability(probability, monitored, fits)
    onset_index = first_onset(flags)
    cell_ha = cell_m**2 / M2_PER_HA
    onset_index_mmu = apply_mmu(onset_index, cell_ha, params.mmu.mmu_ha)
    detected_mask = onset_index >= 0
    kept_mask = onset_index_mmu >= 0
    detected_cells = int(detected_mask.sum())
    kept_cells = int(kept_mask.sum())
    logger.info(
        "Rule %s: %d cells detected, %d after the MMU", rule, detected_cells, kept_cells
    )
    return DecisionResult(
        rule=rule,
        onset_index=onset_index,
        onset_index_mmu=onset_index_mmu,
        probability=probability,
        fits=fits,
    )


def onset_index_to_year(
    onset_index: np.ndarray, years: Sequence[int], monitored: np.ndarray
) -> np.ndarray:
    """Convert onset indices to the ``onset_year`` raster convention.

    Args:
        onset_index: int8 ``(rows, cols)``, -1 = none.
        years: NAIP years, length ``T``.
        monitored: bool ``(T, rows, cols)``.

    Returns:
        int16 ``(rows, cols)``: ``years[onset_index]`` (the later image of the
        interval) where detected, :data:`ONSET_YEAR_NONE` (0) where monitored
        on some date but not detected, :data:`ONSET_YEAR_UNMONITORED` (-1)
        where never monitored.
    """
    later_monitored = monitored[1:]
    ever_monitored = later_monitored.any(axis=0)
    onset_year = np.full(onset_index.shape, ONSET_YEAR_NONE, dtype="int16")
    onset_year[~ever_monitored] = ONSET_YEAR_UNMONITORED
    detected = onset_index >= 0
    year_array = np.asarray(years, dtype="int16")
    detected_index = onset_index[detected]
    onset_year[detected] = year_array[detected_index]
    return onset_year


__all__ = [
    "NO_ONSET",
    "ONSET_YEAR_NONE",
    "ONSET_YEAR_UNMONITORED",
    "DateFit",
    "DecisionResult",
    "apply_mmu",
    "cell_lean_mask",
    "decide",
    "first_onset",
    "fit_date_models",
    "mask_probability",
    "onset_index_to_year",
    "persistence_probability",
    "rule_d1",
    "rule_d2",
    "rule_spatial",
    "skipped_fit",
    "spatial_loss_probability",
]
