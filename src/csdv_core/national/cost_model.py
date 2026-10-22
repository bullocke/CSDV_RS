"""csdv_core.national.cost_model — national compute, crown and storage totals from benchmarks.

Three pieces connect benchmark timings to CONUS totals.

1. Per-block cost. Segmentation time per block is modelled as a fixed cost
   plus a cost per canopy pixel, ``t = a + b * canopy_mpx``, fitted per engine
   on benchmark blocks. Crowns per block are modelled as proportional to
   canopy pixels.
2. Census calibration. The overview census reads canopy fractions at about
   19 m, which differ from 0.6 m fractions. A line through the benchmark tiles
   maps overview ``frac_ge2`` to full-resolution ``frac_ge2``.
3. Expansion. The census tiles are a simple random sample of the inventory, so
   a national total is the sample sum times ``N / n``. A bootstrap over sample
   tiles gives the interval.

All functions are pure. Inputs are DataFrames from the benchmark and census
tables. Outputs are dataclasses and DataFrames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "CensusCalibration",
    "LinearCost",
    "NationalTotal",
    "expand_total",
    "fit_block_cost",
    "fit_census_calibration",
    "fit_ratio",
    "predict_tiles",
    "wall_days",
]


@dataclass(frozen=True)
class LinearCost:
    """``seconds = intercept_s + per_canopy_mpx_s * canopy_mpx`` for one block.

    Attributes:
        engine: Engine the fit belongs to.
        intercept_s: Fixed seconds per block, whatever its canopy.
        per_canopy_mpx_s: Seconds per million canopy pixels.
        r2: Coefficient of determination on the fitted blocks.
        n: Blocks used.
        block_px: Interior block side the blocks were run at.
    """

    engine: str
    intercept_s: float
    per_canopy_mpx_s: float
    r2: float
    n: int
    block_px: int

    def seconds(self, n_blocks: np.ndarray, canopy_px: np.ndarray) -> np.ndarray:
        """Predicted seconds for tiles of ``n_blocks`` blocks and ``canopy_px`` canopy."""
        return self.intercept_s * np.asarray(n_blocks) + self.per_canopy_mpx_s * (
            np.asarray(canopy_px, dtype="float64") / 1e6
        )


@dataclass(frozen=True)
class CensusCalibration:
    """Full-resolution canopy fraction as a linear function of the overview fraction."""

    intercept: float
    slope: float
    r2: float
    n: int

    def __call__(self, overview_frac: np.ndarray) -> np.ndarray:
        x = np.asarray(overview_frac, dtype="float64")
        return np.clip(self.intercept + self.slope * x, 0.0, 1.0)


@dataclass(frozen=True)
class NationalTotal:
    """A national total with a bootstrap 95 percent interval."""

    name: str
    total: float
    lo: float
    hi: float
    n_sample: int
    n_population: int


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else float("nan")
    return float(coef[0]), float(coef[1]), r2


def fit_block_cost(
    blocks: pd.DataFrame, engine: str, *, time_col: str = "compute_s"
) -> LinearCost:
    """Fit ``seconds = a + b * canopy_mpx`` to one engine's benchmark blocks.

    Blocks with errors or without canopy are excluded from the fit, because a
    block without canopy is skipped before either engine runs.

    Args:
        blocks: Rows of :class:`~csdv_core.national.benchmark.BlockResult`.
        engine: Engine to fit.
        time_col: ``compute_s`` (stage sum) or ``wall_s``.
    """
    sub = blocks[(blocks["engine"] == engine) & (blocks["canopy_px"] > 0)]
    if "error" in sub:
        sub = sub[sub["error"].fillna("") == ""]
    if len(sub) < 2:
        raise ValueError(f"Need at least two {engine} blocks, got {len(sub)}")
    a, b, r2 = _ols(sub["canopy_px"] / 1e6, sub[time_col])
    block_px = int(sub["block_px"].iloc[0]) if "block_px" in sub else 0
    return LinearCost(engine, max(a, 0.0), b, r2, len(sub), block_px)


def fit_ratio(y: pd.Series, x: pd.Series) -> float:
    """Least-squares slope of ``y`` on ``x`` through the origin."""
    xv = np.asarray(x, dtype="float64")
    yv = np.asarray(y, dtype="float64")
    ok = np.isfinite(xv) & np.isfinite(yv)
    return float(np.sum(xv[ok] * yv[ok]) / np.sum(xv[ok] ** 2))


def fit_census_calibration(
    tiles: pd.DataFrame,
    *,
    overview_col: str = "frac_ge2",
    full_col: str = "frac_ge2_full",
) -> CensusCalibration:
    """Map overview canopy fraction to full-resolution canopy fraction."""
    ok = tiles[[overview_col, full_col]].dropna()
    a, b, r2 = _ols(ok[overview_col], ok[full_col])
    return CensusCalibration(a, b, r2, len(ok))


def predict_tiles(
    census: pd.DataFrame,
    *,
    cost: dict[str, LinearCost],
    calibration: CensusCalibration,
    crowns_per_canopy_mpx: float | np.ndarray,
) -> pd.DataFrame:
    """Predict seconds per engine, canopy pixels and crowns for each censused tile.

    Args:
        census: Census rows with ``full_width``, ``full_height``,
            ``valid_frac`` and ``frac_ge2``.
        cost: Block cost per engine. Each engine's own block size sets the
            number of blocks per tile.
        calibration: Overview to full-resolution canopy mapping.
        crowns_per_canopy_mpx: Crowns per million canopy pixels, one value or
            one per census row (for example by ecoregion).

    Returns:
        ``census`` with ``canopy_px``, ``n_crowns`` and ``<engine>_s`` added.
    """
    out = census.copy()
    full_px = out["full_width"].astype("float64") * out["full_height"]
    out["canopy_px"] = full_px * out["valid_frac"] * calibration(out["frac_ge2"])
    ratio = np.asarray(crowns_per_canopy_mpx, dtype="float64")
    out["n_crowns"] = ratio * out["canopy_px"].to_numpy() / 1e6
    for engine, model in cost.items():
        side = model.block_px or 2048
        n_blocks = np.ceil(out["full_width"] / side) * np.ceil(
            out["full_height"] / side
        )
        # Every block pays the fixed cost. A block with no canopy is in fact
        # skipped, so this errs high, by at most the intercept times the
        # number of empty blocks.
        out[f"{engine}_s"] = model.seconds(n_blocks, out["canopy_px"])
    return out


def expand_total(
    values: pd.Series,
    n_population: int,
    *,
    name: str = "",
    n_boot: int = 2000,
    seed: int = 0,
) -> NationalTotal:
    """Expand a simple-random-sample sum to the population, with a bootstrap CI."""
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    n = len(v)
    scale = n_population / n
    rng = np.random.default_rng(seed)
    boots = np.array([v[rng.integers(0, n, n)].sum() * scale for _ in range(n_boot)])
    return NationalTotal(
        name=name,
        total=float(v.sum() * scale),
        lo=float(np.percentile(boots, 2.5)),
        hi=float(np.percentile(boots, 97.5)),
        n_sample=n,
        n_population=int(n_population),
    )


def wall_days(core_hours: float, cores: int, *, efficiency: float = 0.85) -> float:
    """Wall-clock days to spend ``core_hours`` on ``cores`` at an occupancy ``efficiency``."""
    if cores <= 0 or efficiency <= 0:
        return math.inf
    return core_hours / (cores * efficiency) / 24.0
