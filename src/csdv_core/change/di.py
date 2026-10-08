"""Disturbance Index (DI): standardize an image against the forest population.

The DI expresses each pixel as a z-score against the forest pixels of the same
image, so a disturbance shows up as a strong negative departure regardless of
the image's radiometry, phenology or CHM model bias. It was developed for
Landsat and has since been used with Sentinel-1/2, PlanetScope and NISAR. This
module holds the standardization step only. The decision step (empirical null,
probabilities, clusters, accumulation) is in :mod:`csdv_core.change.di_decide`.

Four ways to standardize a layer ``x`` (``METHODS``):

- ``scene``: ``(x - centre_F) / scale_F`` over every forest pixel in the image.
  The robust default uses the median and 1.4826 * MAD, so the disturbances
  being sought barely move the statistics.
- ``local``: the same with the mean and SD of forest pixels in a square moving
  window, to absorb illumination stripes, seam lines and terrain gradients.
  Pixels that are already obvious anomalies against the scene are left out of
  the window statistics (one-step trimming).
- ``pixmean``: subtract each pixel's own mean over earlier forest dates, then
  apply ``scene``. Removes fixed spatial patterns (species, stand height, slope).
- ``pixz``: divide that anomaly by the pixel's own SD, shrunk toward the pooled
  forest SD because NAIP gives only a handful of earlier dates.

All functions work on in-memory 2-D arrays with NaN for missing values and a
boolean ``pop`` array marking the forest population.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import ndimage

from csdv_core.masks.forest import MAD_SCALE

logger = logging.getLogger(__name__)

Method = Literal["scene", "local", "pixmean", "pixz"]
METHODS: tuple[str, ...] = ("scene", "local", "pixmean", "pixz")
#: Fewest valid population pixels for image statistics.
MIN_POP = 200


@dataclass(frozen=True)
class PopStats:
    """Centre and scale of a layer over the forest population."""

    centre: float
    scale: float
    n: int


def population_stats(
    v: np.ndarray, pop: np.ndarray, *, robust: bool = True
) -> PopStats | None:
    """Median/scaled-MAD (``robust``) or mean/SD of ``v`` over ``pop``.

    A saturated layer (e.g. canopy cover, where most forest cells are 1.0) has
    a MAD of zero, and then the mean and SD are used instead. Returns None
    when fewer than ``MIN_POP`` valid pixels remain or the scale is zero.
    """
    vals = v[pop & np.isfinite(v)]
    if vals.size < MIN_POP:
        return None
    if robust:
        centre = float(np.median(vals))
        scale = float(MAD_SCALE * np.median(np.abs(vals - centre)))
        if scale > 0:
            return PopStats(centre, scale, int(vals.size))
        logger.debug("MAD is zero; falling back to mean and SD")
        centre, scale = float(vals.mean()), float(vals.std())
    else:
        centre, scale = float(vals.mean()), float(vals.std())
    if not scale > 0:
        return None
    return PopStats(centre, scale, int(vals.size))


def scene_z(v: np.ndarray, pop: np.ndarray, *, robust: bool = True) -> np.ndarray:
    """Standardize ``v`` with whole-image population statistics."""
    st = population_stats(v, pop, robust=robust)
    if st is None:
        return np.full(v.shape, np.nan, "float32")
    return ((v - st.centre) / st.scale).astype("float32")


def local_z(
    v: np.ndarray,
    pop: np.ndarray,
    size_px: int,
    *,
    min_frac: float = 0.2,
    trim_z: float | None = 3.0,
    sd_floor: float = 0.25,
) -> np.ndarray:
    """Standardize against forest pixels in a ``size_px`` square moving window.

    Window means and SDs come from box filters of the masked layer, normalized
    by the share of valid forest in the window. Population pixels with a robust
    scene z below ``-trim_z`` or above ``trim_z`` are left out of the window
    statistics so that a large disturbance does not inflate its own window SD.
    Windows with less than ``min_frac`` valid forest fall back to the scene
    statistics. The window SD is floored at ``sd_floor`` times the scene
    scale, because a window of saturated values (e.g. full canopy cover)
    otherwise has an SD near zero and turns any small change into a huge z.
    """
    st = population_stats(v, pop, robust=True)
    if st is None:
        return np.full(v.shape, np.nan, "float32")
    w = pop & np.isfinite(v)
    if trim_z is not None:
        w &= np.abs((np.where(w, v, st.centre) - st.centre) / st.scale) <= trim_z
    wf = w.astype("float32")
    x = np.where(w, v, 0.0).astype("float32")
    frac = ndimage.uniform_filter(wf, size_px, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        m = ndimage.uniform_filter(x, size_px, mode="constant") / frac
        m2 = ndimage.uniform_filter(x * x, size_px, mode="constant") / frac
    sd = np.sqrt(np.maximum(m2 - m * m, 0.0))
    sd = np.maximum(sd, sd_floor * st.scale)
    thin = (frac < min_frac) | ~(sd > 0)
    m[thin] = st.centre
    sd[thin] = st.scale
    return ((v - m) / sd).astype("float32")


class PixelHistory:
    """Running per-pixel count, sum and sum of squares over earlier dates."""

    def __init__(self, shape: tuple[int, int]):
        self.n = np.zeros(shape, "float32")
        self.s = np.zeros(shape, "float64")
        self.ss = np.zeros(shape, "float64")

    def add(self, x: np.ndarray, where: np.ndarray) -> None:
        """Add ``x`` at pixels in ``where`` (NaN pixels are skipped)."""
        m = where & np.isfinite(x)
        v = x[m].astype("float64")
        self.n[m] += 1
        self.s[m] += v
        self.ss[m] += v * v

    def mean(self, min_n: int = 1) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            out = self.s / self.n
        out[self.n < min_n] = np.nan
        return out.astype("float32")

    def var(self, min_n: int = 2) -> np.ndarray:
        """Sample variance (ddof = 1); NaN where fewer than ``min_n`` dates."""
        with np.errstate(invalid="ignore", divide="ignore"):
            v = (self.ss - self.s**2 / self.n) / (self.n - 1)
        v = np.maximum(v, 0.0)
        v[self.n < max(min_n, 2)] = np.nan
        return v.astype("float32")


def shrunk_sd(
    hist: PixelHistory, pop: np.ndarray, *, min_n: int, prior_df: float
) -> np.ndarray:
    """Per-pixel SD shrunk toward the pooled population variance.

    ``sd^2 = ((n - 1) s^2 + k s0^2) / (n - 1 + k)`` with ``k = prior_df`` and
    ``s0^2`` the median per-pixel variance over ``pop``. With three or four
    dates the raw SD is too noisy to divide by, and the prior keeps a pixel
    that happened to be stable from producing huge z-scores.
    """
    var = hist.var(min_n)
    pooled = var[pop & np.isfinite(var)]
    if pooled.size < MIN_POP:
        return np.full(var.shape, np.nan, "float32")
    s0 = float(np.median(pooled))
    df = np.maximum(hist.n - 1, 0)
    with np.errstate(invalid="ignore"):
        out = np.sqrt((df * var + prior_df * s0) / (df + prior_df))
    return out.astype("float32")


def standardize(
    x: np.ndarray,
    pop: np.ndarray,
    method: Method,
    *,
    hist: PixelHistory | None = None,
    robust: bool = True,
    size_px: int = 63,
    min_frac: float = 0.2,
    trim_z: float | None = 3.0,
    min_hist: int = 1,
    prior_df: float = 2.0,
) -> np.ndarray:
    """Standardize one image ``x`` against the forest population ``pop``.

    ``hist`` holds the pixel's earlier dates and is required for ``pixmean``
    (at least ``min_hist`` dates) and ``pixz`` (at least ``max(min_hist, 2)``).
    Pixels without enough history are NaN.
    """
    if method == "scene":
        return scene_z(x, pop, robust=robust)
    if method == "local":
        return local_z(x, pop, size_px, min_frac=min_frac, trim_z=trim_z)
    if hist is None:
        raise ValueError(f"method {method!r} needs a PixelHistory")
    if method == "pixmean":
        return scene_z(x - hist.mean(min_hist), pop, robust=robust)
    if method == "pixz":
        n_min = max(min_hist, 2)
        sd = shrunk_sd(hist, pop, min_n=n_min, prior_df=prior_df)
        with np.errstate(invalid="ignore", divide="ignore"):
            anom = (x - hist.mean(n_min)) / sd
        return scene_z(anom, pop, robust=robust)
    raise ValueError(f"Unknown method {method!r}")


def stouffer(zs: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Weighted Stouffer combination ``sum(w z) / sqrt(sum(w^2))``.

    Pixels where any input is NaN are NaN. The result is re-standardized later
    against its own population, so correlation between inputs only changes
    the scale, not the ranking.
    """
    w = np.ones(len(zs)) if weights is None else np.asarray(weights, float)
    num = sum(wi * z for wi, z in zip(w, zs, strict=True))
    return (num / np.sqrt((w**2).sum())).astype("float32")
