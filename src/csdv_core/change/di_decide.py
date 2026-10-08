"""Decisions on Disturbance Index (DI) images without fixed z thresholds.

Three steps turn a DI image into calibrated evidence:

1. **Empirical null** (Efron 2004, central matching). A normal curve is fitted
   to the centre of the forest DI histogram (between its quartiles by
   default), so each image defines its own "undisturbed" distribution. The
   fit also estimates ``p0``, the share of forest pixels that follow it.
2. **Local false discovery rate**. ``lfdr(z) = p0 f0(z) / f(z)`` is the
   posterior probability that a pixel with DI ``z`` belongs to the null, with
   ``f`` the smoothed histogram of all forest pixels. ``1 - lfdr`` is the
   probability of disturbance. The test is one-sided (losses only), and lfdr
   is forced to rise monotonically toward the null centre.
3. **Spatial and temporal pooling**. Connected pixels with probability above
   one half form clusters, and each cluster's mass is compared with the
   masses of clusters formed by the mirrored image (``2 mu0 - z``). These
   share the image's spatial autocorrelation but cannot contain losses, which
   gives each cluster a false discovery rate. Across dates, per-pixel
   log-likelihood ratios are accumulated in a CUSUM-style recursion so that
   weak but repeated evidence builds up and a normal observation resets it.

Decisions use the Bayes boundary P > 0.5 (more likely disturbed than not)
rather than a tuned threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.special import expit, logit
from scipy.stats import norm

logger = logging.getLogger(__name__)

#: lfdr is clipped to this range before forming log-likelihood ratios.
LFDR_CLIP = (1e-4, 1 - 1e-4)
EIGHT = np.ones((3, 3), bool)


@dataclass(frozen=True)
class EmpiricalNull:
    """Normal null ``N(mu0, sigma0)`` holding a share ``p0`` of the population."""

    mu0: float
    sigma0: float
    p0: float
    method: str


@dataclass
class LfdrModel:
    """One-sided local fdr curve on a z grid for one image."""

    null: EmpiricalNull
    centres: np.ndarray
    lfdr_curve: np.ndarray

    def lfdr(self, z: np.ndarray) -> np.ndarray:
        """Local fdr at ``z``; NaN stays NaN, and z at or above mu0 gives 1."""
        out = np.interp(z, self.centres, self.lfdr_curve).astype("float32")
        out[z >= self.null.mu0] = 1.0
        out[~np.isfinite(z)] = np.nan
        return out

    def prob(self, z: np.ndarray) -> np.ndarray:
        """Posterior probability of disturbance, ``1 - lfdr``."""
        return 1.0 - self.lfdr(z)

    def prior(self, min_prior: float = 0.0) -> float:
        """Prior probability of disturbance, ``1 - p0``, floored at ``min_prior``."""
        return float(np.clip(max(1.0 - self.null.p0, min_prior), *LFDR_CLIP))

    def llr(self, z: np.ndarray, min_prior: float = 0.0) -> np.ndarray:
        """Log-likelihood ratio ``log f1/f0``, i.e. posterior minus prior log-odds.

        When the null fit absorbs the whole population (``p0 = 1``, common for
        skewed layers), ``1 - p0`` is zero and a normal observation would carry
        no evidence. ``min_prior`` floors the prior so that it does. NaN
        becomes 0 (no evidence either way).
        """
        lf = np.clip(self.lfdr(z), *LFDR_CLIP)
        out = logit(1.0 - lf) - logit(self.prior(min_prior))
        return np.nan_to_num(out, nan=0.0).astype("float32")


def _sample(z: np.ndarray, where: np.ndarray | None, n_max: int, seed: int):
    vals = z[np.isfinite(z) if where is None else where & np.isfinite(z)]
    if vals.size > n_max:
        vals = np.random.default_rng(seed).choice(vals, n_max, replace=False)
    return vals.astype("float64")


def fit_null(
    z: np.ndarray,
    where: np.ndarray | None = None,
    *,
    pct0: float = 0.25,
    bin_w: float = 0.05,
    n_max: int = 2_000_000,
    seed: int = 0,
) -> EmpiricalNull:
    """Efron's central-matching estimate of the empirical null.

    A quadratic is fitted to the log histogram between the ``pct0`` and
    ``1 - pct0`` quantiles (Efron's ``locfdr`` default ``pct0 = 1/4``). Falls
    back to the median and IQR / 1.349 with ``p0 = 1`` if the fit is not
    concave.
    """
    vals = _sample(z, where, n_max, seed)
    if vals.size < 100:
        raise ValueError(f"Too few values to fit a null ({vals.size})")
    lo, hi = np.quantile(vals, [pct0, 1 - pct0])
    edges = np.arange(lo, hi + bin_w, bin_w)
    counts, edges = np.histogram(vals, edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    ok = counts > 0
    if ok.sum() >= 5:
        c2, c1, c0 = np.polyfit(
            centres[ok], np.log(counts[ok]), 2, w=np.sqrt(counts[ok])
        )
        if c2 < 0:
            sigma0 = float(np.sqrt(-1.0 / (2.0 * c2)))
            mu0 = float(-c1 / (2.0 * c2))
            log_p0 = (
                c0
                + mu0**2 / (2 * sigma0**2)
                + np.log(np.sqrt(2 * np.pi) * sigma0 / (vals.size * bin_w))
            )
            p0 = float(min(1.0, np.exp(log_p0)))
            return EmpiricalNull(mu0, sigma0, p0, "central")
    med = float(np.median(vals))
    q1, q3 = np.quantile(vals, [0.25, 0.75])
    sigma = float((q3 - q1) / 1.349)
    if not sigma > 0:  # saturated layer: half the values share one level
        sigma = float(vals.std())
    logger.debug("Central matching failed; using median/IQR null")
    return EmpiricalNull(med, sigma, 1.0, "iqr")


def lfdr_model(
    z: np.ndarray,
    where: np.ndarray | None = None,
    *,
    null: EmpiricalNull | None = None,
    bin_w: float = 0.05,
    smooth_bins: float = 2.0,
    n_max: int = 2_000_000,
    seed: int = 0,
) -> LfdrModel:
    """Fit the null (unless given) and the one-sided lfdr curve for losses."""
    vals = _sample(z, where, n_max, seed)
    null = null or fit_null(z, where, bin_w=bin_w, n_max=n_max, seed=seed)
    lo = min(float(vals.min()), null.mu0 - 8 * null.sigma0)
    edges = np.arange(lo - bin_w, null.mu0 + bin_w, bin_w)
    counts, edges = np.histogram(vals, edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    f = ndimage.gaussian_filter1d(counts.astype(float), smooth_bins, mode="nearest")
    f0 = vals.size * bin_w * null.p0 * norm.pdf(centres, null.mu0, null.sigma0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lf = np.where(f > 0, f0 / f, 1.0)
    lf = np.clip(lf, 0.0, 1.0)
    # Monotone: moving away from the null centre lfdr can only fall.
    lf = np.minimum.accumulate(lf[::-1])[::-1]
    return LfdrModel(null, centres, lf)


def bh_reject(
    z: np.ndarray, null: EmpiricalNull, where: np.ndarray, *, q: float = 0.05
) -> np.ndarray:
    """Benjamini-Hochberg on one-sided p-values from the empirical null."""
    out = np.zeros(z.shape, bool)
    m = where & np.isfinite(z)
    p = norm.cdf(z[m], null.mu0, null.sigma0)
    order = np.argsort(p)
    k = np.arange(1, p.size + 1)
    passed = p[order] <= q * k / p.size
    if passed.any():
        cut = p[order][np.flatnonzero(passed).max()]
        out[m] = p <= cut
    return out


# ----------------------------------------------------------------------------
# Spatial support and persistence
# ----------------------------------------------------------------------------


def smooth_z(z: np.ndarray, where: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian-weighted mean of ``z`` over valid cells (normalized convolution).

    Smoothing averages speckle down while a patch wider than ``sigma_px``
    keeps most of its depth. The result is not re-scaled here: the empirical
    null fitted to the smoothed field absorbs the smaller noise SD, so a
    coherent patch ends up many null SDs from the centre and a lone cell does
    not. Cells outside ``where`` are NaN.
    """
    w = (where & np.isfinite(z)).astype("float32")
    x = np.where(w > 0, z, 0.0).astype("float32")
    num = ndimage.gaussian_filter(x, sigma_px, mode="constant")
    den = ndimage.gaussian_filter(w, sigma_px, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~(w > 0)] = np.nan
    return out.astype("float32")


def spatial_prob(
    z: np.ndarray,
    where: np.ndarray,
    sigmas_px: list[float],
    **lfdr_kw,
) -> np.ndarray:
    """Loss probability with spatial support: max over smoothing scales.

    Each scale gets its own empirical null and lfdr, so every map is
    calibrated before the maximum is taken.
    """
    best = np.zeros(z.shape, "float32")
    for s in sigmas_px:
        zs = smooth_z(z, where, s)
        m = lfdr_model(zs, where, **lfdr_kw)
        best = np.fmax(best, np.nan_to_num(m.prob(zs)))
    return best


def gain_prob(z: np.ndarray, where: np.ndarray, **lfdr_kw) -> np.ndarray:
    """Probability that ``z`` is a real gain (upper tail), via ``-z``."""
    m = lfdr_model(-z, where, **lfdr_kw)
    return np.nan_to_num(m.prob(-z)).astype("float32")


# ----------------------------------------------------------------------------
# Clusters
# ----------------------------------------------------------------------------


@dataclass
class ClusterResult:
    """Clusters of one image and their probability of being real change."""

    labels: np.ndarray  # int32, 0 = none
    mass: np.ndarray  # (n + 1,) mass per label, index 0 unused
    prob: np.ndarray  # (n + 1,) 1 - cluster q-value, index 0 = 0
    n_null: int

    def prob_map(self) -> np.ndarray:
        return self.prob[self.labels].astype("float32")


def _clusters(z, model: LfdrModel, where, p_form: float):
    flag = where & (model.prob(z) > p_form)
    labels, n = ndimage.label(flag, EIGHT)
    weight = np.maximum(0.0, (model.null.mu0 - np.nan_to_num(z, nan=model.null.mu0)))
    weight /= model.null.sigma0
    mass = ndimage.sum(weight, labels, np.arange(1, n + 1)) if n else np.zeros(0)
    return labels.astype("int32"), np.asarray(mass, float)


def cluster_probability(
    z: np.ndarray,
    model: LfdrModel,
    where: np.ndarray,
    *,
    p_form: float = 0.5,
) -> ClusterResult:
    """Clusters of P > ``p_form`` pixels, scored against a mirrored null.

    Mass is the summed null-standardized drop ``(mu0 - z) / sigma0``. The
    mirrored image ``2 mu0 - z`` turns gains into pseudo-losses that share the
    image's texture and autocorrelation. For a cluster of mass ``m``, the
    estimated FDR is ``N_mirror(>= m) / N_obs(>= m)``. Its q-value (the
    smallest FDR over thresholds that would include it) gives ``prob = 1 - q``.
    """
    labels, mass = _clusters(z, model, where, p_form)
    _, null_mass = _clusters(2 * model.null.mu0 - z, model, where, p_form)
    prob = np.zeros(mass.size + 1, "float32")
    if mass.size:
        obs_sorted = np.sort(mass)
        null_sorted = np.sort(null_mass)
        n_obs = mass.size - np.searchsorted(obs_sorted, mass, side="left")
        n_null = null_mass.size - np.searchsorted(null_sorted, mass, side="left")
        fdr = np.minimum(1.0, n_null / np.maximum(n_obs, 1))
        # q-value: minimum FDR over thresholds at or below this mass.
        order = np.argsort(mass)
        q = np.minimum.accumulate(fdr[order])
        qv = np.empty_like(q)
        qv[order] = q
        prob[1:] = 1.0 - qv
    return ClusterResult(labels, np.concatenate([[0.0], mass]), prob, null_mass.size)


# ----------------------------------------------------------------------------
# Evidence over time
# ----------------------------------------------------------------------------


@dataclass
class AccumResult:
    """Per-pixel accumulated evidence over dates."""

    prob: np.ndarray  # (T, H, W) float32 posterior after each date
    onset: np.ndarray  # (T, H, W) int8 index of the run start, -1 none

    def first(self, monitored: np.ndarray, p_detect: float = 0.5) -> np.ndarray:
        """Onset index of the first detection (``prob > p_detect`` while the
        onset date was monitored), -1 for none. ``monitored`` is (T, H, W)."""
        t_count = self.prob.shape[0]
        out = np.full(self.prob.shape[1:], -1, "int16")
        for t in range(t_count):
            on = self.onset[t]
            ok_on = np.zeros(on.shape, bool)
            valid = on >= 0
            idx = np.where(valid, on, 0)
            ok_on[valid] = np.take_along_axis(monitored, idx[None], 0)[0][valid]
            hit = (out < 0) & ok_on & (self.prob[t] > p_detect)
            out[hit] = on[hit]
        return out


def accumulate(llr: np.ndarray, *, prior: float, decay: float = 1.0) -> AccumResult:
    """CUSUM-style accumulation of log-likelihood ratios over dates.

    ``S_t = max(0, decay * S_{t-1}) + llr_t`` and ``P_t = expit(logit(prior)
    + S_t)``. A run starts where ``S_{t-1} <= 0``. With one date this reduces
    to ``1 - lfdr``. ``decay < 1`` lets old evidence fade, which suits NDVI
    because it recovers within one NAIP revisit.
    """
    t_count = llr.shape[0]
    s = np.zeros(llr.shape[1:], "float32")
    start = np.full(llr.shape[1:], -1, "int8")
    prob = np.empty(llr.shape, "float32")
    onset = np.full(llr.shape, -1, "int8")
    base = float(logit(np.clip(prior, *LFDR_CLIP)))
    for t in range(t_count):
        carry = np.maximum(0.0, decay * s)
        start[carry <= 0] = t
        s = carry + llr[t]
        prob[t] = expit(base + s)
        onset[t] = np.where(s > 0, start, -1)
    return AccumResult(prob, onset)


def confirm_next(llr: np.ndarray, *, prior: float) -> np.ndarray:
    """Posterior for a change at date t given dates t and t + 1.

    ``P_t = expit(logit(prior) + llr_t + llr_{t+1})``. The last date has no
    follow-up and uses ``llr_t`` alone. Meant for CHM state layers, where a
    real loss persists to the next NAIP date and a model artefact does not.
    """
    base = float(logit(np.clip(prior, *LFDR_CLIP)))
    nxt = np.concatenate([llr[1:], np.zeros_like(llr[:1])], axis=0)
    return expit(base + llr + nxt).astype("float32")
