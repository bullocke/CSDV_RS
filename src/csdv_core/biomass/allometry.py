"""csdv_core.biomass.allometry — tree AGB and stem diameter from height and crown diameter.

Implements the remote-sensing allometries of Jucker et al. (2017, Global
Change Biology 23:177-190, doi:10.1111/gcb.13388), which predict a tree's
aboveground biomass directly from the compound variable height times crown
diameter, without first predicting stem diameter.

AGB (Eqn 7)::

    AGB = (0.016 + a_G) * (H * CD) ** (2.013 + b_G) * exp(0.204**2 / 2)

with ``a_G = 0.093`` and ``b_G = -0.223`` for gymnosperms and both zero for
angiosperms. H and CD are in metres and AGB in kg of oven-dry mass. The
exponent near 2 for angiosperms means AGB scales close to ``H**2 * CD**2``,
which is proportional to ``H**2`` times crown area, so two merged crowns of
equal height carry about the same summed AGB as the two crowns apart. That
property is what Gate D of the national feasibility study tests.

Global stem diameter (Eqn 6)::

    D = 0.557 * (H * CD) ** 0.809 * exp(0.056**2 / 2)

with D in cm. Group- and region-specific diameter equations are in the
paper's Appendix S4 and are not reproduced here.

Also implements the biome-wide Tallo models of the refAGB project
(:class:`TalloLoglog2`), fitted to the CONUS records of the Tallo database
(Jucker et al. 2022, Global Change Biology 28:5254-5268) with allodb AGB as the
response and NEON-origin records excluded::

    ln(AGB) = a + b * ln(H) + c * ln(CD)

with one fit per RESOLVE biome. The point prediction multiplies the
back-transformed value by an empirical smearing factor. The stored 3x3
covariance of ``(a, b, c)`` supports Monte Carlo draws of the coefficients,
which propagate the systematic error that every tree in a map shares.

Inputs: arrays of height and crown diameter in metres. Outputs: arrays of AGB
(kg) or diameter (cm).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

#: Angiosperm scaling constant and exponent of the AGB model.
JUCKER_AGB_ALPHA = 0.016
JUCKER_AGB_BETA = 2.013
#: Gymnosperm offsets added to the angiosperm constant and exponent.
JUCKER_AGB_ALPHA_GYMNO = 0.093
JUCKER_AGB_BETA_GYMNO = -0.223
#: Residual standard error of the AGB model on the log scale.
JUCKER_AGB_SIGMA = 0.204

JUCKER_D_ALPHA = 0.557
JUCKER_D_BETA = 0.809
JUCKER_D_SIGMA = 0.056

__all__ = [
    "jucker_agb_kg",
    "jucker_diameter_cm",
    "TalloLoglog2",
    "tallo_model",
    "tallo_biomes",
]


def jucker_agb_kg(
    height_m: ArrayLike,
    crown_diam_m: ArrayLike,
    *,
    gymnosperm: ArrayLike = False,
) -> np.ndarray:
    """Tree aboveground biomass from height and crown diameter.

    Args:
        height_m: Tree height in metres.
        crown_diam_m: Crown diameter in metres.
        gymnosperm: True for gymnosperm trees, either one value or one per
            tree. A fraction between 0 and 1 blends the two forms linearly,
            which suits a gridded softwood share when the type of a single
            crown is unknown.

    Returns:
        AGB in kg, NaN where height or crown diameter is missing or not
        positive.
    """
    h = np.asarray(height_m, dtype="float64")
    cd = np.asarray(crown_diam_m, dtype="float64")
    g = np.clip(np.asarray(gymnosperm, dtype="float64"), 0.0, 1.0)
    hcd = h * cd
    ok = np.isfinite(hcd) & (h > 0) & (cd > 0)
    safe = np.where(ok, hcd, 1.0)
    correction = np.exp(JUCKER_AGB_SIGMA**2 / 2.0)
    angio = JUCKER_AGB_ALPHA * safe**JUCKER_AGB_BETA
    gymno = (JUCKER_AGB_ALPHA + JUCKER_AGB_ALPHA_GYMNO) * safe ** (
        JUCKER_AGB_BETA + JUCKER_AGB_BETA_GYMNO
    )
    agb = ((1.0 - g) * angio + g * gymno) * correction
    return np.where(ok, agb, np.nan)


def jucker_diameter_cm(height_m: ArrayLike, crown_diam_m: ArrayLike) -> np.ndarray:
    """Stem diameter from height and crown diameter (global model, Eqn 6).

    Returns:
        Diameter in cm, NaN where either input is missing or not positive.
    """
    h = np.asarray(height_m, dtype="float64")
    cd = np.asarray(crown_diam_m, dtype="float64")
    hcd = h * cd
    ok = np.isfinite(hcd) & (h > 0) & (cd > 0)
    d = JUCKER_D_ALPHA * np.where(ok, hcd, 1.0) ** JUCKER_D_BETA
    return np.where(ok, d * np.exp(JUCKER_D_SIGMA**2 / 2.0), np.nan)


@dataclass(frozen=True)
class TalloLoglog2:
    """One biome's Tallo AGB model, ``ln(AGB_kg) = a + b ln(H_m) + c ln(CD_m)``.

    Attributes:
        biome: RESOLVE biome name.
        a: Intercept, uncentred.
        b: Height exponent.
        c: Crown diameter exponent. CD is a diameter, not a radius.
        sigma: Residual standard deviation on the log scale.
        smearing: Empirical smearing factor for the back-transformed mean.
        cov: 3x3 covariance of ``(a, b, c)``.
        n: Trees in the fit.
        r2_log: R² on the log scale.
    """

    biome: str
    a: float
    b: float
    c: float
    sigma: float
    smearing: float
    cov: np.ndarray
    n: int
    r2_log: float

    def agb_kg(
        self,
        height_m: ArrayLike,
        crown_diam_m: ArrayLike,
        *,
        coefs: np.ndarray | None = None,
        smear: bool = True,
    ) -> np.ndarray:
        """Tree AGB in kg. NaN where height or crown diameter is not positive.

        Args:
            height_m: Tree height in metres.
            crown_diam_m: Crown diameter in metres.
            coefs: Optional ``(a, b, c)`` replacing the fitted values, one row
                per draw, as from :meth:`draws`. The result then gains a
                leading draw axis.
            smear: Apply the smearing factor. Leave it on for a mean
                prediction. Turn it off when residual error is simulated
                separately, as refAGB's Monte Carlo does.
        """
        h = np.asarray(height_m, dtype="float64")
        cd = np.asarray(crown_diam_m, dtype="float64")
        ok = np.isfinite(h) & np.isfinite(cd) & (h > 0) & (cd > 0)
        lh = np.log(np.where(ok, h, 1.0))
        lcd = np.log(np.where(ok, cd, 1.0))
        scale = self.smearing if smear else 1.0
        if coefs is None:
            agb = scale * np.exp(self.a + self.b * lh + self.c * lcd)
            return np.where(ok, agb, np.nan)
        k = np.atleast_2d(np.asarray(coefs, dtype="float64"))
        shape = (k.shape[0],) + (1,) * lh.ndim
        lp = (
            k[:, 0].reshape(shape)
            + k[:, 1].reshape(shape) * lh
            + k[:, 2].reshape(shape) * lcd
        )
        return np.where(ok, scale * np.exp(lp), np.nan)

    def draws(self, n: int, *, seed: int | None = None) -> np.ndarray:
        """``n`` draws of ``(a, b, c)`` from their sampling distribution."""
        rng = np.random.default_rng(seed)
        mean = np.array([self.a, self.b, self.c])
        return rng.multivariate_normal(mean, self.cov, size=int(n))


@lru_cache(maxsize=1)
def _tallo_table() -> pd.DataFrame:
    ref = resources.files("csdv_core.biomass").joinpath("coefs/tallo_loglog2.csv")
    with resources.as_file(ref) as path:
        return pd.read_csv(path, comment="#").set_index("biome")


def tallo_biomes() -> tuple[str, ...]:
    """RESOLVE biomes with a Tallo AGB model."""
    return tuple(_tallo_table().index)


def tallo_model(biome: str) -> TalloLoglog2:
    """The Tallo loglog2 model for one RESOLVE biome.

    Raises:
        KeyError: The biome has no fit. Mediterranean is the notable gap among
            the biomes that contain NEON sites.
    """
    table = _tallo_table()
    if biome not in table.index:
        raise KeyError(f"No Tallo AGB model for {biome!r}; known: {tallo_biomes()}")
    r = table.loc[biome]
    cov = np.array(
        [
            [r["cov_aa"], r["cov_ab"], r["cov_ac"]],
            [r["cov_ab"], r["cov_bb"], r["cov_bc"]],
            [r["cov_ac"], r["cov_bc"], r["cov_cc"]],
        ],
        dtype="float64",
    )
    return TalloLoglog2(
        biome=biome,
        a=float(r["a"]),
        b=float(r["b"]),
        c=float(r["c"]),
        sigma=float(r["sigma"]),
        smearing=float(r["smearing"]),
        cov=cov,
        n=int(r["n"]),
        r2_log=float(r["r2_log"]),
    )
