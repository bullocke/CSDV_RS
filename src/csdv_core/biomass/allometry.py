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

Inputs: arrays of height and crown diameter in metres. Outputs: arrays of AGB
(kg) or diameter (cm).
"""

from __future__ import annotations

import numpy as np
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
