"""csdv_core.zonal.crowns — crown statistics for one stand.

Crowns are segmented over the whole scene, not stand by stand, so that a tree
near a boundary is delineated against its real neighbours. Segmentation runs in
the naipcrowns package (:func:`naipcrowns.tiles.segment_raster`, or
``csdv segment-crowns``). This module only reads the crowns it writes. Each crown is then
assigned to the stand containing its centroid, which counts every crown exactly
once even where crowns straddle a boundary.

Below a minimum crown count no statistic is reported, because a coefficient of
variation over a handful of crowns says more about the sample than the stand.
The count and the reason travel with the result rather than leaving a blank.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

DIAMETER_COLUMN = "crown_diam_m"

#: Minimum crowns before a diameter statistic is reported.
#:
#: Measured, not assumed. The narrowest ``crown_cv`` band in
#: ``config/stages.yaml`` is 0.10 wide, so a stand cannot be placed in a band
#: unless its crown_cv interval is narrower than that. Bootstrapping the
#: interval against sample size puts the crossing at 75 crowns, which at the
#: measured density is about 1.3 ha or 3.3 acres. See
#: ``docs/guides/crown_segmentation/optimization/``.
#:
#: This was 3, inherited from the windowed implementation. At three crowns the
#: 90 percent interval is 0.41 wide, four times the width of the band it is
#: meant to resolve, and the estimate is biased low as well as noisy: mean
#: crown_cv reads 0.24 at n=3 against 0.33 at n above 50.
#:
#: The cost is coverage. 26 of the 40 Elkinsville calibration stands clear 75
#: crowns, where all 40 cleared 3.
MIN_CROWNS = 75

__all__ = [
    "DIAMETER_COLUMN",
    "MIN_CROWNS",
    "CrownStats",
    "crown_diameter_stats",
    "crowns_in_stand",
]


@dataclass(frozen=True)
class CrownStats:
    """Crown diameter statistics for one stand.

    Attributes:
        n_crowns: Crowns whose centroid falls inside the stand.
        cv: Standard deviation of crown diameter over its mean, dimensionless.
        p90: 90th percentile crown diameter in metres, the diagnostic for
            selective removal of the largest trees.
        mean: Mean crown diameter in metres.
        median: Median crown diameter in metres.
        std: Standard deviation of crown diameter in metres.
        min_crowns: The threshold that was applied.
        reason: Empty when computed, otherwise why the statistics are NaN.
    """

    n_crowns: int
    cv: float
    p90: float
    mean: float
    median: float
    std: float
    min_crowns: int = MIN_CROWNS
    reason: str = ""

    def as_metrics(self) -> dict[str, float]:
        """Return the statistics under the names the metric registry uses."""
        return {
            "crown_cv": self.cv,
            "crown_p90": self.p90,
            "crown_mean": self.mean,
            "crown_median": self.median,
            "crown_std": self.std,
            "crown_count": float(self.n_crowns),
        }


def crowns_in_stand(
    crowns: gpd.GeoDataFrame,
    geometry: BaseGeometry,
) -> gpd.GeoDataFrame:
    """Select the crowns whose centroid falls inside ``geometry``.

    Centroid assignment is what keeps a crown straddling a boundary from being
    counted in two stands, or split into two partial crowns.
    """
    if crowns.empty:
        return crowns
    inside = crowns.geometry.centroid.within(geometry)
    return crowns.loc[inside]


def crown_diameter_stats(
    crowns: gpd.GeoDataFrame,
    *,
    diam_column: str = DIAMETER_COLUMN,
    min_crowns: int = MIN_CROWNS,
) -> CrownStats:
    """Summarise crown diameters, or report why they could not be summarised.

    Args:
        crowns: Crowns already restricted to one stand.
        diam_column: Column holding crown diameter in metres.
        min_crowns: Below this many crowns nothing is reported.

    Returns:
        A :class:`CrownStats`. Every statistic is NaN when the stand holds
        fewer than ``min_crowns`` crowns, and ``reason`` says so.

    Raises:
        KeyError: If ``diam_column`` is missing from a non-empty frame.
    """
    if crowns.empty:
        return CrownStats(
            n_crowns=0,
            cv=float("nan"),
            p90=float("nan"),
            mean=float("nan"),
            median=float("nan"),
            std=float("nan"),
            min_crowns=min_crowns,
            reason=f"n_crowns=0 < min_crowns={min_crowns}",
        )
    if diam_column not in crowns.columns:
        raise KeyError(
            f"Crown frame has no {diam_column!r} column; "
            f"segment_crowns emits {DIAMETER_COLUMN!r}"
        )
    diameters = crowns[diam_column].to_numpy(dtype=np.float64)
    diameters = diameters[np.isfinite(diameters)]
    n = int(diameters.size)
    if n < min_crowns:
        return CrownStats(
            n_crowns=n,
            cv=float("nan"),
            p90=float("nan"),
            mean=float("nan"),
            median=float("nan"),
            std=float("nan"),
            min_crowns=min_crowns,
            reason=f"n_crowns={n} < min_crowns={min_crowns}",
        )
    mean = float(np.mean(diameters))
    std = float(np.std(diameters))
    return CrownStats(
        n_crowns=n,
        cv=float(std / mean) if mean > 0 else float("nan"),
        p90=float(np.percentile(diameters, 90)),
        mean=mean,
        median=float(np.median(diameters)),
        std=std,
        min_crowns=min_crowns,
    )
