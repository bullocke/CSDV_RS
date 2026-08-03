"""csdv_core.examples.screen — pick stands that can carry a worked example.

Not every delineated stand can illustrate the classification system. Some were
disturbed before the imagery record begins, some are too small to hold enough
crowns for a crown statistic, some are cut off at the edge of the mapping
module, and some had a second disturbance partway through, which is interesting
in itself but confounds a clean trajectory.

The point of running this as a table rather than by eye is that the choice of
example becomes auditable. A stand that fails a criterion is still reported,
with the criterion named, so a reader can see what was passed over and why.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ACRE_M2 = 4046.856

__all__ = ["ACRE_M2", "ScreenCriteria", "screen_stands"]


@dataclass(frozen=True)
class ScreenCriteria:
    """Thresholds a stand must meet to carry a worked example.

    Attributes:
        years: Imagery years available.
        min_acres: Smallest stand to consider. The photo-interpretation
            protocol sets a one-acre minimum mapping unit and three acres where
            a footprint is subdivided, so three is the smaller of the two units
            the interpreters actually work at.
        min_post_dates: Imagery dates at or after the disturbance was first
            seen. A trajectory needs a sequence, not a pair.
        require_pre_date: Require at least one imagery date at or before the
            last pre-disturbance image, so the example has a before.
        require_complete_footprint: Require the footprint to lie wholly inside
            the mapping module, since a truncated stand is measured over an
            arbitrary part of itself.
        require_single_event: Require no additional disturbance between the
            base and follow-up images. Turn this off when the second event is
            the point, as it is for salvage after a windthrow.
    """

    years: Sequence[int]
    min_acres: float = 3.0
    min_post_dates: int = 3
    require_pre_date: bool = True
    require_complete_footprint: bool = True
    require_single_event: bool = True
    labels: dict[str, str] = field(
        default_factory=lambda: {
            "has_pre_date": "no imagery before the disturbance",
            "enough_post_dates": "too few imagery dates after the disturbance",
            "large_enough": "smaller than the minimum mapping unit",
            "complete_footprint": "footprint truncated at the module edge",
            "single_event": "a second disturbance after the base image",
        }
    )


def screen_stands(
    stands: gpd.GeoDataFrame,
    criteria: ScreenCriteria,
) -> pd.DataFrame:
    """Score every stand against the criteria and say which it fails.

    Args:
        stands: Frame from :func:`csdv_core.io.stands.read_ais_stands`.
        criteria: Thresholds to apply.

    Returns:
        A frame carrying ``acres``, ``bbox_fill``, the count of usable imagery
        dates before and after the disturbance, one boolean column per
        criterion, an overall ``passes``, and a readable ``fails`` listing every
        criterion the stand missed.
    """
    years = np.asarray(sorted(criteria.years))
    out = pd.DataFrame(
        {
            "stand_id": stands["stand_id"].to_numpy(),
            "dist_label": stands["dist_label"].to_numpy(),
            "dist_group": stands["dist_group"].to_numpy(),
            "acres": stands["area_m2"].to_numpy(dtype=float) / ACRE_M2,
            "bbox_fill": stands["bbox_fill"].to_numpy(dtype=float),
            "last_pre": stands["LastImageryPreDist"].to_numpy(),
            "first_post": stands["FirstImageryPostDist"].to_numpy(),
        }
    )
    out["n_pre_dates"] = [int((years <= y).sum()) for y in out["last_pre"]]
    out["n_post_dates"] = [int((years >= y).sum()) for y in out["first_post"]]

    checks: dict[str, np.ndarray] = {
        "has_pre_date": out["n_pre_dates"].to_numpy() >= 1,
        "enough_post_dates": out["n_post_dates"].to_numpy() >= criteria.min_post_dates,
        "large_enough": out["acres"].to_numpy() >= criteria.min_acres,
        "complete_footprint": stands["WithinMappingarea"].to_numpy() == 1,
        "single_event": stands["AdditionalDisturbance"].to_numpy() == 0,
    }
    applied = {
        "has_pre_date": criteria.require_pre_date,
        "enough_post_dates": True,
        "large_enough": True,
        "complete_footprint": criteria.require_complete_footprint,
        "single_event": criteria.require_single_event,
    }
    for name, values in checks.items():
        out[name] = values

    passes = np.ones(len(out), dtype=bool)
    for name, values in checks.items():
        if applied[name]:
            passes &= values
    out["passes"] = passes
    out["fails"] = [
        "; ".join(
            criteria.labels.get(name, name)
            for name, values in checks.items()
            if applied[name] and not values[i]
        )
        for i in range(len(out))
    ]

    logger.info(
        "%d of %d stands pass the screen over %s",
        int(passes.sum()),
        len(out),
        list(years),
    )
    for name, values in checks.items():
        if applied[name]:
            logger.info("  %-24s %d stands fail", name, int((~values).sum()))
    return out.sort_values(["passes", "acres"], ascending=[False, False]).reset_index(
        drop=True
    )
