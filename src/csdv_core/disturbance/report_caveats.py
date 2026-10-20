"""csdv_core.disturbance.report_caveats: run-specific caveats for the report.

Two caveats depend on the run itself. Objects whose onset falls in the last
interval have no later image, so nothing confirms that their loss persists.
Intervals whose two images were flown at different times of the growing
season compare canopies in different phenological states (September CHMs
tend to be taller than June CHMs), so part of their CHM difference may be
seasonal.

Inputs: the input inventory, the years and the objects table. Outputs:
caveat sentences for :attr:`csdv_core.disturbance.report.RunSummary.caveats`.
"""

from __future__ import annotations

import datetime
import logging
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import geopandas as gpd

from csdv_core.disturbance.outlines import interval_label

if TYPE_CHECKING:
    from csdv_core.disturbance.report import InputFileInfo

logger = logging.getLogger(__name__)

#: Largest day-of-year gap between the two images of an interval before the
#: report warns about phenology (six weeks).
SEASON_GAP_DAYS = 42
#: Objects column that flags a later image after the onset image.
FOLLOW_UP_COLUMN = "follow_up"


def earliest_acquisition_dates(
    inputs: Iterable[InputFileInfo],
) -> dict[int, datetime.date]:
    """Earliest acquisition date of each year in the inventory.

    Args:
        inputs: Input inventory entries (CHM and NAIP files).

    Returns:
        Year to earliest date. Years without any parsed date are absent.
    """
    earliest: dict[int, datetime.date] = {}
    for entry in inputs:
        if entry.acquisition_date is None:
            continue
        acquisition_date = datetime.date.fromisoformat(entry.acquisition_date)
        current = earliest.get(entry.year)
        if current is None or acquisition_date < current:
            earliest[entry.year] = acquisition_date
    return earliest


def season_caveats(inputs: Iterable[InputFileInfo], years: Sequence[int]) -> list[str]:
    """Caveats for intervals whose images differ by more than six weeks of season.

    For each interval, the day-of-year difference of the earliest acquisition
    dates of its two years is compared with :data:`SEASON_GAP_DAYS`.
    Intervals with a missing date are not checked.

    Args:
        inputs: Input inventory entries.
        years: NAIP years of the run, ascending.

    Returns:
        One sentence per interval above the threshold, in interval order.
    """
    dates_by_year = earliest_acquisition_dates(inputs)
    caveats: list[str] = []
    for year_index in range(1, len(years)):
        year_before = years[year_index - 1]
        year = years[year_index]
        date_before = dates_by_year.get(year_before)
        date_after = dates_by_year.get(year)
        if date_before is None or date_after is None:
            continue
        day_before = date_before.timetuple().tm_yday
        day_after = date_after.timetuple().tm_yday
        season_gap = abs(day_after - day_before)
        if season_gap <= SEASON_GAP_DAYS:
            continue
        label = interval_label(year_before, year)
        caveats.append(
            f"{label}: images from {date_before.isoformat()} and "
            f"{date_after.isoformat()} differ by {season_gap} days of season, "
            "so CHM differences between them may reflect phenology."
        )
    return caveats


def follow_up_caveat(objects: gpd.GeoDataFrame, years: Sequence[int]) -> str:
    """Caveat that counts the objects without a later image.

    Uses the ``follow_up`` column when present, else treats objects whose
    onset year is the last year as lacking a follow-up image.

    Args:
        objects: Objects GeoDataFrame (after the MMU).
        years: NAIP years of the run, ascending.

    Returns:
        One sentence with the count and area of objects without a follow-up
        image.
    """
    last_year = years[-1]
    if FOLLOW_UP_COLUMN in objects.columns:
        follow_up = objects[FOLLOW_UP_COLUMN].astype(bool)
        lacking = ~follow_up
    else:
        lacking = objects["onset_year"] == last_year
    lacking_count = int(lacking.sum())
    lacking_area = objects.loc[lacking, "area_ha"]
    lacking_ha = float(lacking_area.sum())
    total_count = len(objects)
    last_label = interval_label(years[-2], last_year)
    return (
        f"{lacking_count} of {total_count} objects ({lacking_ha:.2f} ha) have no "
        f"follow-up image. Their onset falls in the last interval ({last_label}), "
        "so no later image confirms that the loss persists."
    )


__all__ = [
    "FOLLOW_UP_COLUMN",
    "SEASON_GAP_DAYS",
    "earliest_acquisition_dates",
    "follow_up_caveat",
    "season_caveats",
]
