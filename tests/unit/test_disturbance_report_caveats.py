"""Tests for the run-specific caveats of the disturbance report."""

from __future__ import annotations

import datetime

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from csdv_core.disturbance.report import InputFileInfo
from csdv_core.disturbance.report_caveats import (
    SEASON_GAP_DAYS,
    earliest_acquisition_dates,
    follow_up_caveat,
    season_caveats,
)

#: Years of the synthetic run.
YEARS = (2019, 2021, 2023)


def _entry(year: int, date_text: str | None, kind: str = "chm") -> InputFileInfo:
    """One inventory entry with the given acquisition date.

    Args:
        year: NAIP year.
        date_text: ISO date or None.
        kind: ``"chm"`` or ``"naip"``.

    Returns:
        The inventory entry.
    """
    return InputFileInfo(
        year=year,
        kind=kind,
        path=f"/data/{year}.tif",
        acquisition_date=date_text,
        dtype="float32",
        scale=1.0,
        nodata=None,
    )


def _objects(onset_years: list[int], follow_up: list[bool] | None) -> gpd.GeoDataFrame:
    """Objects with one 0.5 ha square per onset year.

    Args:
        onset_years: Onset year of each object.
        follow_up: ``follow_up`` values, or None to leave the column out.

    Returns:
        The objects GeoDataFrame.
    """
    record: dict[str, list] = {
        "onset_year": onset_years,
        "area_ha": [0.5] * len(onset_years),
    }
    if follow_up is not None:
        record["follow_up"] = follow_up
    frame = pd.DataFrame(record)
    squares = [box(index, 0, index + 1, 1) for index in range(len(onset_years))]
    return gpd.GeoDataFrame(frame, geometry=squares, crs="EPSG:32617")


def test_earliest_acquisition_dates_takes_minimum_and_skips_none() -> None:
    inputs = [
        _entry(2019, "2019-07-02"),
        _entry(2019, "2019-06-30", kind="naip"),
        _entry(2021, None),
    ]
    dates = earliest_acquisition_dates(inputs)
    assert dates == {2019: datetime.date(2019, 6, 30)}


def test_season_caveat_flags_large_day_of_year_gap() -> None:
    inputs = [
        _entry(2019, "2019-09-01"),
        _entry(2021, "2021-09-20"),
        _entry(2021, "2021-10-01"),
        _entry(2023, "2023-06-23"),
    ]
    caveats = season_caveats(inputs, YEARS)
    assert caveats == [
        "2021-2023: images from 2021-09-20 and 2023-06-23 differ by 89 days of "
        "season, so CHM differences between them may reflect phenology."
    ]


def test_season_caveat_threshold_is_exclusive() -> None:
    first_date = datetime.date(2019, 6, 1)
    second_date = datetime.date(2021, 6, 1) + datetime.timedelta(days=SEASON_GAP_DAYS)
    inputs = [
        _entry(2019, first_date.isoformat()),
        _entry(2021, second_date.isoformat()),
    ]
    caveats = season_caveats(inputs, (2019, 2021))
    assert caveats == []


def test_follow_up_caveat_counts_objects_without_later_image() -> None:
    objects = _objects([2021, 2023, 2023], [True, False, False])
    caveat = follow_up_caveat(objects, YEARS)
    assert caveat.startswith("2 of 3 objects (1.00 ha) have no follow-up image")
    assert "(2021-2023)" in caveat


def test_follow_up_caveat_without_column_uses_last_year() -> None:
    objects = _objects([2021, 2023], None)
    caveat = follow_up_caveat(objects, YEARS)
    assert caveat.startswith("1 of 2 objects (0.50 ha)")
