"""csdv_core.zonal.compute — run every stand metric over a module.

This is the entry point the analysis notebooks call: given the stand polygons
and one canopy height model plus one NAIP image per date, produce a tidy table
of every metric for every stand on every date.

Two invariants hold the results together. All dates sit on one grid, so a
difference between dates needs no resampling and gap persistence compares like
with like. And crowns are segmented once per date over the whole scene under a
single peak-detection rule, so a crown statistic means the same thing in a small
stand as in a large one. :func:`assert_common_grid` checks the first before any
metric is computed, because a silent misalignment would corrupt every change
metric downstream while leaving each single-date value looking reasonable.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from csdv_core.zonal.crowns import MIN_CROWNS, crown_diameter_stats, crowns_in_stand
from csdv_core.zonal.deltas import add_change_metrics
from csdv_core.zonal.mask import read_stand_array
from csdv_core.zonal.pixel import (
    CANOPY_HEIGHT_THRESHOLD_M,
    crown_fraction,
    gap_fraction,
    gap_persistence,
    height_stats,
    mid_canopy_fraction,
    shrub_fraction,
    small_tree_fraction,
    tall_canopy_fraction,
)
from csdv_core.zonal.record import StandMetricRecord, records_to_frame
from csdv_core.zonal.spatial import stand_spatial_metrics
from csdv_core.zonal.texture import texture_entropy

logger = logging.getLogger(__name__)

#: NAIP band the texture metric reads. NAIP is delivered red, green, blue, near
#: infrared, and the near infrared band separates sunlit canopy from shadow more
#: cleanly than the visible bands.
NAIP_NIR_BAND = 4

__all__ = [
    "DateInputs",
    "assert_common_grid",
    "compute_module_metrics",
    "compute_stand_record",
]


@dataclass(frozen=True)
class DateInputs:
    """The rasters and provenance for one imagery date.

    Attributes:
        year: Imagery year.
        date: Acquisition date as ``YYYY-MM-DD``.
        native_res_m: Ground sample distance of the source imagery. NAIP is
            1.0 m in Indiana before 2016 and 0.6 m after, and a canopy height
            model derived from the coarser imagery is not equivalent to one
            derived from the finer, even after both land on the same grid.
        chm_path: Canopy height model in metres.
        naip_path: Four-band NAIP image, needed for texture. Optional.
        chm_scale: Multiplier applied on read. Use 0.01 for a centimetre
            product, 1.0 for one already in metres.
    """

    year: int
    date: str
    native_res_m: float
    chm_path: Path
    naip_path: Path | None = None
    chm_scale: float = 1.0


def assert_common_grid(paths: Sequence[Path | str]) -> None:
    """Check that every raster shares one CRS, transform and shape.

    Raises:
        ValueError: On the first raster that disagrees with the first one.
    """
    reference: tuple[object, object, tuple[int, int]] | None = None
    reference_path: Path | None = None
    for path in paths:
        with rasterio.open(path) as src:
            current = (src.crs, src.transform, (src.height, src.width))
        if reference is None:
            reference, reference_path = current, Path(path)
            continue
        if current != reference:
            raise ValueError(
                f"{Path(path).name} is not on the same grid as "
                f"{reference_path.name if reference_path else '?'}:\n"
                f"  got      crs={current[0]} shape={current[2]}\n"
                f"           transform={current[1]}\n"
                f"  expected crs={reference[0]} shape={reference[2]}\n"
                f"           transform={reference[1]}"
            )
    logger.info("All %d rasters share one grid", len(paths))


def compute_stand_record(
    stand_id: str,
    geometry: object,
    date: DateInputs,
    *,
    crowns: gpd.GeoDataFrame | None = None,
    prev_chm_path: Path | None = None,
    prev_chm_scale: float = 1.0,
    area_m2: float | None = None,
    height_threshold_m: float = CANOPY_HEIGHT_THRESHOLD_M,
    min_crowns: int = MIN_CROWNS,
    texture_levels: int = 16,
) -> StandMetricRecord:
    """Compute every metric for one stand on one date.

    Args:
        stand_id: Stand identifier.
        geometry: Stand geometry in the raster CRS.
        date: The date's rasters and provenance.
        crowns: Crowns segmented over the scene for this date. Crowns are
            assigned to the stand by centroid. Omit to skip crown metrics.
        prev_chm_path: Previous date's canopy height model, for gap
            persistence. Omit at the first date.
        prev_chm_scale: Read scale for ``prev_chm_path``.
        area_m2: Stand area from the geometry. Computed from the mask when not
            given, which differs slightly because of the pixel-centre rule.
        height_threshold_m: Gap and canopy height threshold.
        min_crowns: Minimum crowns before a crown statistic is reported.
        texture_levels: Grey levels for the co-occurrence matrix.

    Returns:
        A :class:`StandMetricRecord`. Any metric that could not be computed is
        NaN with a stated reason.
    """
    chm, window = read_stand_array(
        date.chm_path, geometry, scale=date.chm_scale, stand_id=stand_id
    )
    mask = window.mask
    metrics: dict[str, float] = {}
    support: dict[str, float] = {}
    reasons: dict[str, str] = {}

    metrics["gap_fraction"] = gap_fraction(
        chm, mask, height_threshold_m=height_threshold_m
    )
    metrics["crown_fraction"] = crown_fraction(
        chm, mask, height_threshold_m=height_threshold_m
    )
    metrics["shrub_fraction"] = shrub_fraction(chm, mask)
    metrics["small_tree_fraction"] = small_tree_fraction(chm, mask)
    metrics["mid_canopy_fraction"] = mid_canopy_fraction(chm, mask)
    metrics["tall_canopy_fraction"] = tall_canopy_fraction(chm, mask)
    metrics.update(height_stats(chm, mask, height_threshold_m=height_threshold_m))

    valid = mask & np.isfinite(chm)
    n_pixels = int(valid.sum())
    support["n_valid_pixels"] = float(n_pixels)
    support["nodata_fraction"] = (
        1.0 - n_pixels / window.n_inside if window.n_inside else float("nan")
    )

    spatial, spatial_support, spatial_reasons = stand_spatial_metrics(
        chm,
        mask,
        pixel_size_m=window.pixel_size_m,
        height_threshold_m=height_threshold_m,
    )
    metrics.update(spatial)
    support.update(spatial_support)
    reasons.update(spatial_reasons)

    if prev_chm_path is not None:
        previous, prev_window = read_stand_array(
            prev_chm_path, geometry, scale=prev_chm_scale, stand_id=stand_id
        )
        if prev_window.transform != window.transform:
            raise ValueError(
                f"Stand {stand_id!r}: {Path(prev_chm_path).name} is not on the "
                f"same grid as {Path(date.chm_path).name}. A difference between "
                "dates is only meaningful pixel for pixel, and no resampling is "
                "performed here."
            )
        metrics["gap_persistence"] = gap_persistence(
            previous, chm, prev_window.mask, height_threshold_m=height_threshold_m
        )
    else:
        metrics["gap_persistence"] = float("nan")
        reasons["gap_persistence"] = "first date of the series, nothing to compare to"

    if date.naip_path is not None:
        nir, nir_window = read_stand_array(
            date.naip_path, geometry, band=NAIP_NIR_BAND, stand_id=stand_id
        )
        if nir_window.transform != window.transform:
            raise ValueError(
                f"Stand {stand_id!r}: {Path(date.naip_path).name} is not on the "
                f"same grid as {Path(date.chm_path).name}"
            )
        texture = texture_entropy(nir, nir_window.mask, levels=texture_levels)
        metrics["glcm_texture"] = texture.entropy_bits
        support["texture_n_valid"] = float(texture.n_valid)
        if texture.reason:
            reasons["glcm_texture"] = texture.reason
    else:
        metrics["glcm_texture"] = float("nan")
        reasons["glcm_texture"] = "no NAIP image supplied"

    if crowns is not None:
        stats = crown_diameter_stats(
            crowns_in_stand(crowns, geometry), min_crowns=min_crowns
        )
        metrics.update(stats.as_metrics())
        support["n_crowns"] = float(stats.n_crowns)
        if stats.reason:
            for name in ("crown_cv", "crown_p90", "crown_mean"):
                reasons[name] = stats.reason
    else:
        for name in ("crown_cv", "crown_p90", "crown_mean", "crown_count"):
            metrics[name] = float("nan")
            reasons[name] = "no crown segmentation supplied"

    return StandMetricRecord(
        stand_id=stand_id,
        date=date.date,
        year=date.year,
        native_res_m=date.native_res_m,
        area_m2=float(area_m2) if area_m2 is not None else window.area_m2,
        n_pixels=n_pixels,
        bbox_fill_fraction=window.bbox_fill_fraction,
        metrics=metrics,
        support=support,
        reasons=reasons,
    )


def compute_module_metrics(
    stands: gpd.GeoDataFrame,
    dates: Sequence[DateInputs],
    *,
    crowns_by_year: Mapping[int, gpd.GeoDataFrame] | None = None,
    check_grid: bool = True,
    with_change_metrics: bool = True,
    **record_kwargs: object,
) -> pd.DataFrame:
    """Compute every metric for every stand across every date.

    Args:
        stands: Stand polygons from
            :func:`csdv_core.io.stands.read_ais_stands`.
        dates: Imagery dates in any order; they are sorted by year, and gap
            persistence at each date compares it with the one before.
        crowns_by_year: Scene crowns per year, from
            :func:`csdv_core.zonal.crowns.segment_scene_crowns`.
        check_grid: Verify that every canopy height model shares one grid
            before computing anything.
        with_change_metrics: Append consecutive-date differences.
        **record_kwargs: Forwarded to :func:`compute_stand_record`.

    Returns:
        A tidy frame, one row per stand per date.
    """
    ordered = sorted(dates, key=lambda d: d.year)
    if check_grid:
        assert_common_grid([d.chm_path for d in ordered])

    records: list[StandMetricRecord] = []
    for _, stand in stands.iterrows():
        stand_id = str(stand["stand_id"])
        geometry = stand.geometry
        previous: DateInputs | None = None
        for date in ordered:
            crowns = (crowns_by_year or {}).get(date.year)
            try:
                record = compute_stand_record(
                    stand_id,
                    geometry,
                    date,
                    crowns=crowns,
                    prev_chm_path=previous.chm_path if previous else None,
                    prev_chm_scale=previous.chm_scale if previous else 1.0,
                    area_m2=float(stand.get("area_m2", float("nan"))),
                    **record_kwargs,
                )
            except ValueError:
                logger.exception(
                    "Stand %s at %s could not be computed", stand_id, date.date
                )
                previous = date
                continue
            records.append(record)
            previous = date

    frame = records_to_frame(records)
    logger.info(
        "Computed %d stand-date records over %d stands and %d dates",
        len(frame),
        len(stands),
        len(ordered),
    )
    if with_change_metrics and not frame.empty:
        frame = add_change_metrics(frame)
    return frame
