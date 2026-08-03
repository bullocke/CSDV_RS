"""csdv_core.zonal.crowns — crown statistics for one stand.

Crowns are segmented over the whole scene, not stand by stand, so that a tree
near a boundary is delineated against its real neighbours. Each crown is then
assigned to the stand containing its centroid, which counts every crown exactly
once even where crowns straddle a boundary.

Below a minimum crown count no statistic is reported, because a coefficient of
variation over a handful of crowns says more about the sample than the stand.
The count and the reason travel with the result rather than leaving a blank.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

DIAMETER_COLUMN = "crown_diam_m"

#: Minimum crowns before a diameter statistic is reported. Carried over from
#: the windowed implementation and almost certainly too low for a stable
#: coefficient of variation. The defensible value is the count at which crown
#: CV stops moving, which has not been measured.
MIN_CROWNS = 3

__all__ = [
    "DIAMETER_COLUMN",
    "MIN_CROWNS",
    "CrownStats",
    "crown_diameter_stats",
    "crowns_in_stand",
    "segment_scene_crowns",
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


def segment_scene_crowns(
    chm: np.ndarray,
    transform: object,
    crs: object,
    *,
    block_px: int = 2048,
    overlap_px: int = 64,
    min_height_m: float = 2.0,
    **segment_kwargs: object,
) -> gpd.GeoDataFrame:
    """Segment crowns over a whole scene in overlapping blocks.

    A module-sized canopy height model is too large to watershed in one pass,
    but splitting it changes the answer unless the local-maximum footprint is
    held constant: :func:`csdv_core.segmentation.chm_watershed.segment_crowns`
    otherwise derives that footprint from the mean height of whatever array it
    is given. The scene mean is therefore computed once and passed to every
    block.

    Blocks overlap so that a crown spanning a seam is segmented whole in at
    least one block. Crowns whose centroid falls in another block's interior
    are dropped, which keeps each crown once.

    Args:
        chm: 2-D canopy height array in metres, NaN for nodata.
        transform: Affine transform of ``chm``.
        crs: Output CRS.
        block_px: Interior block side length in pixels.
        overlap_px: Overlap added on every side. Must exceed the largest
            plausible crown radius in pixels.
        min_height_m: Passed through, and used for the scene mean.
        **segment_kwargs: Forwarded to ``segment_crowns``.

    Returns:
        A GeoDataFrame of crowns with ``segment_id``, ``area_m2`` and
        ``crown_diam_m``.
    """
    import rasterio
    from rasterio.windows import Window

    from csdv_core.segmentation.chm_watershed import segment_crowns

    arr = np.asarray(chm, dtype=np.float32)
    canopy = arr[np.isfinite(arr) & (arr >= min_height_m)]
    if canopy.size == 0:
        logger.warning(
            "segment_scene_crowns: no canopy pixels above %.1f m", min_height_m
        )
        return gpd.GeoDataFrame(
            {"segment_id": [], "area_m2": [], DIAMETER_COLUMN: []},
            geometry=[],
            crs=crs,
        )
    scene_mean_h = float(np.mean(canopy))
    rows, cols = arr.shape
    parts: list[gpd.GeoDataFrame] = []
    next_id = 0

    for r0 in range(0, rows, block_px):
        for c0 in range(0, cols, block_px):
            r1 = min(rows, r0 + block_px)
            c1 = min(cols, c0 + block_px)
            er0, ec0 = max(0, r0 - overlap_px), max(0, c0 - overlap_px)
            er1, ec1 = min(rows, r1 + overlap_px), min(cols, c1 + overlap_px)
            block = arr[er0:er1, ec0:ec1]
            if not np.isfinite(block).any():
                continue
            block_transform = rasterio.windows.transform(
                Window(col_off=ec0, row_off=er0, width=ec1 - ec0, height=er1 - er0),
                transform,
            )
            found = segment_crowns(
                block,
                block_transform,
                crs,
                min_height_m=min_height_m,
                mean_height_m=scene_mean_h,
                **segment_kwargs,
            )
            if found.empty:
                continue
            # Keep only crowns centred in this block's interior, so the overlap
            # supplies context without duplicating crowns.
            interior = rasterio.windows.bounds(
                Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0),
                transform,
            )
            centroids = found.geometry.centroid
            keep = (
                (centroids.x >= interior[0])
                & (centroids.x < interior[2])
                & (centroids.y >= interior[1])
                & (centroids.y < interior[3])
            )
            found = found.loc[keep].copy()
            if found.empty:
                continue
            found["segment_id"] = np.arange(next_id, next_id + len(found))
            next_id += len(found)
            parts.append(found)

    if not parts:
        return gpd.GeoDataFrame(
            {"segment_id": [], "area_m2": [], DIAMETER_COLUMN: []},
            geometry=[],
            crs=crs,
        )
    out = gpd.GeoDataFrame(_concat(parts), crs=crs).reset_index(drop=True)
    logger.info(
        "segment_scene_crowns: %d crowns over %d x %d px, scene mean height %.1f m",
        len(out),
        rows,
        cols,
        scene_mean_h,
    )
    return out


def _concat(frames: Iterable[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    import pandas as pd

    return pd.concat(list(frames), ignore_index=True)
