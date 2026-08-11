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
import math
from collections.abc import Iterable
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
#: ``docs/guides/segmentation_optimization/``.
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
    "default_overlap_px",
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


def default_overlap_px(params: object, pixel_size_m: float) -> int:
    """Block halo wide enough that seam crowns match interior crowns.

    A halo of one crown radius is not enough. A crown's boundary is set by
    competition with its neighbours, so those neighbours must be complete too,
    which takes a full crown diameter. Add the smoothing kernel on top, because
    the mean filter reflects at a block edge and manufactures tree tops there.

    Falls back to a generous fixed halo when the parameters carry no radius
    ceiling, since an unbounded watershed gives no crown size to derive from.
    """
    from csdv_core.segmentation.chm_watershed import smooth_kernel_px

    smooth_px = smooth_kernel_px(getattr(params, "smooth_radius_m", 0.0), pixel_size_m)
    max_radius_m = getattr(params, "max_crown_radius_m", None)
    if max_radius_m is None:
        return 128 + smooth_px
    return int(math.ceil(2.0 * float(max_radius_m) / float(pixel_size_m))) + smooth_px


def segment_scene_crowns(
    chm: np.ndarray,
    transform: object,
    crs: object,
    *,
    block_px: int = 2048,
    overlap_px: int | None = None,
    params: object | None = None,
) -> gpd.GeoDataFrame:
    """Segment crowns over a whole scene in overlapping blocks.

    A module-sized canopy height model is too large to watershed in one pass.
    Splitting it used to change the answer, because the search window was
    derived from the mean height of whatever array was passed in, so the scene
    mean had to be threaded through every block. The window is now evaluated
    per pixel from that pixel's own height, so blocks are independent and no
    shared scene statistic is needed.

    Blocks overlap so a crown spanning a seam is segmented whole in at least
    one block. Crowns whose centroid falls in another block's interior are
    dropped, which keeps each crown exactly once.

    Args:
        chm: 2-D canopy height array in metres, NaN for nodata.
        transform: Affine transform of ``chm``.
        crs: Output CRS.
        block_px: Interior block side length in pixels.
        overlap_px: Halo added on every side. Derived from the crown radius
            ceiling when omitted, which is the only value that is safe.
        params: :class:`~csdv_core.segmentation.params.SegmentationParams`.

    Returns:
        A GeoDataFrame of crowns carrying the full crown schema.
    """
    import rasterio
    from rasterio.windows import Window

    from csdv_core.segmentation.chm_watershed import CROWN_COLUMNS, segment_crowns
    from csdv_core.segmentation.params import DEFAULT_PARAMS

    if params is None:
        params = DEFAULT_PARAMS
    pixel_size_m = float(abs(transform.a))  # type: ignore[attr-defined]
    if overlap_px is None:
        overlap_px = default_overlap_px(params, pixel_size_m)

    arr = np.asarray(chm, dtype=np.float32)
    min_height_m = float(getattr(params, "min_height_m", 2.0))
    if not np.any(np.isfinite(arr) & (arr >= min_height_m)):
        logger.warning(
            "segment_scene_crowns: no canopy pixels above %.1f m", min_height_m
        )
        return gpd.GeoDataFrame({c: [] for c in CROWN_COLUMNS}, geometry=[], crs=crs)
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
            found = segment_crowns(block, block_transform, crs, params=params)
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
            # Distance from the nearest seam, so a later QA pass can check that
            # seam crowns look like interior crowns. A step at the seam means
            # the halo is too narrow.
            found["seam_dist_m"] = _seam_distance_m(found, interior)
            parts.append(found)

    if not parts:
        return gpd.GeoDataFrame({c: [] for c in CROWN_COLUMNS}, geometry=[], crs=crs)
    out = gpd.GeoDataFrame(_concat(parts), crs=crs).reset_index(drop=True)
    logger.info(
        "segment_scene_crowns: %d crowns over %d x %d px, %d px halo",
        len(out),
        rows,
        cols,
        overlap_px,
    )
    return out


def _seam_distance_m(
    crowns: gpd.GeoDataFrame, interior: tuple[float, float, float, float]
) -> np.ndarray:
    """Distance from each crown centroid to the nearest block seam."""
    cx = crowns.geometry.centroid.x.to_numpy()
    cy = crowns.geometry.centroid.y.to_numpy()
    return np.minimum(
        np.minimum(cx - interior[0], interior[2] - cx),
        np.minimum(cy - interior[1], interior[3] - cy),
    )


def _concat(frames: Iterable[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    import pandas as pd

    return pd.concat(list(frames), ignore_index=True)
