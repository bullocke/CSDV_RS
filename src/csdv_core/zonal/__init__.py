"""csdv_core.zonal — stand-level metrics computed inside a polygon.

The classification system reports one value per stand per date, where a stand
is the hand-delineated impact polygon the photo interpreters deliver. That is a
different calling convention from :mod:`csdv_core.metrics`, which divides a
raster into a grid of square windows and returns a
:class:`~csdv_core.metrics._result.MetricResult` per metric. The two packages
sit side by side rather than one inside the other, because the windowed metric
registry and orchestrator are keyed to the ``(array, GridSpec, window_m)``
signature and cannot dispatch a polygon.

The rules implemented here follow Section 2 of the project's classification
document:

* Pixel metrics use every pixel whose centre falls inside the polygon.
* Crown metrics assign a crown to the stand containing its centroid, so a crown
  straddling a boundary is counted once, and report nothing below a minimum
  crown count rather than reporting an unstable statistic.
* Texture and spatial metrics need a rectangular support, so they run over the
  bounding box with everything outside the polygon masked, and the in-polygon
  fraction of that box is reported alongside the metric.
* Gap persistence compares two dates on an identical grid with no resampling.

A metric that cannot be computed is reported as NaN with a recorded reason,
never as a silent blank.
"""

from __future__ import annotations

from csdv_core.zonal.crowns import (
    CrownStats,
    crown_diameter_stats,
    crowns_in_stand,
    segment_scene_crowns,
)
from csdv_core.zonal.deltas import add_change_metrics
from csdv_core.zonal.mask import StandWindow, read_stand_array, stand_window
from csdv_core.zonal.pixel import (
    crown_fraction,
    gap_fraction,
    gap_persistence,
    height_band_fraction,
    height_stats,
    mid_canopy_fraction,
    shrub_fraction,
    small_tree_fraction,
    tall_canopy_fraction,
)
from csdv_core.zonal.record import StandMetricRecord, records_to_frame
from csdv_core.zonal.spatial import (
    SpatialResult,
    stand_edge_density,
    stand_linearity,
    stand_row_directionality,
    stand_spatial_metrics,
)
from csdv_core.zonal.texture import TextureResult, texture_entropy

__all__ = [
    "CrownStats",
    "SpatialResult",
    "StandMetricRecord",
    "StandWindow",
    "TextureResult",
    "add_change_metrics",
    "crown_diameter_stats",
    "crown_fraction",
    "crowns_in_stand",
    "gap_fraction",
    "gap_persistence",
    "height_band_fraction",
    "height_stats",
    "mid_canopy_fraction",
    "read_stand_array",
    "records_to_frame",
    "segment_scene_crowns",
    "shrub_fraction",
    "small_tree_fraction",
    "stand_edge_density",
    "stand_linearity",
    "stand_row_directionality",
    "stand_spatial_metrics",
    "stand_window",
    "tall_canopy_fraction",
    "texture_entropy",
]
