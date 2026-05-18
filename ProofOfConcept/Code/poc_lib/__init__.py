"""
poc_lib — Reusable library for CSDV proof-of-concept analyses.

Provides importable I/O helpers, metric computation functions, crown polygon
utilities, figure utilities, and site configuration shared across analysis
scripts in ProofOfConcept/Code/analysis/.

Existing scripts in summary_document/ are not modified; they continue to work
independently. New analysis scripts import from this package instead.
"""

from .crowns import crown_stats_per_window, iou_stats, rasterize_crowns
from .figures import add_scale_bar, panel_label, rgb_display, save_fig, shared_cbar
from .io import (
    clip_and_convert_naip_chm,
    clip_raster_to_bbox,
    find_latest_tif,
    read_band,
)
from .metrics import (
    crown_fraction,
    crown_width_p90,
    gap_fraction,
    metric_difference,
    save_raster,
)
from .sites import SITES, SiteConfig, get_site

__all__ = [
    # sites
    "SiteConfig",
    "SITES",
    "get_site",
    # io
    "find_latest_tif",
    "clip_raster_to_bbox",
    "clip_and_convert_naip_chm",
    "read_band",
    # metrics
    "gap_fraction",
    "crown_fraction",
    "crown_width_p90",
    "metric_difference",
    "save_raster",
    # crowns
    "rasterize_crowns",
    "iou_stats",
    "crown_stats_per_window",
    # figures
    "add_scale_bar",
    "shared_cbar",
    "panel_label",
    "rgb_display",
    "save_fig",
]
