"""csdv_core.viz — figure helpers shared by notebooks and regeneration scripts.

Style constants and export settings live in :mod:`csdv_core.viz.style`, image
panels in :mod:`csdv_core.viz.maps`, and agreement and screening plots in
:mod:`csdv_core.viz.scatter`.
"""

from __future__ import annotations

from csdv_core.viz.maps import (
    chm_panel,
    draw_stand_outline,
    padded_bounds,
    rgb_panel,
    stretch_rgb,
)
from csdv_core.viz.scatter import cover_class_agreement, screening_scatter, spearman
from csdv_core.viz.style import (
    STAGE_COLORS,
    add_scale_bar,
    panel_label,
    save_fig,
    setup_style,
    stage_color,
)

__all__ = [
    "STAGE_COLORS",
    "add_scale_bar",
    "chm_panel",
    "cover_class_agreement",
    "draw_stand_outline",
    "padded_bounds",
    "panel_label",
    "rgb_panel",
    "save_fig",
    "screening_scatter",
    "setup_style",
    "spearman",
    "stage_color",
    "stretch_rgb",
]
