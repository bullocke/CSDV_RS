"""csdv_core.label.outlines: the labelled area of an accepted candidate.

Inputs: a candidate outline, the routed verdicts and, where one was asked
for, the polygon a model drew.
Outputs: an :class:`EventGeometry` that splits the ground into a core
(labelled as loss), an extension (the model's polygon beyond the candidate)
and a trimmed part (candidate that the polygon leaves out).

The candidate outline stays the core whenever both models call it about
right. A model polygon is trusted as a label only for types with sharp
edges. For diffuse types it marks ground that must not become background.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shapely import make_valid
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from csdv_core.config._models import LabelOutlines
from csdv_core.label.routing import LOSS_FULL, Routed
from csdv_core.label.views import View

_EMPTY = Polygon()
_BAD_FIT = ("too_large", "wrong_shape")

__all__ = [
    "EventGeometry",
    "choose_outline",
    "is_crisp",
    "needs_polygon",
    "parse_polygon",
    "polygon_usable",
]


@dataclass
class EventGeometry:
    """Where an accepted candidate is labelled.

    Attributes:
        core: Labelled as loss.
        extension: Model polygon outside the candidate. Loss for a crisp
            event, ignore for a diffuse one.
        trimmed: Candidate outside the model polygon. Ignore.
        source: ``di_refined``, ``di_cells``, ``weak_cells`` or
            ``chm_cells``, with ``+polygon`` when a model polygon was used.
        quality: ``precise`` or ``envelope``.
        crisp: The type has sharp edges, so the extension counts as loss.
        wide_margin: A model called the outline too small and no polygon
            could be used, so more ground around it is left unlabelled.
    """

    core: BaseGeometry
    extension: BaseGeometry
    trimmed: BaseGeometry
    source: str
    quality: str
    crisp: bool
    wide_margin: bool = False

    @property
    def polygon_used(self) -> bool:
        return self.source.endswith("+polygon")


def needs_polygon(routed: Routed) -> bool:
    """Whether the models are unhappy with the candidate's outline."""
    fits = routed.outline_fit
    return any(f != "about_right" for f in fits) or "under_25" in routed.affected_share


def is_crisp(routed: Routed, cfg: LabelOutlines) -> bool:
    """Whether the agreed type has edges sharp enough to trust a polygon."""
    if routed.change_type in cfg.crisp_types:
        return True
    if routed.change_type == "fire" and cfg.crisp_fire_full:
        return (
            routed.first_loss is not None
            and routed.outcomes[routed.first_loss] == LOSS_FULL
        )
    return False


def parse_polygon(data: Any, view: View) -> BaseGeometry | None:
    """The polygon of a model answer in map coordinates, or None."""
    if not isinstance(data, dict):
        return None
    try:
        poly = view.polygon_to_map(data["polygon"])
    except Exception:  # noqa: BLE001
        return None
    poly = make_valid(poly)
    if poly.is_empty or poly.area <= 0:
        return None
    if poly.geom_type not in ("Polygon", "MultiPolygon"):
        poly = poly.buffer(0)
    return poly if not poly.is_empty else None


def polygon_usable(
    polygon: BaseGeometry | None, candidate: BaseGeometry, cfg: LabelOutlines
) -> bool:
    """A polygon is used when it is about the same event as the candidate."""
    if polygon is None or polygon.is_empty or candidate.area <= 0:
        return False
    if polygon.area < cfg.polygon_min_ha * 1e4:
        return False
    if polygon.area > cfg.polygon_max_ratio * candidate.area:
        return False
    covered = polygon.intersection(candidate).area / candidate.area
    return covered >= cfg.polygon_min_cover


def choose_outline(
    candidate: BaseGeometry,
    *,
    source: str,
    geom_src: str,
    routed: Routed,
    polygon: BaseGeometry | None,
    cfg: LabelOutlines,
) -> EventGeometry:
    """Split an accepted candidate into core, extension and trimmed parts.

    Args:
        candidate: Candidate outline.
        source: Candidate source (``di``, ``weak`` or ``chm``).
        geom_src: ``refined`` or ``cells``.
        routed: The routed verdicts.
        polygon: Model polygon in map coordinates, when one was asked for.
        cfg: Outline thresholds.
    """
    name = f"{source}_{geom_src}"
    crisp = is_crisp(routed, cfg)
    if not needs_polygon(routed):
        precise = (
            geom_src == "refined"
            and crisp
            and all(s == "over_75" for s in routed.affected_share)
        )
        return EventGeometry(
            candidate, _EMPTY, _EMPTY, name, "precise" if precise else "envelope", crisp
        )
    doubtful = (
        any(f in _BAD_FIT for f in routed.outline_fit)
        or "under_25" in routed.affected_share
    )
    if polygon_usable(polygon, candidate, cfg):
        extension = polygon.difference(candidate)
        if doubtful:
            core = candidate.intersection(polygon)
            trimmed = candidate.difference(polygon)
        else:
            core, trimmed = candidate, _EMPTY
        return EventGeometry(
            core, extension, trimmed, name + "+polygon", "envelope", crisp
        )
    if doubtful:
        # The models doubt the outline and there is nothing better: label none of it.
        return EventGeometry(_EMPTY, _EMPTY, candidate, name, "envelope", crisp)
    return EventGeometry(
        candidate, _EMPTY, _EMPTY, name, "envelope", crisp, wide_margin=True
    )
