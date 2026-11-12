"""csdv_core.label.views: the image window shown to a vision model.

A :class:`View` is a square piece of ground drawn at a fixed number of
pixels. Models answer in coordinates from 0 to 1000 across the image, which
survives any resizing a provider applies, and the view maps them back to the
ground. :func:`hybrid_view` picks the view of a candidate with the rule
adopted after the pilot: keep 0.6 m resolution, zoom in on small candidates,
and widen the view only when a candidate does not fit.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry

from csdv_core.config._models import LabelViews
from csdv_core.label.lattice import GSD, Bounds

__all__ = ["View", "hybrid_view", "seeded_rng"]


def seeded_rng(key: str) -> np.random.Generator:
    """A generator fixed by ``key``, the same in every process."""
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


@dataclass(frozen=True)
class View:
    """A square window drawn at ``px`` pixels per side."""

    bounds: Bounds
    px: int

    @property
    def edge_m(self) -> float:
        return self.bounds[2] - self.bounds[0]

    @property
    def m_per_px(self) -> float:
        return self.edge_m / self.px

    def to_map(self, nx: float, ny: float) -> tuple[float, float]:
        """The 0-1000 image scale (x right, y down) as map coordinates."""
        return (
            self.bounds[0] + nx / 1000.0 * self.edge_m,
            self.bounds[3] - ny / 1000.0 * self.edge_m,
        )

    def to_px(self, x: float, y: float) -> tuple[float, float]:
        """Map coordinates as image pixels."""
        return (
            (x - self.bounds[0]) / self.m_per_px,
            (self.bounds[3] - y) / self.m_per_px,
        )

    def rings_px(self, geom: BaseGeometry) -> list[list[tuple[float, float]]]:
        """Every ring of a (multi)polygon in image pixels."""
        rings = []
        for poly in getattr(geom, "geoms", [geom]):
            if poly.is_empty or poly.geom_type != "Polygon":
                continue
            for ring in (poly.exterior, *poly.interiors):
                rings.append([self.to_px(x, y) for x, y in ring.coords])
        return rings

    def polygon_to_map(self, vertices: list) -> Polygon:
        """Vertices on the 0-1000 scale as a map polygon."""
        return Polygon([self.to_map(float(x), float(y)) for x, y in vertices])


def _snap(value: float) -> float:
    return round(value / GSD) * GSD


def _inside(view: Bounds, limits: Bounds) -> Bounds:
    """Translate a square so it lies inside ``limits`` (no resize)."""
    dx = max(limits[0] - view[0], 0.0) + min(limits[2] - view[2], 0.0)
    dy = max(limits[1] - view[1], 0.0) + min(limits[3] - view[3], 0.0)
    dx, dy = _snap(dx), _snap(dy)
    return (view[0] + dx, view[1] + dy, view[2] + dx, view[3] + dy)


def hybrid_view(
    geom: BaseGeometry, stack_bounds: Bounds, cfg: LabelViews, key: str = ""
) -> View:
    """View of a candidate inside the imagery kept for its window.

    1. Small candidate (``zoom`` times its longer side is under
       ``detail_m``): a view of that size, not below ``min_m``.
    2. Candidate that fits ``detail_m`` with the margin: the standard view.
    3. Larger candidate: ``margin`` times its longer side, drawn at up to
       ``max_px`` pixels, so resolution stays at 0.6 m up to 768 m.

    The view is never larger than the stack and is shifted to lie inside it.
    It is offset from the candidate's centre by a random amount fixed by
    ``key``, so the candidate is not always in the middle of the image.

    Args:
        geom: Candidate outline, or its visible part, in the stack's CRS.
        stack_bounds: Square extent of the window's stack.
        cfg: View thresholds.
        key: Seed of the offset (the candidate ID).
    """
    seen = geom.intersection(box(*stack_bounds))
    minx, miny, maxx, maxy = (seen if not seen.is_empty else geom).bounds
    extent = max(maxx - minx, maxy - miny)
    stack_edge = stack_bounds[2] - stack_bounds[0]
    px = cfg.px
    if cfg.zoom * extent < cfg.detail_m:
        edge = max(cfg.zoom * extent, cfg.min_m, cfg.margin * extent)
    elif cfg.margin * extent <= cfg.detail_m:
        edge = cfg.detail_m
    else:
        edge = cfg.margin * extent
    edge = _snap(min(edge, stack_edge))
    if edge > cfg.detail_m:
        px = int(min(cfg.max_px, round(edge / GSD)))
    room = max(edge - extent, 0.0) / 4
    dx, dy = seeded_rng("view" + key).uniform(-room, room, 2) if room else (0.0, 0.0)
    cx, cy = (minx + maxx) / 2 + dx, (miny + maxy) / 2 + dy
    x0, y0 = _snap(cx - edge / 2), _snap(cy - edge / 2)
    return View(_inside((x0, y0, x0 + edge, y0 + edge), stack_bounds), px)
