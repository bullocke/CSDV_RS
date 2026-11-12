"""csdv_core.label.labels: label rasters of one window.

Inputs: a unit's cell layers, a window with its imagery stack, and the
candidates of the window with their routed verdicts and outlines.
Outputs: :class:`WindowLabels`, with one reason, agent and change raster per
interval on the window's 0.6 m grid.

The rules are evaluated on the window plus a collar, so a candidate just
outside the window still keeps its ring of unlabelled ground inside it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from shapely.geometry.base import BaseGeometry

from csdv_core.config._models import LabelConfig
from csdv_core.label.inputs import UnitInputs, fill_forest_dips, unstable_intervals
from csdv_core.label.lattice import (
    GSD,
    Bounds,
    Window,
    cells_to_pixels,
    grid_transform,
    pixel_slices,
)
from csdv_core.label.outlines import EventGeometry
from csdv_core.label.routing import LOSS_OUTCOMES, Routed
from csdv_core.label.rules import (
    IGNORE,
    BaseLayers,
    CandidateLayer,
    interval_reason,
    to_change,
)
from csdv_core.label.stacks import WindowStack
from csdv_core.label.verdicts import AGENT_CODES

logger = logging.getLogger(__name__)

__all__ = ["JudgedCandidate", "WindowLabels", "burn", "window_labels"]


@dataclass
class JudgedCandidate:
    """A candidate of a window after routing.

    Attributes:
        cand_id: Candidate ID.
        source: ``di``, ``weak`` or ``chm``.
        geom_src: ``refined`` or ``cells``.
        geometry: Full outline in the unit's CRS.
        flagged: Interval in which its source flagged it.
        routed: Routed verdicts (or the unjudged placeholder).
        event: Core, extension and trimmed parts, for a candidate with loss.
    """

    cand_id: str
    source: str
    geom_src: str
    geometry: BaseGeometry
    flagged: int | None
    routed: Routed
    event: EventGeometry | None = None


@dataclass
class WindowLabels:
    """Label rasters of a window.

    Attributes:
        bounds: Window extent.
        intervals: ``"<date>_<date>"`` of each interval.
        reason: ``(intervals, rows, cols)`` uint8 reason codes.
        agent: ``(intervals, rows, cols)`` uint8 agent codes, 255 none.
        change: ``(intervals, rows, cols)`` uint8: 0, 1, 2, 255.
        prob: ``(intervals, rows, cols)`` uint8 probability in percent, 255
            where not monitored.
        forest: ``(dates, rows, cols)`` uint8 forest mask (dips filled).
        unstable: Whether the unit's forest mask is unreliable, per interval.
    """

    bounds: Bounds
    intervals: list[str]
    reason: np.ndarray
    agent: np.ndarray
    change: np.ndarray
    prob: np.ndarray
    forest: np.ndarray
    unstable: list[bool]


def burn(geom: BaseGeometry | None, bounds: Bounds) -> np.ndarray:
    """Pixels of ``bounds`` whose centre lies inside ``geom``."""
    from rasterio.features import rasterize

    shape = (
        round((bounds[3] - bounds[1]) / GSD),
        round((bounds[2] - bounds[0]) / GSD),
    )
    if geom is None or geom.is_empty:
        return np.zeros(shape, dtype=bool)
    return rasterize(
        [(geom, 1)],
        out_shape=shape,
        transform=grid_transform(bounds),
        fill=0,
        dtype="uint8",
    ).astype(bool)


def _layer(cand: JudgedCandidate, k: int, masks: dict) -> CandidateLayer | None:
    outcome = cand.routed.outcomes[k] if k < len(cand.routed.outcomes) else ""
    if not outcome:
        return None
    layer = CandidateLayer(outcome=outcome, mask=masks[cand.cand_id, "mask"])
    if outcome in LOSS_OUTCOMES and cand.event is not None:
        layer.core = masks[cand.cand_id, "core"]
        layer.extension = masks[cand.cand_id, "extension"]
        layer.trimmed = masks[cand.cand_id, "trimmed"]
        layer.crisp = cand.event.crisp
        layer.precise = cand.event.quality == "precise"
        layer.wide_margin = cand.event.wide_margin
        layer.agent = AGENT_CODES.get(cand.routed.agent or "", IGNORE)
    return layer


def window_labels(
    unit: UnitInputs,
    window: Window,
    stack: WindowStack | None,
    candidates: list[JudgedCandidate],
    cfg: LabelConfig,
    *,
    dates: list[str] | None = None,
) -> WindowLabels:
    """Reason, agent and change rasters of every interval of a window.

    The labels need only the candidates and the cell layers. The imagery
    adds one thing: pixels without imagery on either date become ``no_data``.

    Args:
        unit: The window's unit, with its cell layers.
        window: The window.
        stack: Imagery of the window, or None when it is on another machine.
        candidates: Candidates of the window after routing.
        cfg: Label thresholds.
        dates: Acquisition dates of the window, needed when ``stack`` is None.

    Raises:
        ValueError: If the stack does not hold exactly the unit's years. The
            verdict states and the cell layers are indexed by those years.
    """
    if stack is not None:
        if stack.years != unit.years:
            raise ValueError(
                f"{window.window_id}: stack years {stack.years} differ from the "
                f"unit's {unit.years}"
            )
        dates = stack.dates
    if dates is None or len(dates) != len(unit.years):
        raise ValueError(
            f"{window.window_id}: one acquisition date is needed for each of "
            f"the unit's {len(unit.years)} years"
        )
    pad = cfg.chips.collar_px * GSD
    x0, y0, x1, y1 = window.bounds
    outer = (
        round((x0 - pad) / GSD) * GSD,
        round((y0 - pad) / GSD) * GSD,
        round((x1 + pad) / GSD) * GSD,
        round((y1 + pad) / GSD) * GSD,
    )
    forest = unit.layer("forest")
    if cfg.rules.fill_dips:
        forest = fill_forest_dips(forest)
    unstable = unstable_intervals(forest, cfg.rules.unstable_change)
    forest_px = cells_to_pixels(forest, unit.grid, outer)
    cover_px = cells_to_pixels(unit.layer("chm_cover"), unit.grid, outer)
    height_px = cells_to_pixels(unit.layer("chm_height"), unit.grid, outer)
    prob_px = cells_to_pixels(unit.layer("probability"), unit.grid, outer)
    shape = forest_px.shape[1:]
    imagery = (
        stack.crop(outer).any(axis=1)
        if stack is not None
        else np.ones((len(unit.years), *shape), dtype=bool)
    )

    masks: dict = {}
    for cand in candidates:
        masks[cand.cand_id, "mask"] = burn(cand.geometry, outer)
        if cand.event is not None:
            masks[cand.cand_id, "core"] = burn(cand.event.core, outer)
            masks[cand.cand_id, "extension"] = burn(cand.event.extension, outer)
            masks[cand.cand_id, "trimmed"] = burn(cand.event.trimmed, outer)

    rows, cols = pixel_slices(window.bounds, outer)
    n = len(unit.years) - 1
    reasons, agents, probs = [], [], []
    for k in range(n):
        base = BaseLayers(
            no_data=~imagery[k] | ~imagery[k + 1],
            forest_before=forest_px[k],
            forest_after=forest_px[k + 1],
            cover_before=cover_px[k],
            height_before=height_px[k],
            prob=prob_px[k + 1],
            unstable=unstable[k],
        )
        layers = [layer for c in candidates if (layer := _layer(c, k, masks))]
        reason, agent = interval_reason(base, layers, cfg.rules)
        reasons.append(reason[rows, cols])
        agents.append(agent[rows, cols])
        with np.errstate(invalid="ignore"):
            percent = np.where(
                np.isnan(prob_px[k + 1]), IGNORE, np.round(prob_px[k + 1] * 100)
            )
        probs.append(percent[rows, cols].astype(np.uint8))
    reason = np.stack(reasons)
    return WindowLabels(
        bounds=window.bounds,
        intervals=[f"{a}_{b}" for a, b in zip(dates[:-1], dates[1:], strict=True)],
        reason=reason,
        agent=np.stack(agents),
        change=to_change(reason, cfg.policy),
        prob=np.stack(probs),
        forest=forest_px[:, rows, cols].astype(np.uint8),
        unstable=unstable,
    )
