"""csdv_core.label.fire: settle the agent of a loss inside a mapped fire.

Inputs: fire perimeters (a GeoPackage with an ignition date, ``ig_date``,
such as MTBS) and the routed verdicts of a window. Output: the same verdicts,
where a disputed agent is set to fire.

The two vision models often see a loss inside a burn and disagree about its
cause: one says fire, the other insects and disease, clearcut, partial
harvest, wind or clearing. Such a loss is labelled loss, but without an
agreed agent it adds nothing to the agent head. When the candidate lies
mostly inside a perimeter whose fire started within the interval of its
first loss, the model that said fire is taken. A loss whose agent both
models agree on is never changed, and nothing changes the loss label itself.

Checked on the scale test (2026-11-16): 738 of 744 agreed fire events lie
inside an MTBS perimeter of their interval, and 165 events inside such
perimeters had no agreed agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry

from csdv_core.label.routing import Routed
from csdv_core.label.verdicts import AGENT_OF

logger = logging.getLogger(__name__)

#: Value of ``Routed.agent_source`` for an agent taken from a fire perimeter.
FROM_PERIMETER = "fire_perimeter"

__all__ = ["FROM_PERIMETER", "FirePerimeters", "settle_fire_agent"]


@dataclass
class FirePerimeters:
    """Fire perimeters with their ignition dates, by CRS on demand.

    Attributes:
        frame: Perimeters with a datetime ``ig_date``.
        min_share: Share of a candidate that has to lie inside the
            perimeters of its interval.
    """

    frame: gpd.GeoDataFrame
    min_share: float = 0.5

    def __post_init__(self) -> None:
        self._by_epsg: dict[int, gpd.GeoDataFrame] = {}

    @classmethod
    def read(cls, path: Path, min_share: float = 0.5) -> FirePerimeters:
        """Read a GeoPackage or shapefile with an ``ig_date`` column."""
        frame = gpd.read_file(path)
        if "ig_date" not in frame:
            raise ValueError(f"{path} has no ig_date column")
        frame["ig_date"] = pd.to_datetime(frame["ig_date"])
        frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty]
        logger.info("%d fire perimeters read from %s", len(frame), path)
        return cls(frame[["ig_date", "geometry"]], min_share)

    def in_crs(self, epsg: int) -> gpd.GeoDataFrame:
        if epsg not in self._by_epsg:
            self._by_epsg[epsg] = self.frame.to_crs(epsg)
        return self._by_epsg[epsg]

    def share_inside(
        self, geom: BaseGeometry, epsg: int, start: str, end: str
    ) -> float:
        """Share of ``geom`` inside perimeters that ignited after ``start``
        and on or before ``end``."""
        if geom is None or geom.is_empty or geom.area == 0:
            return 0.0
        frame = self.in_crs(epsg)
        when = (frame["ig_date"] > pd.Timestamp(start)) & (
            frame["ig_date"] <= pd.Timestamp(end)
        )
        near = frame[when & frame.intersects(geom)]
        if near.empty:
            return 0.0
        return float(geom.intersection(near.union_all()).area / geom.area)


def settle_fire_agent(
    routed: Routed,
    geom: BaseGeometry,
    epsg: int,
    dates: list[str],
    perimeters: FirePerimeters,
) -> Routed:
    """Agent fire for a loss the models disagree on, if one said fire and
    the candidate lies inside a perimeter of its first loss interval.

    Args:
        routed: Routed verdicts of one candidate.
        geom: The candidate's outline in ``epsg``.
        epsg: CRS of ``geom``.
        dates: Acquisition dates of the window, one per image.
        perimeters: The fire perimeters.

    Returns:
        ``routed`` unchanged, or a copy with ``agent="fire"`` and
        ``agent_source`` set to :data:`FROM_PERIMETER`. The change type stays
        unset, because the models still disagree about it.
    """
    k = routed.first_loss
    if k is None or routed.agent is not None or k + 1 >= len(dates):
        return routed
    agents = {AGENT_OF.get(t or "") for t in routed.change_types}
    if "fire" not in agents:
        return routed
    share = perimeters.share_inside(geom, epsg, dates[k], dates[k + 1])
    if share < perimeters.min_share:
        return routed
    return replace(routed, agent="fire", agent_source=FROM_PERIMETER)
