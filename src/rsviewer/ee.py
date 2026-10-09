"""rsviewer.ee: Earth Engine images as map layers.

Earth Engine serves its own XYZ tiles, so an ``ee.Image`` becomes a
:class:`rsviewer.layers.TileLayerSpec` through ``getMapId`` with no local tile
server involved. ``earthengine-api`` is imported only when these functions are
called, so the rest of the package works without it. geemap is not needed, but
``viewer.map`` is a leafmap map, so geemap's tools can still be used on it.
"""

from __future__ import annotations

import logging
from typing import Any

from rsviewer.layers import Category, TileLayerSpec

logger = logging.getLogger(__name__)

_initialized = False


def init(project: str | None = None) -> None:
    """Initialize Earth Engine once per process.

    Args:
        project: Google Cloud project registered for Earth Engine.

    Raises:
        RuntimeError: Earth Engine is not authenticated. Run
            ``earthengine authenticate`` (or ``ee.Authenticate()``) once.
    """
    global _initialized
    if _initialized:
        return
    import ee

    try:
        ee.Initialize(project=project)
    except Exception as exc:  # noqa: BLE001  (ee raises several types here)
        raise RuntimeError(
            "Earth Engine could not initialize. Authenticate once with "
            "`earthengine authenticate`, then retry."
        ) from exc
    _initialized = True
    logger.info("Earth Engine initialized (project %s)", project)


def ee_layer(
    image: Any,
    vis: dict[str, Any],
    name: str,
    *,
    group: str = "Earth Engine",
    opacity: float = 1.0,
    visible: bool = False,
    legend: dict[Any, Category] | None = None,
) -> TileLayerSpec:
    """Tile layer spec for an ``ee.Image`` (or anything with ``getMapId``).

    Args:
        image: ``ee.Image``, ``ee.ImageCollection`` (mosaicked by EE) or
            ``ee.FeatureCollection``.
        vis: Visualization parameters, e.g. ``{"bands": ["R", "G", "B"],
            "min": 0, "max": 255}``.
        name: Layer name.
        group: Layer panel group.
        opacity: Initial opacity.
        visible: Whether the layer starts switched on.
        legend: Optional categorical legend.

    Returns:
        The tile layer spec.
    """
    map_id = image.getMapId(vis)
    url = map_id["tile_fetcher"].url_format
    return TileLayerSpec(
        name=name,
        url=url,
        attribution="Google Earth Engine",
        group=group,
        opacity=opacity,
        visible=visible,
        legend=legend,
    )
