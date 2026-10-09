"""rsviewer: interactive remote sensing map viewer for VS Code and Jupyter.

A small layer on top of leafmap (ipyleaflet) and localtileserver for looking
at project rasters and vectors next to a satellite basemap, switching layers
from selectors and plotting values at clicked points. It runs in the VS Code
Interactive Window from ``# %%`` cells or in any Jupyter front end. See
``README.md`` in this folder.

This package must not import project code (e.g. ``csdv_core``), so that it can
move to its own repository unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rsviewer.config import Catalog, load_catalog
from rsviewer.layers import (
    Category,
    RasterLayer,
    RasterStyle,
    TileLayerSpec,
    VectorLayer,
    VectorStyle,
)
from rsviewer.panels import InfoPanel, LayerPanel, PlotPanel, SelectorPanel
from rsviewer.viewer import BASEMAPS, Viewer

__version__ = "0.1.0"

__all__ = [
    "BASEMAPS",
    "Catalog",
    "Category",
    "InfoPanel",
    "LayerPanel",
    "PlotPanel",
    "RasterLayer",
    "RasterStyle",
    "SelectorPanel",
    "TileLayerSpec",
    "VectorLayer",
    "VectorStyle",
    "Viewer",
    "from_catalog",
    "load_catalog",
]


def from_catalog(
    path: Path | str, variables: dict[str, Any] | None = None, **viewer_kw: Any
) -> Viewer:
    """Build a :class:`Viewer` from a YAML catalogue and add its layers.

    Args:
        path: Catalogue YAML.
        variables: Extra ``{name}`` variables for the catalogue.
        **viewer_kw: Passed to :class:`Viewer`, overriding catalogue settings.

    Returns:
        The viewer. Call ``.show()`` to display it.
    """
    cat = load_catalog(path, variables)
    kw = dict(title=cat.title, zoom=cat.zoom, basemap=cat.basemap)
    if cat.center:
        kw["center"] = cat.center
    kw.update(viewer_kw)
    v = Viewer(**kw)
    v.add_many(cat.layers)
    if not cat.center:
        for name in v.names:
            if v._entries[name].bounds is not None:
                v.zoom_to(name)
                break
    return v
