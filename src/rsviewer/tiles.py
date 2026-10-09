"""rsviewer.tiles: turn layer specs into ipyleaflet layers.

Rasters are served by localtileserver, which starts one local HTTP tile server
per Python process and renders PNG tiles on request with rio-tiler. Vector
files are reprojected to WGS84 and drawn as ipyleaflet GeoJSON. This module
also builds the HTML legend for each style.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import geopandas as gpd
import ipyleaflet
import numpy as np
import pandas as pd
import shapely
from localtileserver import TileClient, get_leaflet_tile_layer
from matplotlib.colors import ListedColormap, to_rgba
from rio_tiler.colormap import cmap as rio_cmaps

from rsviewer.layers import (
    Category,
    RasterLayer,
    RasterStyle,
    TileLayerSpec,
    VectorLayer,
    VectorStyle,
)

logger = logging.getLogger(__name__)

#: Max tile zoom. Leaflet scales the deepest native tiles beyond it.
MAX_ZOOM = 22


@lru_cache(maxsize=256)
def tile_client(path: str) -> TileClient:
    """A cached localtileserver client for ``path``."""
    return TileClient(path)


def _lookup_colormap(categories: dict[int, Category]) -> ListedColormap:
    """256-entry colormap with each category at its pixel value.

    The tile server applies it to uint8 pixels without rescaling, so pixel
    value ``k`` gets entry ``k`` exactly.
    """
    colors = [(0.5, 0.5, 0.5, 1.0)] * 256
    for value, cat in categories.items():
        colors[value] = to_rgba(cat.color)
    return ListedColormap(colors)


def raster_tile_layer(spec: RasterLayer, path: Path) -> ipyleaflet.TileLayer:
    """ipyleaflet tile layer for a raster spec whose file is ``path``."""
    st = spec.style
    kw: dict[str, Any] = dict(
        indexes=list(st.bands),
        name=spec.name,
        opacity=spec.opacity,
        max_zoom=MAX_ZOOM,
        max_native_zoom=MAX_ZOOM,
    )
    if st.nodata is not None:
        kw["nodata"] = st.nodata
    if st.categories:
        kw["colormap"] = _lookup_colormap(st.categories)
    else:
        if st.colormap is not None and len(st.bands) == 1:
            kw["colormap"] = st.colormap
        if st.vmin is not None:
            kw["vmin"] = st.vmin
        if st.vmax is not None:
            kw["vmax"] = st.vmax
    layer = get_leaflet_tile_layer(tile_client(str(path)), **kw)
    layer.visible = spec.visible
    return layer


def url_tile_layer(spec: TileLayerSpec) -> ipyleaflet.TileLayer:
    """ipyleaflet tile layer for an XYZ URL."""
    return ipyleaflet.TileLayer(
        url=spec.url,
        name=spec.name,
        attribution=spec.attribution,
        opacity=spec.opacity,
        visible=spec.visible,
        max_zoom=MAX_ZOOM,
    )


def read_vector(spec: VectorLayer) -> gpd.GeoDataFrame:
    """Read a vector spec as WGS84, with popup-safe attribute types."""
    if not spec.path.exists():
        raise FileNotFoundError(f"{spec.name}: vector file not found: {spec.path}")
    gdf = gpd.read_file(spec.path, layer=spec.layer)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if spec.simplify_m and gdf.crs is not None and gdf.crs.is_projected:
        gdf = gdf.set_geometry(gdf.geometry.simplify(spec.simplify_m))
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    # About 0.1 m at mid latitudes. Halves the GeoJSON sent to the map.
    gdf = gdf.set_geometry(shapely.set_precision(gdf.geometry.values, 1e-6))
    for col in gdf.columns.drop(gdf.geometry.name):
        dt = gdf[col].dtype
        # Numbers and booleans serialize as they are (NaN becomes null).
        # Everything else (strings, dates, categoricals) becomes plain text.
        if not (pd.api.types.is_bool_dtype(dt) or pd.api.types.is_numeric_dtype(dt)):
            gdf[col] = gdf[col].astype(str).where(gdf[col].notna(), "")
    return gdf


def _feature_style(st: VectorStyle, props: dict[str, Any], opacity: float) -> dict:
    color = st.color
    if st.by is not None and st.categories:
        cat = st.categories.get(str(props.get(st.by)))
        if cat is not None:
            color = cat.color
    fill = st.fill_color if st.fill_color is not None else color
    return dict(
        color=color,
        weight=st.weight,
        opacity=opacity,
        fillColor=fill,
        fillOpacity=(st.fill_opacity if st.fill_color is not None or st.by else 0.0)
        * opacity,
    )


#: GeoJSON for a layer whose features have not been sent yet.
EMPTY_FC: dict = {"type": "FeatureCollection", "features": []}


def geojson(gdf: gpd.GeoDataFrame) -> dict:
    """GeoJSON dict of a WGS84 frame."""
    return json.loads(gdf.to_json(drop_id=True))


def vector_layer(spec: VectorLayer, data: dict) -> ipyleaflet.GeoJSON:
    """ipyleaflet GeoJSON layer for GeoJSON ``data`` (WGS84)."""
    st, op = spec.style, spec.opacity
    layer = ipyleaflet.GeoJSON(
        data=data,
        name=spec.name,
        style_callback=lambda feature: _feature_style(st, feature["properties"], op),
        hover_style={"weight": st.weight + 2, "fillOpacity": 0.35},
        visible=spec.visible,
    )
    return layer


def restyle_vector(
    layer: ipyleaflet.GeoJSON, spec: VectorLayer, opacity: float
) -> None:
    """Apply a new opacity to a GeoJSON layer drawn by :func:`vector_layer`."""
    st = spec.style
    layer.style_callback = lambda feature: _feature_style(
        st, feature["properties"], opacity
    )


# ----------------------------------------------------------------------------
# Legends
# ----------------------------------------------------------------------------


def _swatches(categories: dict[Any, Category], square: bool = True) -> str:
    rows = []
    seen: set[tuple[str, str]] = set()
    for cat in categories.values():
        if (cat.label, cat.color) in seen:  # several values may share one class
            continue
        seen.add((cat.label, cat.color))
        shape = "" if square else "border-radius:2px;background:none;"
        box = (
            f"<span style='display:inline-block;width:12px;height:12px;"
            f"background:{cat.color};border:2px solid {cat.color};{shape}"
            f"margin-right:6px;vertical-align:middle'></span>"
        )
        rows.append(f"<div style='line-height:16px'>{box}{cat.label}</div>")
    return "".join(rows)


def _gradient(name: str) -> str:
    cm = rio_cmaps.get(name)
    stops = []
    for i in range(0, 256, 32):
        r, g, b, _ = cm[i]
        stops.append(f"rgb({r},{g},{b}) {100 * i / 255:.0f}%")
    r, g, b, _ = cm[255]
    stops.append(f"rgb({r},{g},{b}) 100%")
    return f"linear-gradient(to right, {', '.join(stops)})"


def _fmt(v: float) -> str:
    return f"{v:.3g}"


def raster_legend(style: RasterStyle) -> str:
    """HTML legend body for a raster style ('' for RGB)."""
    if style.categories:
        return _swatches(style.categories)
    if style.is_rgb or style.vmin is None or style.vmax is None:
        return ""
    lo = float(np.atleast_1d(style.vmin)[0]) * style.scale
    hi = float(np.atleast_1d(style.vmax)[0]) * style.scale
    bar = (
        f"<div style='height:10px;width:180px;background:"
        f"{_gradient(style.colormap or 'gray')}'></div>"
    )
    unit = f" {style.units}" if style.units else ""
    return (
        f"{bar}<div style='width:180px;display:flex;justify-content:space-between;"
        f"font-size:11px'><span>{_fmt(lo)}</span><span>{_fmt(hi)}{unit}</span></div>"
    )


def vector_legend(style: VectorStyle) -> str:
    """HTML legend body for a vector style."""
    if style.by and style.categories:
        return _swatches(style.categories, square=False)
    return _swatches({0: Category("features", style.color)}, square=False)
