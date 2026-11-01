"""csdv_core.review.mapview: the ipyleaflet map of the review app.

:class:`ReviewMap` owns one ipyleaflet map and changes its layers in place, so
moving from one polygon to the next sends only new tile URLs and a little
GeoJSON to the browser. The layers, from bottom to top, are:

1. Esri World Imagery as a basemap outside the NAIP coverage.
2. NAIP of the selected year, or a before/after swipe of two years.
3. An optional overlay (CHM, CHM change, onset, probability).
4. The 19.2 m objects (dashed, off by default).
5. The outlines of all nearby objects, coloured by onset year.
6. The polygon under review.
7. The drawing layer (Leaflet-Geoman) for redrawn outlines.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable

import geopandas as gpd
import ipyleaflet
import ipywidgets as w
import shapely
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from csdv_core.review.context import FOCUS_COLOUR
from csdv_core.review.imagery import ImagerySource

MAX_ZOOM = 21
#: Zoom up to which the server renders the local rasters (beyond, the browser
#: scales the tiles).
LOCAL_MAX_NATIVE_ZOOM = 19
ESRI_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/"
    "MapServer/tile/{z}/{y}/{x}"
)
DRAW_COLOUR = "#ff2bd6"
#: Attributes sent with each outline (for the click readout).
POPUP_COLUMNS = ["object_id", "onset_year", "interval", "area_ha", "geom_src", "growth"]

__all__ = ["ReviewMap", "to_geojson", "zoom_for_bounds"]


def to_geojson(gdf: gpd.GeoDataFrame, simplify_m: float = 0.6) -> dict:
    """WGS84 GeoJSON of a projected frame, simplified and rounded for the map."""
    if gdf.empty:
        return {"type": "FeatureCollection", "features": []}
    gdf = gdf[[c for c in POPUP_COLUMNS if c in gdf] + ["geometry"]].copy()
    if simplify_m and gdf.crs is not None and gdf.crs.is_projected:
        gdf["geometry"] = gdf.geometry.simplify(simplify_m)
    gdf = gdf.to_crs(4326)
    gdf["geometry"] = shapely.set_precision(gdf.geometry.values, 1e-6)
    for col in gdf.columns.drop("geometry"):
        if gdf[col].dtype == object or str(gdf[col].dtype).startswith("str"):
            gdf[col] = gdf[col].astype(str)
    return json.loads(gdf.to_json(drop_id=True))


def zoom_for_bounds(
    bounds: tuple[float, float, float, float],
    width_px: int = 900,
    height_px: int = 600,
    max_zoom: int = 19,
) -> int:
    """Largest Web Mercator zoom at which lon/lat bounds fit the map."""
    west, south, east, north = bounds

    def merc_y(lat: float) -> float:
        lat = max(min(lat, 85.0), -85.0)
        return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

    dx = max(east - west, 1e-9) / 360
    dy = max(merc_y(north) - merc_y(south), 1e-9) / (2 * math.pi)
    zx = math.log2(width_px / 256 / dx)
    zy = math.log2(height_px / 256 / dy)
    return int(max(3, min(max_zoom, math.floor(min(zx, zy)))))


def _tile_layer(url: str, max_native_zoom: int, **kw) -> ipyleaflet.TileLayer:
    return ipyleaflet.TileLayer(
        url=url,
        max_zoom=MAX_ZOOM,
        max_native_zoom=max_native_zoom,
        attribution="",
        **kw,
    )


class ReviewMap:
    """One map whose layers are swapped as the user moves around."""

    def __init__(self, height: str = "64vh") -> None:
        self.map = ipyleaflet.Map(
            center=(39.0, -96.0),
            zoom=4,
            max_zoom=MAX_ZOOM,
            scroll_wheel_zoom=True,
            attribution_control=False,
            layout=w.Layout(height=height, width="100%"),
        )
        self.map.layers = (_tile_layer(ESRI_URL, 19, name="Esri World Imagery"),)
        self.imagery = _tile_layer("", 19, name="NAIP")
        self.overlay = _tile_layer(
            "", LOCAL_MAX_NATIVE_ZOOM, name="Overlay", opacity=0.6
        )
        self.cells = ipyleaflet.GeoJSON(
            data=to_geojson(gpd.GeoDataFrame()),
            style={
                "color": "#ffffff",
                "weight": 1,
                "dashArray": "3 3",
                "fillOpacity": 0,
            },
            name="19.2 m objects",
        )
        self.outlines = ipyleaflet.GeoJSON(
            data=to_geojson(gpd.GeoDataFrame()),
            hover_style={"weight": 4},
            name="Detections",
        )
        self.focus = ipyleaflet.GeoJSON(
            data=to_geojson(gpd.GeoDataFrame()),
            style={"color": FOCUS_COLOUR, "weight": 3.5, "fillOpacity": 0},
            name="Under review",
        )
        self.draw = ipyleaflet.GeomanDrawControl(
            polygon={
                "pathOptions": {
                    "color": DRAW_COLOUR,
                    "weight": 2.5,
                    "fillOpacity": 0.05,
                }
            },
            polyline={},
            circlemarker={},
            circle={},
            marker={},
            rectangle={
                "pathOptions": {
                    "color": DRAW_COLOUR,
                    "weight": 2.5,
                    "fillOpacity": 0.05,
                }
            },
            drag=False,
            rotate=False,
            cut=True,
            edit=True,
            remove=True,
        )
        self._split: ipyleaflet.SplitMapControl | None = None
        self._outlines_on = True
        self._cells_on = False
        self._overlay_on = False
        self._click: Callable[[dict], None] | None = None
        self.outlines.on_click(self._on_outline_click)
        self.map.add(self.imagery)
        self.map.add(self.outlines)
        self.map.add(self.focus)
        self.map.add(self.draw)
        self.map.add(ipyleaflet.ScaleControl(position="bottomleft", imperial=False))

    # ---- view -----------------------------------------------------------

    def zoom_to(self, geometry_4326: BaseGeometry, pad: float = 2.5) -> None:
        """Centre the map on a polygon, zoomed so it fills about 1/pad of it."""
        west, south, east, north = geometry_4326.bounds
        cx, cy = (west + east) / 2, (south + north) / 2
        hw = max((east - west) / 2 * pad, 0.0015)
        hh = max((north - south) / 2 * pad, 0.001)
        self.map.center = (cy, cx)
        self.map.zoom = zoom_for_bounds((cx - hw, cy - hh, cx + hw, cy + hh))

    # ---- rasters --------------------------------------------------------

    def set_imagery(self, source: ImagerySource) -> None:
        self.clear_split()
        self.imagery.max_native_zoom = source.max_native_zoom
        self.imagery.url = source.url

    def set_split(self, left: ImagerySource, right: ImagerySource) -> None:
        """Swipe between two years (left before, right after)."""
        self.clear_split()
        if self.imagery in self.map.layers:
            self.map.remove(self.imagery)
        self._split = ipyleaflet.SplitMapControl(
            left_layer=_tile_layer(left.url, left.max_native_zoom),
            right_layer=_tile_layer(right.url, right.max_native_zoom),
        )
        self.map.add(self._split)

    def clear_split(self) -> None:
        if self._split is not None:
            self.map.remove(self._split)
            self._split = None
        if self.imagery not in self.map.layers:
            self.map.add(self.imagery, index=1)

    def set_overlay(self, url: str | None, opacity: float = 0.6) -> None:
        """Show a local raster overlay above the imagery, or remove it."""
        if url is None:
            if self._overlay_on:
                self.map.remove(self.overlay)
                self._overlay_on = False
            return
        self.overlay.url = url
        self.overlay.opacity = opacity
        if not self._overlay_on:
            index = (
                list(self.map.layers).index(self.imagery) + 1
                if (self.imagery in self.map.layers)
                else 1
            )
            self.map.add(self.overlay, index=index)
            self._overlay_on = True

    # ---- vectors --------------------------------------------------------

    def set_context(
        self,
        outlines: gpd.GeoDataFrame,
        cells: gpd.GeoDataFrame,
        focus: gpd.GeoDataFrame,
        colours: dict[int, str],
        *,
        simplify_m: float = 0.6,
    ) -> None:
        """Replace the nearby outlines, 19.2 m objects and the focus polygon."""

        def style(feature: dict) -> dict:
            year = feature["properties"].get("onset_year")
            colour = (
                colours.get(int(year), "#ffffff") if year is not None else "#ffffff"
            )
            return {"color": colour, "weight": 2, "opacity": 0.95, "fillOpacity": 0}

        self.outlines.style_callback = style
        self.outlines.data = to_geojson(outlines, simplify_m)
        self.cells.data = to_geojson(cells, 0)
        self.focus.data = to_geojson(focus, 0)

    def show_outlines(self, on: bool) -> None:
        for layer in (self.outlines, self.focus):
            if on and layer not in self.map.layers:
                self.map.add(layer)
            elif not on and layer in self.map.layers:
                self.map.remove(layer)
        self._outlines_on = on

    def show_cells(self, on: bool) -> None:
        if on and self.cells not in self.map.layers:
            index = (
                list(self.map.layers).index(self.outlines)
                if (self.outlines in self.map.layers)
                else len(self.map.layers)
            )
            self.map.add(self.cells, index=index)
        elif not on and self.cells in self.map.layers:
            self.map.remove(self.cells)
        self._cells_on = on

    def on_outline_click(self, callback: Callable[[dict], None] | None) -> None:
        """Call ``callback(properties)`` when an outline is clicked."""
        self._click = callback

    def _on_outline_click(self, **kwargs) -> None:
        if self._click is not None:
            self._click(kwargs.get("properties") or {})

    # ---- drawing --------------------------------------------------------

    def load_drawing(self, geometry_4326: BaseGeometry | None) -> None:
        """Put a (multi)polygon into the drawing layer for editing, or clear it."""
        if geometry_4326 is None or geometry_4326.is_empty:
            self.draw.data = []
            return
        parts = getattr(geometry_4326, "geoms", [geometry_4326])
        style = {"color": DRAW_COLOUR, "weight": 2.5, "fillOpacity": 0.05}
        self.draw.data = [
            {
                "type": "Feature",
                "properties": {"style": style},
                "geometry": shapely.geometry.mapping(part),
            }
            for part in parts
        ]

    def drawn_geometry(self) -> BaseGeometry | None:
        """Union of the polygons in the drawing layer (WGS84), or None."""
        polys = []
        for feature in self.draw.data or []:
            geom = feature.get("geometry")
            if geom and geom.get("type") in ("Polygon", "MultiPolygon"):
                polys.append(shape(geom))
        if not polys:
            return None
        return shapely.union_all([shapely.make_valid(p) for p in polys])
