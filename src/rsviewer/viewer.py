"""rsviewer.viewer: the map plus side panel that projects build on.

:class:`Viewer` wraps a leafmap (ipyleaflet) map and keeps a registry of named
layers built from :mod:`rsviewer.layers` specs. Layers can be added, replaced
in place (keeping their draw order), hidden or removed by name, which is what
interactive selectors need. The side panel holds a basemap picker, the layer
panel, an info readout and any panels the project adds. Click handlers get the
clicked longitude and latitude. By default a click also reads every visible
raster at that point.

Typical use in a ``# %%`` cell::

    v = Viewer(title="Site A")
    v.add(RasterLayer("NAIP 2018", path=cog, style=RasterStyle(bands=(1, 2, 3))))
    v.zoom_to("NAIP 2018")
    v.show()
"""

from __future__ import annotations

import html
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ipyleaflet
import ipywidgets as w
import leafmap.leafmap as leafmap
import numpy as np

from rsviewer import tiles
from rsviewer.layers import AnyLayer, RasterLayer, TileLayerSpec, VectorLayer
from rsviewer.panels import PANEL_WIDTH, InfoPanel, LayerPanel
from rsviewer.query import sample_raster

logger = logging.getLogger(__name__)

#: Basemap name to XYZ URL. ``None`` draws no basemap.
BASEMAPS: dict[str, str | None] = {
    "Esri World Imagery": "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "Google Satellite": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    "USGS Imagery": "https://basemap.nationalmap.gov/arcgis/rest/services/"
    "USGSImageryOnly/MapServer/tile/{z}/{y}/{x}",
    "OpenStreetMap": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "Esri Topo": "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
    "None": None,
}

ClickHandler = Callable[[float, float], None]

#: ipyleaflet's own Map methods. leafmap's ``add`` rebuilds a hidden layer
#: manager widget on every call and never closes the old one, which creates
#: hundreds of widgets for a few dozen layers and slows the first display.
_LeafletMap = ipyleaflet.Map


@dataclass
class _Entry:
    spec: AnyLayer
    layer: ipyleaflet.Layer
    path: Path | None = None
    bounds: tuple[float, float, float, float] | None = None  # W, S, E, N
    pending: dict | None = None  # GeoJSON not yet sent (hidden vector layer)


class _PanelLogHandler(logging.Handler):
    """Shows warnings from callbacks in the side panel, where they are visible."""

    def __init__(self, target: w.HTML) -> None:
        super().__init__(level=logging.WARNING)
        self.target = target

    def emit(self, record: logging.LogRecord) -> None:
        msg = html.escape(self.format(record))
        color = "#b00" if record.levelno >= logging.ERROR else "#a60"
        self.target.value = f"<span style='color:{color};font-size:11px'>{msg}</span>"


class Viewer:
    """Interactive map with a layer registry and a side panel.

    Args:
        title: Heading shown above the side panel.
        center: Initial ``(lat, lon)``.
        zoom: Initial zoom.
        height: Map height (CSS).
        basemap: Initial basemap, a key of :data:`BASEMAPS`.
        inspect: Read visible rasters at each clicked point into the info panel.
    """

    def __init__(
        self,
        title: str = "",
        center: tuple[float, float] = (39.0, -86.0),
        zoom: int = 13,
        height: str = "680px",
        basemap: str = "Esri World Imagery",
        inspect: bool = True,
    ) -> None:
        self.map = leafmap.Map(
            center=list(center),
            zoom=zoom,
            height=height,
            draw_control=False,
            measure_control=False,
            toolbar_control=False,
            scale_control=True,
        )
        self.map.layout.flex = "1 1 auto"
        self._base_name = ""
        self.set_basemap(basemap)
        self._entries: dict[str, _Entry] = {}
        self._click_handlers: list[ClickHandler] = []
        self._split: ipyleaflet.SplitMapControl | None = None
        self.inspect = inspect
        self.marker = ipyleaflet.CircleMarker(
            location=list(center), radius=6, color="#00e5ff", fill_opacity=0, weight=2
        )
        self._marker_on = False

        self.layers = LayerPanel(self.set_visible, self.set_opacity)
        self.info = InfoPanel()
        self.messages = w.HTML("")
        self._log_handler = _PanelLogHandler(self.messages)
        logging.getLogger().addHandler(self._log_handler)
        base_pick = w.Dropdown(
            options=list(BASEMAPS),
            value=basemap,
            description="Basemap",
            layout=w.Layout(width="300px"),
        )
        base_pick.observe(lambda c: self.set_basemap(c["new"]), "value")
        layers_tab = w.VBox([base_pick, self.layers])
        self.tabs = w.Tab(children=[layers_tab, self.info])
        self.tabs.set_title(0, "Layers")
        self.tabs.set_title(1, "Info")
        head = w.HTML(f"<h3 style='margin:4px 0'>{html.escape(title)}</h3>")
        self.sidebar = w.VBox(
            [head, self.tabs, self.messages],
            layout=w.Layout(width=f"{PANEL_WIDTH + 20}px", overflow_y="auto"),
        )
        self.widget = w.HBox(
            [self.map, self.sidebar],
            layout=w.Layout(width="100%", align_items="stretch"),
        )
        self.map.on_interaction(self._interaction)

    # ------------------------------------------------------------------
    # Basemaps
    # ------------------------------------------------------------------

    def set_basemap(self, name: str) -> None:
        """Switch the basemap (a key of :data:`BASEMAPS`)."""
        if name not in BASEMAPS:
            raise ValueError(f"unknown basemap {name!r}, choose from {list(BASEMAPS)}")
        url = BASEMAPS[name]
        new = ipyleaflet.TileLayer(
            url=url or "",
            name="basemap",
            base=True,
            max_zoom=22,
            visible=url is not None,
        )
        if url and "arcgisonline" in url:
            new.max_native_zoom = 19
        old = self.map.layers[0] if self.map.layers else None
        if old is not None and (old.base or old.name in ("OpenStreetMap", "basemap")):
            _LeafletMap.substitute(self.map, old, new)
        else:
            self.map.layers = (new, *self.map.layers)
        self._base_name = name

    # ------------------------------------------------------------------
    # Layer registry
    # ------------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        """Names of the registered layers, in draw order."""
        return list(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def spec(self, name: str) -> AnyLayer:
        """The spec a layer was built from."""
        return self._entries[name].spec

    def _build(self, spec: AnyLayer) -> _Entry:
        if isinstance(spec, RasterLayer):
            path = spec.resolve()
            layer = tiles.raster_tile_layer(spec, path)
            s, n, w_, e = tiles.tile_client(str(path)).bounds()
            return _Entry(spec, layer, path, (w_, s, e, n))
        if isinstance(spec, VectorLayer):
            gdf = tiles.read_vector(spec)
            data = tiles.geojson(gdf)
            # Hidden layers start empty, so opening the viewer does not send
            # every polygon to the browser. The data follows when switched on.
            layer = tiles.vector_layer(spec, data if spec.visible else tiles.EMPTY_FC)
            layer.on_click(lambda feature, **kw: self._feature_clicked(spec, feature))
            bounds = tuple(float(b) for b in gdf.total_bounds) if len(gdf) else None
            return _Entry(
                spec, layer, spec.path, bounds, None if spec.visible else data
            )
        if isinstance(spec, TileLayerSpec):
            return _Entry(spec, tiles.url_tile_layer(spec))
        raise TypeError(f"unsupported layer spec {type(spec).__name__}")

    def _legend(self, spec: AnyLayer) -> str:
        if isinstance(spec, RasterLayer):
            return tiles.raster_legend(spec.style)
        if isinstance(spec, VectorLayer):
            return tiles.vector_legend(spec.style)
        if spec.legend:
            return tiles._swatches(spec.legend)
        return ""

    def add(self, spec: AnyLayer) -> bool:
        """Add a layer, or replace an existing layer with the same name.

        Missing files are logged and skipped, so one absent output does not
        stop the rest of a catalogue from loading.

        Returns:
            Whether the layer was added.
        """
        if spec.name in self._entries:
            return self.replace(spec.name, spec)
        try:
            entry = self._build(spec)
        except FileNotFoundError as exc:
            logger.warning("Skipped layer: %s", exc)
            return False
        self._entries[spec.name] = entry
        if isinstance(spec, RasterLayer):
            # Rasters go below every vector layer so outlines stay on top.
            vec = [
                e.layer
                for e in self._entries.values()
                if isinstance(e.spec, VectorLayer)
            ]
            layers = list(self.map.layers)
            idx = min(
                (layers.index(v) for v in vec if v in layers), default=len(layers)
            )
            layers.insert(idx, entry.layer)
            self.map.layers = tuple(layers)
        else:
            _LeafletMap.add(self.map, entry.layer)
        self.layers.add(
            spec.name,
            spec.group,
            spec.visible,
            spec.opacity,
            self._legend(spec),
            spec.description,
        )
        return True

    def add_many(self, specs: Sequence[AnyLayer]) -> None:
        """Add several layers in order."""
        for spec in specs:
            self.add(spec)

    def replace(self, name: str, spec: AnyLayer) -> bool:
        """Swap the layer called ``name`` for ``spec``, keeping its draw order.

        The new layer keeps the old layer's visibility and opacity, so a
        selector change does not reset what the user switched on.
        """
        old = self._entries.get(name)
        if old is None:
            return self.add(spec)
        spec.visible = old.layer.visible
        spec.opacity = old.spec.opacity
        try:
            entry = self._build(spec)
        except FileNotFoundError as exc:
            logger.warning("Kept old layer: %s", exc)
            return False
        if self._split is not None and old.layer in (
            self._split.left_layer,
            self._split.right_layer,
        ):
            self.unsplit()
        _LeafletMap.substitute(self.map, old.layer, entry.layer)
        del self._entries[name]
        self._entries[spec.name] = entry
        self.layers.remove(name)
        self.layers.add(
            spec.name,
            spec.group,
            spec.visible,
            spec.opacity,
            self._legend(spec),
            spec.description,
        )
        return True

    def remove(self, name: str) -> None:
        """Remove a layer by name (no error if absent)."""
        entry = self._entries.pop(name, None)
        if entry is None:
            return
        if entry.layer in self.map.layers:
            _LeafletMap.remove(self.map, entry.layer)
        self.layers.remove(name)

    def set_visible(self, name: str, visible: bool) -> None:
        """Switch a layer on or off."""
        entry = self._entries.get(name)
        if entry is not None:
            if visible and entry.pending is not None:
                entry.layer.data = entry.pending
                entry.pending = None
            entry.layer.visible = visible
            entry.spec.visible = visible
            self.layers.set_checked(name, visible)

    def set_opacity(self, name: str, opacity: float) -> None:
        """Set a layer's opacity, 0 to 1."""
        entry = self._entries.get(name)
        if entry is None:
            return
        entry.spec.opacity = opacity
        if isinstance(entry.spec, VectorLayer):
            tiles.restyle_vector(entry.layer, entry.spec, opacity)
        else:
            entry.layer.opacity = opacity

    def zoom_to(self, name: str) -> None:
        """Fit the map to a layer's extent."""
        b = self._entries[name].bounds
        if b is not None:
            self.map.fit_bounds([[b[1], b[0]], [b[3], b[2]]])

    # ------------------------------------------------------------------
    # Panels and comparison
    # ------------------------------------------------------------------

    def add_panel(self, title: str, widget: w.Widget, select: bool = False) -> None:
        """Add a tab to the side panel (before the Info tab)."""
        kids = list(self.tabs.children)
        titles = [self.tabs.get_title(i) for i in range(len(kids))]
        kids.insert(len(kids) - 1, widget)
        titles.insert(len(titles) - 1, title)
        self.tabs.children = kids
        for i, t in enumerate(titles):
            self.tabs.set_title(i, t)
        if select:
            self.tabs.selected_index = titles.index(title)

    def split(self, left: str, right: str) -> None:
        """Show two layers side by side with a draggable divider."""
        self.unsplit()
        lft, rgt = self._entries[left].layer, self._entries[right].layer
        self.set_visible(left, True)
        self.set_visible(right, True)
        self._split = ipyleaflet.SplitMapControl(left_layer=lft, right_layer=rgt)
        _LeafletMap.add(self.map, self._split)

    def unsplit(self) -> None:
        """Remove the side-by-side divider."""
        if self._split is not None:
            if self._split in self.map.controls:
                _LeafletMap.remove(self.map, self._split)
            self._split = None

    # ------------------------------------------------------------------
    # Clicks
    # ------------------------------------------------------------------

    def on_click(self, handler: ClickHandler) -> None:
        """Register ``handler(lon, lat)`` for map clicks."""
        self._click_handlers.append(handler)

    def _interaction(self, **kw: Any) -> None:
        if kw.get("type") != "click":
            return
        lat, lon = kw["coordinates"]
        self.marker.location = [lat, lon]
        if not self._marker_on:
            _LeafletMap.add(self.map, self.marker)
            self._marker_on = True
        self.messages.value = ""
        if self.inspect:
            self._inspect(lon, lat)
        for handler in self._click_handlers:
            try:
                handler(lon, lat)
            except Exception:  # noqa: BLE001  (show any handler error in the panel)
                logger.exception(
                    "Click handler %s failed", getattr(handler, "__name__", handler)
                )

    def _inspect(self, lon: float, lat: float) -> None:
        rows: list[tuple[str, Any]] = [("lat, lon", f"{lat:.6f}, {lon:.6f}")]
        for name, e in self._entries.items():
            if not (isinstance(e.spec, RasterLayer) and e.layer.visible and e.path):
                continue
            st = e.spec.style
            smp = sample_raster(e.path, lon, lat, scale=st.scale)
            if smp is None:
                continue
            vals = smp.values[[b - 1 for b in st.bands]]
            if st.categories and np.isfinite(vals[0]):
                cat = st.categories.get(int(round(vals[0] / st.scale)))
                rows.append((name, cat.label if cat else f"{vals[0]:.4g}"))
            else:
                txt = ", ".join(
                    "NaN" if not np.isfinite(v) else f"{v:.4g}" for v in vals
                )
                rows.append((name, f"{txt} {st.units}".strip()))
        self.info.show_table("Pixel values", rows)

    def _feature_clicked(self, spec: VectorLayer, feature: dict) -> None:
        props = feature.get("properties", {})
        keys = spec.popup if spec.popup is not None else list(props)
        self.info.show_table(spec.name, [(k, props.get(k, "")) for k in keys])
        self.tabs.selected_index = len(self.tabs.children) - 1

    # ------------------------------------------------------------------

    def show(self) -> w.HBox:
        """The full widget. Return it as the last expression of a cell."""
        return self.widget

    def close(self) -> None:
        """Detach the panel log handler (call before discarding the viewer)."""
        logging.getLogger().removeHandler(self._log_handler)
