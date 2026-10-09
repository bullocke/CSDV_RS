"""rsviewer.layers: layer and style definitions.

Layers are plain dataclasses that say what to show and how to style it. They
hold no map state, so a project can build them in a catalogue (YAML or Python)
and hand them to :class:`rsviewer.viewer.Viewer`, which turns them into
ipyleaflet layers. Raster layers point at a GeoTIFF (ideally a COG) or at a
callable that produces one on demand, which is how cached, derived layers are
built only when first shown.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: A colour as a CSS hex string, e.g. ``"#1b9e77"``.
Color = str


@dataclass(frozen=True)
class Category:
    """One class of a categorical raster or vector style."""

    label: str
    color: Color


@dataclass(frozen=True)
class RasterStyle:
    """How a raster is rendered.

    Args:
        bands: 1-based band indexes. One band is colour-mapped, three are RGB.
        colormap: rio-tiler colormap name (lower case, e.g. ``"viridis"``,
            ``"rdbu"``) for a single continuous band.
        vmin: Low end of the stretch, in raw pixel units. A list gives one value
            per band for RGB.
        vmax: High end of the stretch, in raw pixel units.
        nodata: Override for the file's nodata value. Pixels with this value are
            transparent.
        categories: For a categorical uint8 band, pixel value to class. Pixels
            are drawn with an exact lookup, no stretching. Values without a
            class should be nodata, since the tile server cannot make a single
            value transparent.
        scale: Multiplier from raw pixel values to display units, used for the
            legend and for point queries (e.g. 0.01 for int16 z x 100).
        units: Display units for the legend and point queries.
    """

    bands: tuple[int, ...] = (1,)
    colormap: str | None = None
    vmin: float | list[float] | None = None
    vmax: float | list[float] | None = None
    nodata: float | None = None
    categories: dict[int, Category] | None = None
    scale: float = 1.0
    units: str = ""

    def __post_init__(self) -> None:
        if not self.bands:
            raise ValueError("bands must hold at least one band index")
        if self.categories and len(self.bands) != 1:
            raise ValueError("a categorical style needs exactly one band")
        if self.categories and any(not 0 <= k <= 255 for k in self.categories):
            raise ValueError("category values must be uint8 (0-255)")

    @property
    def is_rgb(self) -> bool:
        return len(self.bands) == 3


@dataclass
class RasterLayer:
    """A raster shown through the local tile server.

    Args:
        name: Unique layer name, shown in the layer panel.
        path: GeoTIFF path. Give either ``path`` or ``source``.
        source: Callable returning the GeoTIFF path, called when the layer is
            first added. Use it for cached layers that are built on demand.
        style: Rendering style.
        group: Layer panel group.
        opacity: Initial opacity, 0 to 1.
        visible: Whether the layer starts switched on.
        description: Optional note shown in the layer panel tooltip.
    """

    name: str
    path: Path | None = None
    source: Callable[[], Path] | None = None
    style: RasterStyle = field(default_factory=RasterStyle)
    group: str = "Layers"
    opacity: float = 1.0
    visible: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        if (self.path is None) == (self.source is None):
            raise ValueError(f"{self.name}: give exactly one of path or source")
        if self.path is not None:
            self.path = Path(self.path)

    def resolve(self) -> Path:
        """Return the raster path, building it through ``source`` if needed.

        Raises:
            FileNotFoundError: The path does not exist.
        """
        path = Path(self.source()) if self.source is not None else self.path
        if not path.exists():
            raise FileNotFoundError(f"{self.name}: raster not found: {path}")
        return path


@dataclass(frozen=True)
class VectorStyle:
    """How vector features are drawn.

    Args:
        color: Outline colour.
        weight: Outline width in pixels.
        fill_color: Fill colour. ``None`` draws outlines only.
        fill_opacity: Fill opacity, 0 to 1.
        by: Attribute that selects a colour from ``categories``.
        categories: Attribute value (as a string) to class. Features whose value
            is missing from the map keep ``color``.
    """

    color: Color = "#ffff00"
    weight: float = 2.0
    fill_color: Color | None = None
    fill_opacity: float = 0.15
    by: str | None = None
    categories: dict[str, Category] | None = None


@dataclass
class VectorLayer:
    """A vector file (GeoPackage, shapefile, GeoJSON) drawn as GeoJSON.

    Args:
        name: Unique layer name.
        path: Vector file path.
        layer: Layer name inside a multi-layer file such as a GeoPackage.
        style: Drawing style.
        popup: Attributes shown when a feature is clicked. ``None`` shows all.
        group: Layer panel group.
        opacity: Initial opacity (scales outline and fill), 0 to 1.
        visible: Whether the layer starts switched on. A hidden layer sends its
            features to the map only when it is first switched on.
        description: Optional note for the layer panel.
        simplify_m: Display simplification tolerance in metres (applied when
            the file's CRS is projected). Pixel-traced outlines shrink several
            fold at one pixel. ``None`` or 0 keeps every vertex.
    """

    name: str
    path: Path
    layer: str | None = None
    style: VectorStyle = field(default_factory=VectorStyle)
    popup: list[str] | None = None
    group: str = "Vectors"
    opacity: float = 1.0
    visible: bool = True
    description: str = ""
    simplify_m: float | None = 0.6

    def __post_init__(self) -> None:
        self.path = Path(self.path)


@dataclass
class TileLayerSpec:
    """Any XYZ tile service, e.g. an Earth Engine map id URL.

    Args:
        name: Unique layer name.
        url: XYZ URL template with ``{z}``, ``{x}`` and ``{y}``.
        attribution: Attribution text.
        group: Layer panel group.
        opacity: Initial opacity, 0 to 1.
        visible: Whether the layer starts switched on.
        legend: Optional categorical legend.
        description: Optional note for the layer panel.
    """

    name: str
    url: str
    attribution: str = ""
    group: str = "Tiles"
    opacity: float = 1.0
    visible: bool = True
    legend: dict[Any, Category] | None = None
    description: str = ""


AnyLayer = RasterLayer | VectorLayer | TileLayerSpec
