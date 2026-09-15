"""Shared helpers for the metric example figures in ``docs/metrics.md``.

Every script in this directory renders one metric for one stand from the
Elkinsville calibration delivery. This module holds what those scripts have in
common: where the data lives, the panels that ``csdv_core.viz`` does not already
provide, the stage envelope bands, and a shared command line.

Anything already in ``csdv_core.viz`` is imported and used as is. Only the parts
that do not exist there are written here. If a helper below turns out to serve
three or more metrics it is a candidate to move into ``csdv_core.viz``.

Figures carry as little text as possible. Year labels, row labels, colour keys
and axis labels go on the figure. Everything else goes in the caption in
``docs/metrics.md``.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
from matplotlib.axes import Axes
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.path import Path as Path_
from shapely.geometry.base import BaseGeometry

from csdv_core.config import load_stages
from csdv_core.io.paths import project_paths
from csdv_core.io.satellite_io import (
    ANNUAL_NAME,
    OBSERVATIONS_NAME,
    read_annual,
    read_observations,
)
from csdv_core.io.stands import read_ais_stands
from csdv_core.viz.maps import read_band_window
from csdv_core.viz.style import (
    GRID,
    INK,
    MUTED,
    STAGE_COLORS,
    save_fig,
    stage_color,
)

logger = logging.getLogger(__name__)

#: Repository root, found from this file rather than from the working
#: directory, so a script runs the same from anywhere.
REPO = Path(__file__).resolve().parents[4]

#: Where the figures embedded in ``docs/metrics.md`` are written.
DOC_FIGURE_DIR = REPO / "docs" / "images" / "metrics"

#: Canopy height threshold that separates gap from canopy, in metres. Matches
#: ``csdv_core.zonal.pixel.CANOPY_HEIGHT_THRESHOLD_M``.
CANOPY_THRESHOLD_M = 2.0

#: Colours for a two-class canopy height mask. Both come from the stage
#: palette: the open class takes the colour of the most open stage, the canopy
#: class the colour of the closed-canopy stage.
GAP_COLOR = STAGE_COLORS["ESI"]
CANOPY_COLOR = STAGE_COLORS["LSE"]

#: How much of its colour an out-of-polygon pixel keeps. The metric counts
#: in-stand pixels only, so everything outside is washed out.
OUTSIDE_STRENGTH = 0.22

#: Colour ramp for crown diameter, shared by the map overlay and the strip
#: panel so the two rows read as one system. Deliberately not ``viridis``,
#: which the canopy height row already uses.
CROWN_CMAP = "plasma"

#: Ends of the crown diameter ramp, in metres. The 1st to 99th percentile range
#: across the module is 4.6 to 19.0 m under the 2026 segmentation, so this
#: covers the body of the distribution without spending the ramp on the tail.
#:
#: These were 5 and 60 before the re-tune, when segments were canopy clusters
#: averaging 33 m across. Left unchanged they put every crown in the bottom
#: quarter of the ramp.
CROWN_DIAM_MIN_M = 4.0
CROWN_DIAM_MAX_M = 20.0

#: Chip framing, matching the worked-example figures in the classification
#: document so the two sets of figures read alike.
PAD_FRACTION = 0.12
MIN_PAD_M = 25.0


@dataclass(frozen=True)
class ExampleSite:
    """Where one calibration site's inputs live.

    Attributes:
        site: Site code used in every path, for example ``ElkinsvilleNE``.
        gdb: The interpreter geodatabase.
        naip_dir: Parent of the per-year NAIP directories.
        chm_dir: Parent of the per-year canopy height model directories.
        metrics_parquet: Stand metrics written by the zonal pipeline.
        crowns_dir: Per-year crown GeoPackages.
        chm_scale: Read scale for the canopy height model. Use ``1.0`` for
            float32 metres and ``0.01`` for uint16 centimetres.
        annual_parquet: One row per stand per year from the satellite pipeline.
        observations_parquet: One row per stand per Landsat scene, before
            quality control.
    """

    site: str
    gdb: Path
    naip_dir: Path
    chm_dir: Path
    metrics_parquet: Path
    crowns_dir: Path
    chm_scale: float = 1.0
    annual_parquet: Path | None = None
    observations_parquet: Path | None = None

    def naip(self, year: int) -> Path:
        """Return the NAIP image for ``year``."""
        return _one_tif(self.naip_dir / str(year))

    def chm(self, year: int) -> Path:
        """Return the canopy height model for ``year``."""
        return _one_tif(self.chm_dir / str(year))

    def crowns(self, year: int) -> Path:
        """Return the crown GeoPackage for ``year``.

        The path is explicit rather than routed through
        ``ProjectPaths.crowns_dir``, which resolves to
        ``results/crowns/<site>/<year>/crowns.gpkg``. The stand pipeline writes
        these somewhere else, and only this location holds real files.
        """
        return self.crowns_dir / f"crowns_{year}.gpkg"


def _one_tif(directory: Path) -> Path:
    """Return the single GeoTIFF in ``directory``.

    Raises:
        FileNotFoundError: If the directory holds no GeoTIFF.
    """
    matches = sorted(directory.glob("*.tif"))
    if not matches:
        raise FileNotFoundError(f"No GeoTIFF under {directory}")
    if len(matches) > 1:
        logger.warning(
            "%d GeoTIFFs under %s, using %s", len(matches), directory, matches[0].name
        )
    return matches[0]


def elkinsville() -> ExampleSite:
    """Return the Elkinsville site, resolved through ``CSDV_*`` roots.

    The geodatabase name is doubled on disk. The outer directory is not a valid
    FileGDB, so the inner path is the one that opens.
    """
    paths = project_paths()
    name = "Indiana-ElkinsvilleNE_revised.gdb"
    stands = paths.stands_dir("ElkinsvilleNE")
    return ExampleSite(
        site="ElkinsvilleNE",
        gdb=paths.data_root / "calibration" / name / name,
        naip_dir=paths.data_root / "naip" / "ElkinsvilleNE",
        chm_dir=paths.data_root / "naip_chm" / "ElkinsvilleNE",
        metrics_parquet=stands / "stand_metrics.parquet",
        crowns_dir=_current_crowns_dir(stands),
        annual_parquet=stands / ANNUAL_NAME,
        observations_parquet=(
            paths.data_root / "satellite" / "ElkinsvilleNE" / OBSERVATIONS_NAME
        ),
    )


def _current_crowns_dir(stands: Path) -> Path:
    """Crown directory for the segmentation parameters now in force.

    Crowns are written under a hash of the parameter set, so a figure cannot
    quietly draw crowns from a different segmentation than the metric table it
    is plotting. Falls back to the flat directory, which holds the crowns from
    before the 2026 re-tune.
    """
    from csdv_core.segmentation.params import DEFAULT_PARAMS

    keyed = stands / "crowns" / f"seg-{DEFAULT_PARAMS.key}"
    if keyed.is_dir():
        return keyed
    logger.warning("No crowns under %s, falling back to the flat directory", keyed)
    return stands / "crowns"


@dataclass(frozen=True)
class ExampleContext:
    """The tables an example figure reads, loaded once.

    Attributes:
        site: The site these tables describe.
        stands: Stand polygons with the interpreter attributes attached.
        metrics: One row per stand per date from the zonal pipeline.
    """

    site: ExampleSite
    stands: gpd.GeoDataFrame
    metrics: pd.DataFrame


def load_example_context(site: ExampleSite) -> ExampleContext:
    """Read the stand polygons and the stand metrics table.

    Raises:
        FileNotFoundError: If either input is missing.
    """
    if not site.gdb.exists():
        raise FileNotFoundError(f"Geodatabase not found: {site.gdb}")
    if not site.metrics_parquet.exists():
        raise FileNotFoundError(
            f"Stand metrics not found: {site.metrics_parquet}. "
            "Run the stand pipeline before building an example figure."
        )
    stands = read_ais_stands(site.gdb)
    metrics = pd.read_parquet(site.metrics_parquet)
    logger.info("Loaded %d stands and %d metric rows", len(stands), len(metrics))
    return ExampleContext(site=site, stands=stands, metrics=metrics)


def stand_row(stands: gpd.GeoDataFrame, stand_id: str):
    """Return the single stand record for ``stand_id``.

    Raises:
        KeyError: If no stand carries that identifier.
    """
    match = stands[stands["stand_id"] == stand_id]
    if match.empty:
        raise KeyError(
            f"Unknown stand_id {stand_id!r}. "
            f"First few available: {sorted(stands['stand_id'])[:6]}"
        )
    return match.iloc[0]


def disturbance_years(stand) -> tuple[float | None, float | None]:
    """Return the interpreter's pre and post disturbance imagery years.

    Returns:
        ``(last_pre, first_post)``, either of which is None when the delivery
        left the field unset.
    """

    def _value(column: str) -> float | None:
        raw = stand.get(column)
        if raw is None or pd.isna(raw):
            return None
        year = float(raw)
        return year if year > 1900 else None

    return _value("LastImageryPreDist"), _value("FirstImageryPostDist")


def stand_metric_series(metrics: pd.DataFrame, stand_id: str) -> pd.DataFrame:
    """Return one stand's rows, sorted by year."""
    subset = metrics[metrics["stand_id"] == stand_id]
    if subset.empty:
        raise KeyError(f"No metric rows for stand_id {stand_id!r}")
    return subset.sort_values("year").reset_index(drop=True)


def stand_mask(
    geometry: BaseGeometry,
    shape: tuple[int, int],
    transform,
) -> np.ndarray:
    """Boolean in-stand mask for a displayed block.

    Pixel-centre membership, matching the rule ``zonal/mask.py`` applies when
    the metric is computed, so the pixels a panel draws are the pixels the
    number counted.
    """
    return rasterio.features.geometry_mask(
        [geometry],
        out_shape=shape,
        transform=transform,
        invert=True,
        all_touched=False,
    )


def class_panel(
    ax: Axes,
    classes: np.ndarray,
    transform,
    *,
    colors: Sequence[str],
    geometry: BaseGeometry,
    inside: np.ndarray | None = None,
    invalid: np.ndarray | None = None,
    dim_outside: bool = True,
) -> np.ndarray:
    """Draw an integer class array as flat colour, washed out beyond the stand.

    This is the panel that stands in for a per-pixel metric. The caller builds
    the class array with the same function the metric uses, so the picture and
    the number cannot drift apart.

    Args:
        ax: Axis to draw on.
        classes: Integer array indexing into ``colors``.
        transform: Affine of the displayed block.
        colors: One colour per class, in class order.
        geometry: Stand geometry in the raster CRS.
        inside: In-stand mask. Derived from ``geometry`` when omitted.
        invalid: Pixels to draw as nodata grey, whatever class they carry.
        dim_outside: Wash out pixels whose centre falls outside the polygon.

    Returns:
        The boolean in-stand mask of the displayed block.
    """
    from csdv_core.viz.maps import draw_stand_outline

    classes = np.asarray(classes)
    if inside is None:
        inside = stand_mask(geometry, classes.shape, transform)

    rgb = np.zeros((*classes.shape, 3), dtype=np.float32)
    for value, color in enumerate(colors):
        rgb[classes == value] = _hex_to_rgb(color)
    if invalid is not None:
        rgb[invalid] = _hex_to_rgb(MUTED)

    if dim_outside:
        outside = ~inside
        rgb[outside] = 1.0 - (1.0 - rgb[outside]) * OUTSIDE_STRENGTH

    ax.imshow(rgb, interpolation="nearest")
    draw_stand_outline(ax, geometry, transform)
    ax.set_xticks([])
    ax.set_yticks([])
    return inside


def class_key(
    cax: Axes,
    colors: Sequence[str],
    labels: Sequence[str],
    *,
    label: str = "",
    fontsize: float | None = None,
) -> None:
    """Draw a discrete colour key for :func:`class_panel`.

    Args:
        cax: A narrow axis reserved for the key.
        colors: The same colours passed to the panel, in class order.
        labels: One label per class.
        label: Optional axis label.
        fontsize: Tick label size. Three classes need a smaller size than two
            to fit the height of one image row.
    """
    if len(colors) != len(labels):
        raise ValueError(f"{len(colors)} colours against {len(labels)} labels")
    n = len(colors)
    cmap = ListedColormap(list(colors))
    bar = ColorbarBase(
        cax,
        cmap=cmap,
        norm=BoundaryNorm(list(range(n + 1)), cmap.N),
        ticks=[i + 0.5 for i in range(n)],
        orientation="vertical",
    )
    bar.ax.set_yticklabels(list(labels))
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0)
    if fontsize is not None:
        bar.ax.tick_params(labelsize=fontsize)
    if label:
        bar.set_label(label)


def band_panel(
    ax: Axes,
    values: np.ndarray,
    transform,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    geometry: BaseGeometry,
    inside: np.ndarray | None = None,
    dim_outside: bool = True,
):
    """Draw a continuous single-band array, washed out beyond the stand.

    ``chm_panel`` cannot serve here. It hard-wires viridis and a 0 to 30 m
    range, which suits canopy height and nothing else. This takes an array the
    caller already holds, which is what a metric computed from an intermediate
    array needs.

    Returns:
        ``(mappable, inside)``. The mappable carries the colourbar.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.pyplot import get_cmap

    from csdv_core.viz.maps import draw_stand_outline

    values = np.asarray(values, dtype=np.float32)
    if inside is None:
        inside = stand_mask(geometry, values.shape, transform)

    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = get_cmap(cmap)
    rgb = colormap(norm(values))[..., :3].astype(np.float32)
    rgb[~np.isfinite(values)] = _hex_to_rgb(MUTED)

    if dim_outside:
        outside = ~inside
        rgb[outside] = 1.0 - (1.0 - rgb[outside]) * OUTSIDE_STRENGTH

    ax.imshow(rgb, interpolation="nearest")
    draw_stand_outline(ax, geometry, transform)
    ax.set_xticks([])
    ax.set_yticks([])
    return ScalarMappable(norm=norm, cmap=colormap), inside


def height_class_panel(
    ax: Axes,
    chm_path: Path | str,
    bounds: tuple[float, float, float, float],
    *,
    geometry: BaseGeometry,
    threshold_m: float = CANOPY_THRESHOLD_M,
    scale: float = 1.0,
    max_px: int | None = 500,
    dim_outside: bool = True,
) -> np.ndarray:
    """Draw the canopy height model split into two classes at ``threshold_m``.

    This is the metric before it is reduced to a single number. Pixels outside
    the polygon are washed out, because a fraction metric counts in-stand
    pixels only and a reader should be able to see which pixels those are.

    Args:
        ax: Axis to draw on.
        chm_path: Canopy height model for one date.
        bounds: Map bounds of the chip.
        geometry: Stand geometry in the raster CRS.
        threshold_m: Height that separates the two classes.
        scale: Read scale for the canopy height model.
        max_px: Cap on the longer side of the read, decimating as needed.
        dim_outside: Wash out pixels whose centre falls outside the polygon.

    Returns:
        The boolean in-stand mask of the displayed block, so a caller can check
        the drawn classes against the value in the metrics table.
    """
    arr, transform = read_band_window(chm_path, bounds, scale=scale, max_px=max_px)
    return class_panel(
        ax,
        (arr >= threshold_m).astype(np.uint8),
        transform,
        colors=(GAP_COLOR, CANOPY_COLOR),
        geometry=geometry,
        invalid=~np.isfinite(arr),
        dim_outside=dim_outside,
    )


def height_class_key(
    cax: Axes,
    *,
    threshold_m: float = CANOPY_THRESHOLD_M,
    label: str = "",
) -> None:
    """Draw a two-swatch colour key for :func:`height_class_panel`.

    Args:
        cax: A narrow axis reserved for the key.
        threshold_m: The height that splits the two classes.
        label: Optional axis label.
    """
    class_key(
        cax,
        (GAP_COLOR, CANOPY_COLOR),
        (f"< {threshold_m:g} m", f"≥ {threshold_m:g} m"),
        label=label,
    )


def _hex_to_rgb(color: str) -> np.ndarray:
    """Convert a ``#rrggbb`` string to a float RGB triple in 0 to 1."""
    from matplotlib.colors import to_rgb

    return np.asarray(to_rgb(color), dtype=np.float32)


@dataclass(frozen=True)
class EnvelopeBand:
    """One stage envelope, merged with any stage sharing its bounds.

    Attributes:
        lo: Lower bound, resolved against the axis when the envelope is open.
        hi: Upper bound, likewise.
        codes: Stage codes sharing these bounds, in ``stage_order``.
        open_lo: True when the envelope set no lower bound.
        open_hi: True when it set no upper bound.
    """

    lo: float
    hi: float
    codes: tuple[str, ...]
    open_lo: bool = False
    open_hi: bool = False

    @property
    def color(self) -> str:
        """Colour of the first stage in the group, which is what is drawn."""
        return stage_color(self.codes[0])

    @property
    def name(self) -> str:
        """The stage codes, as they appear inside a band."""
        return " / ".join(self.codes)

    @property
    def label(self) -> str:
        """Codes and range together, for a legend entry."""
        lo = "open" if self.open_lo else f"{self.lo:.2f}"
        hi = "open" if self.open_hi else f"{self.hi:.2f}"
        return f"{self.name}  {lo} to {hi}"


def stage_envelope_groups(
    metric: str,
    *,
    site_type_key: str = "type_00",
    low: float = 0.0,
    high: float = 1.0,
) -> list[EnvelopeBand]:
    """Return the merged stage envelopes for ``metric``, in stage order.

    Stages whose envelope is identical are merged, because several metrics do
    not separate every stage on their own and drawing four bands on top of each
    other would hide that.

    Args:
        metric: Metric name as it appears in ``stages.yaml``.
        site_type_key: Which site type's envelopes to read.
        low: Value an open lower bound resolves to, normally the axis floor.
        high: Value an open upper bound resolves to.

    Returns:
        One :class:`EnvelopeBand` per distinct pair of bounds. Empty when no
        stage constrains this metric.
    """
    config = load_stages()
    order = config.stage_order or list(config.stages)

    groups: dict[tuple[float, float, bool, bool], list[str]] = {}
    for code in order:
        stage = config.stages.get(code)
        if stage is None:
            continue
        span = stage.envelopes.get(site_type_key, {}).get(metric)
        if span is None:
            continue
        key = (
            float(span.min) if span.min is not None else low,
            float(span.max) if span.max is not None else high,
            span.min is None,
            span.max is None,
        )
        groups.setdefault(key, []).append(code)

    return [
        EnvelopeBand(lo, hi, tuple(codes), open_lo=open_lo, open_hi=open_hi)
        for (lo, hi, open_lo, open_hi), codes in groups.items()
    ]


def stage_envelope_legend_handles(
    metric: str,
    *,
    site_type_key: str = "type_00",
    low: float = 0.0,
    high: float = 1.0,
    alpha: float = 0.6,
) -> list:
    """Legend patches for a metric whose bands cannot be labelled in place.

    ``ndvi_mean`` is the case this exists for. Six of its seven envelopes sit
    between 0.70 and 0.93 and their midpoints fall within 0.02 of each other,
    so no font size fits a stage code beside each one. The bands still have to
    be named, and a legend names them without pretending they are separable.

    The swatches are drawn more opaque than the bands themselves. On the panel
    the bands overlap and their colours composite, so a swatch matched to the
    plotted alpha would come out near white and tell the reader nothing. Read a
    swatch as the identity of a band, not as the exact colour on the axis.
    """
    from matplotlib.patches import Patch

    return [
        Patch(facecolor=band.color, alpha=alpha, lw=0, label=band.label)
        for band in stage_envelope_groups(
            metric, site_type_key=site_type_key, low=low, high=high
        )
    ]


def stage_envelope_bands(
    ax: Axes,
    metric: str,
    *,
    site_type_key: str = "type_00",
    alpha: float = 0.42,
    label: bool = True,
    fontsize: float = 6.5,
) -> int:
    """Shade each stage's envelope for ``metric`` behind a time series.

    Stages whose envelope for this metric is identical are merged into one
    band with a combined label. Several metrics do not separate every stage on
    their own, and drawing four bands on top of each other would hide that.

    Call this before the series is plotted, and set the y limits first, because
    an envelope with an open bound is drawn out to the current limit.

    Args:
        ax: Axis holding the time series.
        metric: Metric name as it appears in ``stages.yaml``.
        site_type_key: Which site type's envelopes to read.
        alpha: Band opacity.
        label: Draw the stage codes inside each band.
        fontsize: Label size.

    Returns:
        The number of bands drawn, which is zero when no stage constrains this
        metric.
    """
    import matplotlib.patheffects as pe
    from matplotlib.transforms import blended_transform_factory

    low, high = ax.get_ylim()
    bands = stage_envelope_groups(
        metric, site_type_key=site_type_key, low=low, high=high
    )

    transform = blended_transform_factory(ax.transAxes, ax.transData)
    for band in bands:
        ax.axhspan(band.lo, band.hi, color=band.color, alpha=alpha, lw=0, zorder=-2)
        # An envelope can run past the plotted range. The crown_cv old growth
        # band reaches 1.5 while the axis usually stops well below that, and a
        # label at the band's true midpoint would float outside the panel.
        visible_lo, visible_hi = max(band.lo, low), min(band.hi, high)
        if visible_hi <= visible_lo:
            continue
        if label:
            ax.text(
                0.012,
                (visible_lo + visible_hi) / 2.0,
                band.name,
                transform=transform,
                fontsize=fontsize,
                color=INK,
                va="center",
                ha="left",
                # Above the disturbance shading at zorder 0 and above the series
                # line at zorder 3, but below the rings at zorder 5. A label has
                # to sit on the band it names, and on a metric whose series runs
                # across the left edge, such as ndvi_seasonal_amplitude, there
                # is no free height to move it to. Losing a few characters'
                # width of a dense line costs less than an unreadable code. The
                # halo makes the interruption look deliberate.
                zorder=4,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
            )
    logger.info("Drew %d envelope band(s) for %s", len(bands), metric)
    return len(bands)


def reference_band(
    ax: Axes,
    metrics: pd.DataFrame,
    metric: str,
    *,
    lo: float = 0.1,
    hi: float = 0.9,
    color: str = MUTED,
    alpha: float = 0.16,
    label: str | None = None,
) -> int:
    """Shade where the rest of the module sits, year by year.

    Two metrics here have no stage envelope, so a series panel drawn the usual
    way has nothing behind it and a reader has no way to judge whether a value
    is unusual. This puts the other thirty-nine stands behind the line instead:
    the band is the ``lo`` to ``hi`` quantile across every stand at that date,
    and the line inside it is the median.

    It also answers a question a stage envelope cannot. A metric that moves
    with the acquisition rather than with the forest moves the whole band, and
    a stand that only tracks the band is telling you about the scene.

    Args:
        ax: Axis holding the time series.
        metrics: The full stand metrics table, every stand.
        metric: Column to summarise.
        lo: Lower quantile.
        hi: Upper quantile.
        color: Band and median colour.
        alpha: Band opacity.
        label: Legend name for the band. None leaves it out of the legend.

    Returns:
        The number of stands the band was built from.
    """
    if metric not in metrics.columns:
        logger.warning("No %s column, drawing no reference band", metric)
        return 0
    frame = metrics[["stand_id", "year", metric]].dropna()
    if frame.empty:
        logger.warning("Every %s value is null, drawing no reference band", metric)
        return 0
    grouped = frame.groupby("year")[metric]
    years = np.asarray(sorted(grouped.groups), dtype=float)
    low = grouped.quantile(lo).reindex(years).to_numpy(dtype=float)
    high = grouped.quantile(hi).reindex(years).to_numpy(dtype=float)
    mid = grouped.median().reindex(years).to_numpy(dtype=float)

    ax.fill_between(
        years, low, high, color=color, alpha=alpha, lw=0, zorder=-2, label=label
    )
    ax.plot(years, mid, "-", color=color, lw=1.0, alpha=0.75, zorder=-1)
    n_stands = int(frame["stand_id"].nunique())
    logger.info(
        "Reference band for %s over %d stands, %.0f to %.0f percentile",
        metric,
        n_stands,
        lo * 100,
        hi * 100,
    )
    return n_stands


def mark_mapped_dates(
    ax: Axes,
    years: Sequence[float],
    values: Sequence[float],
    *,
    color: str = INK,
    size: float = 11.0,
) -> None:
    """Ring the dates that also appear as image panels.

    The ring is what ties a point on the series to a column of chips, without
    putting any text on the figure.
    """
    ax.plot(
        np.asarray(years, dtype=float),
        np.asarray(values, dtype=float),
        "o",
        mfc="none",
        mec=color,
        mew=1.3,
        ms=size,
        zorder=5,
        # A ring on a value at the very edge of the range would otherwise be
        # sliced in half by the axis.
        clip_on=False,
    )


def load_satellite_annual(site: ExampleSite, stand_id: str) -> pd.DataFrame:
    """Return one stand's annual Landsat metrics, sorted by year.

    This reads ``satellite_annual.parquet`` rather than ``stand_metrics``. The
    satellite join has not been re-applied since the segmentation re-tune, so
    the stand metrics table carries no ``ndvi_*`` columns at all. The annual
    table is the better source in any case: it runs one value a year back to
    the mid-1980s against six NAIP dates.

    Raises:
        FileNotFoundError: If the annual table has not been written.
        KeyError: If the stand has no rows.
    """
    if site.annual_parquet is None or not site.annual_parquet.exists():
        raise FileNotFoundError(
            f"No annual satellite table: {site.annual_parquet}. "
            "Run `csdv satellite annual` before building a satellite figure."
        )
    frame = read_annual(site.annual_parquet)
    subset = frame[frame["stand_id"] == stand_id]
    if subset.empty:
        raise KeyError(f"No satellite rows for stand_id {stand_id!r}")
    return subset.sort_values("year").reset_index(drop=True)


def load_satellite_observations(
    site: ExampleSite,
    stand_id: str,
    *,
    index: str = "ndvi",
    quality_controlled: bool = True,
) -> pd.DataFrame:
    """Return one stand's per-scene observations, sorted by date.

    Quality control is applied per stand with the parameters from
    ``satellite.yaml``, which is exactly what ``annual_table`` does before it
    computes anything. Skipping it would draw a cloud of observations that the
    annual value was never fitted to.

    Args:
        site: Site whose observation table to read.
        stand_id: Stand to return.
        index: Index column the quality gates read.
        quality_controlled: Apply the gates. False returns the raw rows, which
            is only useful for showing what the gates removed.

    Raises:
        FileNotFoundError: If the observation table is missing.
        KeyError: If the stand has no rows.
    """
    from csdv_core.config import load_satellite
    from csdv_core.satellite.annual import filter_observations

    if site.observations_parquet is None or not site.observations_parquet.exists():
        raise FileNotFoundError(
            f"No observation table: {site.observations_parquet}. "
            "Run `csdv satellite extract` before building a satellite figure."
        )
    frame = read_observations(site.observations_parquet)
    subset = frame[frame["stand_id"] == stand_id]
    if subset.empty:
        raise KeyError(f"No observations for stand_id {stand_id!r}")
    if quality_controlled:
        cfg = load_satellite()
        subset, counts = filter_observations(
            subset,
            index=index,
            min_pixels=cfg.qa.min_pixels,
            min_effective_pixels=cfg.qa.min_effective_pixels,
            min_coverage_fraction=cfg.qa.min_coverage_fraction,
            min_area_m2=cfg.qa.min_area_m2,
            valid_range=tuple(cfg.qa.index_valid_range),
        )
        logger.info(
            "%s: %d observations kept, dropped %s",
            stand_id,
            len(subset),
            ", ".join(f"{n} {cause}" for cause, n in counts.items()) or "nothing",
        )
    return subset.sort_values("date").reset_index(drop=True)


def decimal_year(observations: pd.DataFrame) -> np.ndarray:
    """Year plus day-of-year fraction, for plotting scenes on a year axis."""
    year = pd.to_numeric(observations["year"], errors="coerce").to_numpy(dtype=float)
    doy = pd.to_numeric(observations["doy"], errors="coerce").to_numpy(dtype=float)
    return year + (doy - 1.0) / 365.25


def detail_bounds(
    geometry: BaseGeometry,
    size_m: float,
) -> tuple[float, float, float, float]:
    """A square box of ``size_m`` centred on the stand.

    A stand of a few hundred acres decimates to the point where an individual
    crown is a handful of pixels. Cropping to a detail box keeps the segments
    legible while the panels below still summarise the whole stand.
    """
    minx, miny, maxx, maxy = geometry.bounds
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    half = size_m / 2.0
    return (cx - half, cy - half, cx + half, cy + half)


def load_stand_crowns(
    site: ExampleSite,
    year: int,
    geometry: BaseGeometry,
) -> gpd.GeoDataFrame:
    """Return the crowns whose centroid falls inside the stand.

    Membership matches ``zonal/crowns.py::crowns_in_stand`` exactly, so the
    drawn segments are the ones the metric counted. The read is filtered by
    bounding box first, because each scene-wide GeoPackage holds roughly 40,000
    crowns and runs to about 125 MB.

    Raises:
        FileNotFoundError: If that year's crowns have not been segmented.
    """
    path = site.crowns(year)
    if not path.exists():
        raise FileNotFoundError(
            f"No crowns for {year}: {path}. Run the stand pipeline first."
        )
    crowns = gpd.read_file(path, bbox=geometry.bounds)
    if crowns.empty:
        return crowns
    inside = crowns.geometry.centroid.within(geometry)
    return crowns.loc[inside]


def _polygon_path(geometry: BaseGeometry, inverse) -> Path_:
    """Build one matplotlib path in pixel space, holes included.

    Crown polygons carry interior rings. Walking only the exterior would fill
    the holes in, so every ring becomes its own subpath.
    """
    vertices: list[tuple[float, float]] = []
    codes: list[int] = []
    parts = geometry.geoms if geometry.geom_type.startswith("Multi") else [geometry]
    for part in parts:
        if part.geom_type != "Polygon":
            continue
        for ring in (part.exterior, *part.interiors):
            points = [inverse * (x, y) for x, y in zip(*ring.xy, strict=True)]
            if len(points) < 3:
                continue
            vertices.extend(points)
            codes.extend([Path_.MOVETO] + [Path_.LINETO] * (len(points) - 1))
            codes[-1] = Path_.CLOSEPOLY
    if not vertices:
        return Path_(np.zeros((0, 2)))
    return Path_(np.asarray(vertices), codes)


def crown_overlay(
    ax: Axes,
    crowns: gpd.GeoDataFrame,
    transform,
    *,
    vmin: float = CROWN_DIAM_MIN_M,
    vmax: float = CROWN_DIAM_MAX_M,
    cmap: str = CROWN_CMAP,
    alpha: float = 0.52,
    edge: str = "white",
    linewidth: float = 0.5,
    diam_column: str = "crown_diam_m",
):
    """Fill crown polygons on an image axis, coloured by diameter.

    Args:
        ax: Axis already showing an array read with ``transform``.
        crowns: Crowns to draw, in the raster CRS.
        transform: Affine of the displayed block, as returned by
            ``rgb_panel``. Use the returned one, because it carries any
            decimation the read applied.
        vmin: Diameter at the bottom of the colour ramp.
        vmax: Diameter at the top.
        cmap: Colour ramp, shared with the strip panel.
        alpha: Fill opacity, low enough that the imagery reads through.
        edge: Outline colour.
        linewidth: Outline width.
        diam_column: Column holding the crown diameter.

    Returns:
        The drawn collection, which doubles as the mappable for a colourbar.
    """
    from matplotlib.collections import PathCollection

    frozen = bool(ax.images)
    if frozen:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

    inverse = ~transform
    paths = [_polygon_path(geom, inverse) for geom in crowns.geometry]
    collection = PathCollection(
        paths,
        array=crowns[diam_column].to_numpy(dtype=float),
        cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        alpha=alpha,
        edgecolors=edge,
        linewidths=linewidth,
        zorder=4,
    )
    ax.add_collection(collection)
    if frozen:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    return collection


def diameter_strip_panel(
    ax: Axes,
    years: Sequence[int],
    diameters_by_year: Mapping[int, np.ndarray],
    *,
    vmin: float = CROWN_DIAM_MIN_M,
    vmax: float = CROWN_DIAM_MAX_M,
    cmap: str = CROWN_CMAP,
    min_crowns: int = 3,
    jitter: float = 0.26,
    seed: int = 0,
) -> None:
    """One dot per segment per date, with the mean and a band of ±1 standard deviation.

    This panel is the coefficient of variation made visible. The band half
    width over the mean line is the metric, so a wide band on a low mean is a
    high value. The dots are drawn individually rather than summarised, which
    keeps a date built on three segments from looking like a date built on
    eight hundred.

    Args:
        ax: Axis to draw on.
        years: Dates on the x axis, in order.
        diameters_by_year: Crown diameters per year, in metres.
        vmin: Bottom of the shared colour ramp.
        vmax: Top of the shared colour ramp.
        cmap: Colour ramp, shared with :func:`crown_overlay`.
        min_crowns: Dates below this many segments are drawn but not
            summarised, matching the gate the metric applies.
        jitter: Horizontal spread of the dots, in years.
        seed: Fixed so the jitter is identical on every run.
    """
    rng = np.random.default_rng(seed)
    positions = np.arange(len(years), dtype=float)

    for x, year in zip(positions, years, strict=True):
        values = np.asarray(diameters_by_year.get(year, []), dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        if values.size >= min_crowns:
            mean = float(np.mean(values))
            std = float(np.std(values))
            ax.add_patch(
                Rectangle(
                    (x - 0.34, mean - std),
                    0.68,
                    2.0 * std,
                    facecolor=GRID,
                    edgecolor="none",
                    alpha=0.85,
                    zorder=1,
                )
            )
            ax.plot(
                [x - 0.34, x + 0.34],
                [mean, mean],
                "-",
                color=INK,
                lw=1.6,
                zorder=3,
                solid_capstyle="butt",
            )
        ax.scatter(
            x + rng.uniform(-jitter, jitter, size=values.size),
            values,
            c=values,
            cmap=cmap,
            norm=Normalize(vmin=vmin, vmax=vmax),
            s=7.0,
            linewidths=0.0,
            alpha=0.85,
            zorder=2,
            # Eight hundred dots per date is a lot of vector geometry.
            rasterized=True,
        )

    ax.set_xlim(-0.6, len(years) - 0.4)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.35, axis="y")


def column_year_labels(
    axes: Sequence[Axes],
    years: Sequence[int | str],
    *,
    fontsize: float | None = None,
) -> None:
    """Put the year above each column, on the top row only.

    A column standing for a pair of dates passes a string such as
    ``"2016 to 2018"`` instead of a year, which needs a smaller size to fit.
    """
    for ax, year in zip(axes, years, strict=True):
        ax.set_title(str(year), fontsize=fontsize)


def check_value(
    name: str,
    computed: float,
    expected: float,
    *,
    tolerance: float = 1e-9,
) -> bool:
    """Compare a recomputed metric against the value in the table.

    Step 5 of the procedure in ``README.md``. A figure that does not reproduce
    the number it illustrates is drawing something else, and the only way to
    know is to compute it again from the source raster.

    Returns:
        True when the two agree, or when both are NaN.
    """
    both_nan = not np.isfinite(computed) and not np.isfinite(expected)
    agree = both_nan or bool(abs(float(computed) - float(expected)) <= tolerance)
    if agree:
        logger.info("%s: recomputed %.6f matches the table", name, computed)
    else:
        logger.error(
            "%s: recomputed %.6f against %.6f in the table", name, computed, expected
        )
    return agree


def row_label(ax: Axes, text: str, *, color: str = MUTED) -> None:
    """Label a row of image panels down its left edge."""
    ax.set_ylabel(text, fontsize=8, color=color)


@dataclass(frozen=True)
class ExamplePreset:
    """One named figure: which stand, which dates, where it lands.

    A metric script declares its presets so every figure in the docs can be
    regenerated by name rather than by remembering a stand identifier and a
    year list.

    Attributes:
        stand: Stand identifier.
        years: Years for the image columns.
        slug: Output file stem under ``docs/images/metrics/``.
        detail_m: Crop the image row to a square box of this size, centred on
            the stand. Use on a stand too large to show detail at full extent.
            None frames the whole stand.
    """

    stand: str
    years: tuple[int, ...]
    slug: str
    detail_m: float | None = None

    @property
    def out_path(self) -> Path:
        """Where this preset's figure is written."""
        return DOC_FIGURE_DIR / f"{self.slug}.png"


def build_parser(
    description: str,
    presets: Mapping[str, ExamplePreset],
    *,
    default: str,
) -> argparse.ArgumentParser:
    """Return the command line shared by every metric example script.

    ``--preset`` picks a named figure. The stand, the years and the chip
    framing are all options on top of that, so a figure can be retuned without
    editing code.
    """
    if default not in presets:
        raise KeyError(f"Default preset {default!r} is not in {sorted(presets)}")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--preset",
        choices=sorted(presets),
        default=default,
        help=f"Named figure to build. Default {default}.",
    )
    parser.add_argument(
        "--stand-id",
        default=None,
        help="Stand to render. Overrides the preset.",
    )
    parser.add_argument(
        "--years",
        default=None,
        help="Comma separated years for the image columns. Overrides the preset.",
    )
    parser.add_argument(
        "--detail-m",
        type=float,
        default=None,
        help=(
            "Crop the image row to a square box of this many metres, centred "
            "on the stand. Overrides the preset."
        ),
    )
    parser.add_argument(
        "--pad-fraction",
        type=float,
        default=PAD_FRACTION,
        help=f"Chip padding as a fraction of the longer side. Default {PAD_FRACTION}.",
    )
    parser.add_argument(
        "--min-pad-m",
        type=float,
        default=MIN_PAD_M,
        help=f"Smallest chip padding in metres. Default {MIN_PAD_M}.",
    )
    parser.add_argument(
        "--max-px",
        type=int,
        default=500,
        help=(
            "Cap on the longer side of each chip read. A chip is displayed at "
            "roughly 320 px, so 500 is already oversampled. Lower this to shrink "
            "the PNG."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output resolution. Default 150.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path. Overrides the preset.",
    )
    parser.add_argument(
        "--no-optimize",
        dest="optimize",
        action="store_false",
        help="Skip the palette re-encode and keep the full RGBA PNG.",
    )
    return parser


def resolve_preset(
    args: argparse.Namespace,
    presets: Mapping[str, ExamplePreset],
) -> ExamplePreset:
    """Apply any command line overrides on top of the chosen preset."""
    preset = presets[args.preset]
    return ExamplePreset(
        stand=args.stand_id or preset.stand,
        years=tuple(parse_years(args.years)) if args.years else preset.years,
        slug=preset.slug,
        detail_m=args.detail_m if args.detail_m is not None else preset.detail_m,
    )


def parse_years(raw: str) -> list[int]:
    """Turn a comma separated year list into integers.

    Raises:
        ValueError: If any entry is not a year.
    """
    years = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not years:
        raise ValueError("No years given")
    return years


#: Soft ceiling for a committed figure, in kilobytes.
SIZE_GUIDE_KB = 400.0


def compress_png(path: Path, *, colors: int = 256) -> float:
    """Re-encode a written PNG to a 256-colour palette in place.

    These figures are committed, and matplotlib writes a 32-bit RGBA canvas
    that compresses poorly. Dropping the unused alpha channel and mapping to a
    palette cuts the file by roughly three quarters at full resolution, which
    is a better trade than rendering at a lower dpi.

    Median cut is the right method here and ``MAXCOVERAGE`` is not. The stage
    envelope bands are large, flat and only a few units apart in each channel.
    Maximum coverage spreads its palette over the colour cube and merges
    neighbouring pale bands into one entry, which erased the boundary between
    the ESI and LSI bands when it was tried. Median cut allocates by pixel
    count, so those large flat areas keep their own entries.

    Dithering is off because it adds noise to the flat bands and to the
    canopy height ramp, and it does not reduce the file.

    Args:
        path: PNG to rewrite.
        colors: Palette size.

    Returns:
        The new file size in kilobytes.
    """
    from PIL import Image

    image = Image.open(path).convert("RGB")
    palette = image.quantize(
        colors=colors,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    )
    palette.save(path, optimize=True, compress_level=9)
    return path.stat().st_size / 1024.0


def finish(fig: Figure, out_path: Path, *, dpi: int, optimize: bool = True) -> Path:
    """Write the figure, compress it, and report its size.

    Figures under ``docs/`` are committed, so the size is worth watching. If one
    comes out heavy, lower ``--max-px`` before lowering ``--dpi``.
    """
    written = save_fig(fig, out_path, dpi=dpi)
    kb = written.stat().st_size / 1024.0
    if optimize:
        kb = compress_png(written)
    logger.info("%s is %.0f KB", written.name, kb)
    if kb > SIZE_GUIDE_KB:
        logger.warning(
            "%s exceeds the %.0f KB guide, consider a lower --max-px",
            written.name,
            SIZE_GUIDE_KB,
        )
    return written


def configure_logging() -> None:
    """Send informational messages to the console."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )


__all__ = [
    "CANOPY_THRESHOLD_M",
    "CROWN_CMAP",
    "CROWN_DIAM_MAX_M",
    "CROWN_DIAM_MIN_M",
    "DOC_FIGURE_DIR",
    "MIN_PAD_M",
    "PAD_FRACTION",
    "REPO",
    "SIZE_GUIDE_KB",
    "EnvelopeBand",
    "ExampleContext",
    "ExamplePreset",
    "ExampleSite",
    "band_panel",
    "build_parser",
    "check_value",
    "class_key",
    "class_panel",
    "column_year_labels",
    "compress_png",
    "configure_logging",
    "crown_overlay",
    "decimal_year",
    "detail_bounds",
    "diameter_strip_panel",
    "disturbance_years",
    "elkinsville",
    "finish",
    "height_class_key",
    "height_class_panel",
    "load_example_context",
    "load_satellite_annual",
    "load_satellite_observations",
    "load_stand_crowns",
    "mark_mapped_dates",
    "parse_years",
    "reference_band",
    "resolve_preset",
    "row_label",
    "stage_envelope_bands",
    "stage_envelope_groups",
    "stage_envelope_legend_handles",
    "stand_mask",
    "stand_metric_series",
    "stand_row",
]
