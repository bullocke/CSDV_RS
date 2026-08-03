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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
from matplotlib.axes import Axes
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from shapely.geometry.base import BaseGeometry

from csdv_core.config import load_stages
from csdv_core.io.paths import project_paths
from csdv_core.io.stands import read_ais_stands
from csdv_core.viz.maps import read_band_window
from csdv_core.viz.style import INK, MUTED, STAGE_COLORS, save_fig, stage_color

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
        chm_scale: Read scale for the canopy height model. Use ``1.0`` for
            float32 metres and ``0.01`` for uint16 centimetres.
    """

    site: str
    gdb: Path
    naip_dir: Path
    chm_dir: Path
    metrics_parquet: Path
    chm_scale: float = 1.0

    def naip(self, year: int) -> Path:
        """Return the NAIP image for ``year``."""
        return _one_tif(self.naip_dir / str(year))

    def chm(self, year: int) -> Path:
        """Return the canopy height model for ``year``."""
        return _one_tif(self.chm_dir / str(year))


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
    return ExampleSite(
        site="ElkinsvilleNE",
        gdb=paths.data_root / "calibration" / name / name,
        naip_dir=paths.data_root / "naip" / "ElkinsvilleNE",
        chm_dir=paths.data_root / "naip_chm" / "ElkinsvilleNE",
        metrics_parquet=paths.stands_dir("ElkinsvilleNE") / "stand_metrics.parquet",
    )


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
    inside = rasterio.features.geometry_mask(
        [geometry],
        out_shape=arr.shape,
        transform=transform,
        invert=True,
        all_touched=False,
    )

    rgb = np.zeros((*arr.shape, 3), dtype=np.float32)
    rgb[...] = _hex_to_rgb(GAP_COLOR)
    rgb[arr >= threshold_m] = _hex_to_rgb(CANOPY_COLOR)
    rgb[~np.isfinite(arr)] = _hex_to_rgb(MUTED)

    if dim_outside:
        outside = ~inside
        rgb[outside] = 1.0 - (1.0 - rgb[outside]) * OUTSIDE_STRENGTH

    ax.imshow(rgb, interpolation="nearest")
    from csdv_core.viz.maps import draw_stand_outline

    draw_stand_outline(ax, geometry, transform)
    ax.set_xticks([])
    ax.set_yticks([])
    return inside


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
    cmap = ListedColormap([GAP_COLOR, CANOPY_COLOR])
    bar = ColorbarBase(
        cax,
        cmap=cmap,
        norm=BoundaryNorm([0, 1, 2], cmap.N),
        ticks=[0.5, 1.5],
        orientation="vertical",
    )
    bar.ax.set_yticklabels([f"< {threshold_m:g} m", f"≥ {threshold_m:g} m"])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0)
    if label:
        bar.set_label(label)


def _hex_to_rgb(color: str) -> np.ndarray:
    """Convert a ``#rrggbb`` string to a float RGB triple in 0 to 1."""
    from matplotlib.colors import to_rgb

    return np.asarray(to_rgb(color), dtype=np.float32)


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
    from matplotlib.transforms import blended_transform_factory

    config = load_stages()
    order = config.stage_order or list(config.stages)
    low, high = ax.get_ylim()

    groups: dict[tuple[float, float], list[str]] = {}
    for code in order:
        stage = config.stages.get(code)
        if stage is None:
            continue
        envelope = stage.envelopes.get(site_type_key, {})
        span = envelope.get(metric)
        if span is None:
            continue
        bounds = (
            float(span.min) if span.min is not None else low,
            float(span.max) if span.max is not None else high,
        )
        groups.setdefault(bounds, []).append(code)

    transform = blended_transform_factory(ax.transAxes, ax.transData)
    for (lo, hi), codes in groups.items():
        ax.axhspan(lo, hi, color=stage_color(codes[0]), alpha=alpha, lw=0, zorder=-2)
        if label:
            ax.text(
                0.012,
                (lo + hi) / 2.0,
                " / ".join(codes),
                transform=transform,
                fontsize=fontsize,
                color=INK,
                va="center",
                ha="left",
                zorder=-1,
            )
    logger.info("Drew %d envelope band(s) for %s", len(groups), metric)
    return len(groups)


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


def column_year_labels(axes: Sequence[Axes], years: Sequence[int]) -> None:
    """Put the year above each column, on the top row only."""
    for ax, year in zip(axes, years, strict=True):
        ax.set_title(str(year))


def row_label(ax: Axes, text: str, *, color: str = MUTED) -> None:
    """Label a row of image panels down its left edge."""
    ax.set_ylabel(text, fontsize=8, color=color)


def build_parser(
    description: str,
    *,
    default_stand: str,
    default_years: Sequence[int],
    default_out: Path,
) -> argparse.ArgumentParser:
    """Return the command line shared by every metric example script.

    The stand, the years and the chip framing are all options, so a figure can
    be retuned without editing code.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--stand-id",
        default=default_stand,
        help=f"Stand to render. Default {default_stand}.",
    )
    parser.add_argument(
        "--years",
        default=",".join(str(y) for y in default_years),
        help=(
            "Comma separated years for the image columns. Default "
            f"{','.join(str(y) for y in default_years)}."
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
        default=default_out,
        help=f"Output path. Default {default_out}.",
    )
    parser.add_argument(
        "--no-optimize",
        dest="optimize",
        action="store_false",
        help="Skip the palette re-encode and keep the full RGBA PNG.",
    )
    return parser


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
    "DOC_FIGURE_DIR",
    "MIN_PAD_M",
    "PAD_FRACTION",
    "REPO",
    "SIZE_GUIDE_KB",
    "compress_png",
    "ExampleContext",
    "ExampleSite",
    "build_parser",
    "column_year_labels",
    "configure_logging",
    "disturbance_years",
    "elkinsville",
    "finish",
    "height_class_key",
    "height_class_panel",
    "load_example_context",
    "mark_mapped_dates",
    "parse_years",
    "row_label",
    "stage_envelope_bands",
    "stand_metric_series",
    "stand_row",
]
