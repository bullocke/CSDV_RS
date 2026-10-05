"""Shared data and drawing pieces for the lab meeting stage flipbooks.

A flipbook is a series of slide-sized frames for one stand. Every frame shows
all NAIP dates, and one date at a time is highlighted, so a talk can step
through the stand's history: mature forest, disturbance, stand initiation and
early stem exclusion.

The data side follows the residual canopy and gap split in
:mod:`csdv_core.zonal.components`. Components are fixed from the last
pre-disturbance and first post-disturbance canopy height models, and regrowth
is summarised inside the disturbance gaps. Heights are taken over every pixel,
with no 2 m floor, which is what the median height stage rule needs.

Provisional stages follow the UVM suggestion. Median gap height below 1.5 m is
early stand initiation, above 1.5 m late stand initiation, above 3 m early stem
exclusion, and above 10 m late stem exclusion. Dates before the disturbance are
labelled mature, which may also be understory reinitiation.
"""

from __future__ import annotations

import logging
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from scipy import ndimage

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "examples" / "docs" / "metrics"))

from _lib import (  # noqa: E402
    disturbance_years,
    elkinsville,
    load_example_context,
    load_stand_crowns,
    stand_mask,
    stand_row,
)

from csdv_core.viz.style import INK, STAGE_COLORS  # noqa: E402
from csdv_core.zonal.components import (  # noqa: E402
    NEW_GAP,
    PREEXISTING_GAP,
    RESIDUAL,
    classify_components,
    crowns_by_component,
)

logger = logging.getLogger(__name__)

YEARS = (2012, 2014, 2016, 2018, 2020, 2022)
GAP_THRESHOLD_M = 2.0
CACHE_DIR = REPO / "results" / "figures" / "lab_meeting_2026_10" / "cache"
CACHE_VERSION = 3

#: One colour marks "now" everywhere: the chip frame, the year label and the
#: vertical line on every time axis.
NOW_COLOR = "#E4572E"
RESIDUAL_COLOR = "#8a8f94"
REGROWTH_COLOR = "#6a9a2f"

#: (code, label, lower bound of median gap height in metres).
HEIGHT_STAGES = (
    ("ESI", "Early stand initiation", 0.0),
    ("LSI", "Late stand initiation", 1.5),
    ("ESE", "Early stem exclusion", 3.0),
    ("LSE", "Late stem exclusion", 10.0),
)
STAGE_LABELS = {code: label for code, label, _ in HEIGHT_STAGES} | {"MA": "Mature"}
STAGE_FILL = dict(STAGE_COLORS) | {"MA": STAGE_COLORS["MA_OW"]}


@dataclass
class YearStats:
    """Per-date summary of one stand, every pixel counted, no height floor."""

    year: int
    gap_q: dict[str, float]
    residual_q: dict[str, float]
    whole_q: dict[str, float]
    gap_fraction: float
    n_crowns: int
    n_gap_crowns: int
    gap_crown_diam_median: float
    gap_crown_diam_q25: float
    gap_crown_diam_q75: float
    gap_crown_cv: float
    gap_crown_height_median: float
    gap_crown_height_sd: float
    stage: str = ""


@dataclass
class StandSeries:
    """Everything a flipbook frame needs for one stand, read once and cached."""

    stand_id: str
    label: str
    acres: float
    years: tuple[int, ...]
    last_pre: float
    first_post: float
    ref_year: int | None
    post_year: int
    view: tuple[float, float, float, float]
    transform: object
    geometry: object
    inside: np.ndarray
    components: np.ndarray
    rgb: dict[int, np.ndarray]
    chm: dict[int, np.ndarray]
    crowns: dict[int, object]
    stats: dict[int, YearStats] = field(default_factory=dict)

    @property
    def area_ha(self) -> float:
        return float(self.geometry.area) / 1e4


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {k: float("nan") for k in ("p10", "p25", "p50", "p75", "p90", "sd")}
    p10, p25, p50, p75, p90 = np.percentile(values, [10, 25, 50, 75, 90])
    return {
        "p10": float(p10),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p90": float(p90),
        "sd": float(values.std()),
    }


def stage_for(stand: StandSeries, year: int) -> str:
    """Return the provisional stage code for one date of the stand."""
    if year <= stand.last_pre:
        return "MA"
    median = stand.stats[year].gap_q["p50"]
    code = HEIGHT_STAGES[0][0]
    for stage, _, lower in HEIGHT_STAGES[1:]:
        if median > lower:
            code = stage
    return code


def square_view(geometry, raster_path: Path, *, pad_m: float, min_side_m: float):
    """Square map bounds round the stand, shifted to stay inside the raster.

    A stand near the module edge would otherwise get a chip that is cut off on
    one side, and the panels in a strip would no longer line up.
    """
    minx, miny, maxx, maxy = geometry.bounds
    side = max(maxx - minx, maxy - miny) + 2 * pad_m
    side = max(side, min_side_m)
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    with rasterio.open(raster_path) as src:
        left, bottom, right, top = src.bounds
        res = src.res[0]
    side = round(side / res) * res
    x0 = min(max(cx - side / 2, left), right - side)
    y0 = min(max(cy - side / 2, bottom), top - side)
    x0 = left + round((x0 - left) / res) * res
    y0 = top - round((top - y0) / res) * res
    return (x0, y0, x0 + side, y0 + side)


def _read(path: Path, view, bands: list[int]) -> tuple[np.ndarray, object]:
    with rasterio.open(path) as src:
        window = from_bounds(*view, transform=src.transform)
        window = window.round_offsets().round_lengths()
        arr = src.read(bands, window=window).astype(np.float32)
        transform = src.window_transform(window)
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr, transform


def _crown_max_height(crowns, chm: np.ndarray, transform) -> np.ndarray:
    """Maximum canopy height inside each crown polygon, falling back to apex."""
    if len(crowns) == 0:
        return np.zeros(0)
    labels = rasterize(
        ((geom, i + 1) for i, geom in enumerate(crowns.geometry)),
        out_shape=chm.shape,
        transform=transform,
        fill=0,
        dtype="int32",
    )
    values = np.where(np.isfinite(chm), chm, -1.0)
    out = np.asarray(
        ndimage.maximum(values, labels, index=np.arange(1, len(crowns) + 1)),
        dtype=float,
    )
    apex = crowns["apex_h_m"].to_numpy(dtype=float)
    return np.where(out > 0, out, apex)


def load_stand_series(
    stand_id: str,
    *,
    years: tuple[int, ...] = YEARS,
    pad_m: float = 25.0,
    min_side_m: float = 150.0,
    use_cache: bool = True,
) -> StandSeries:
    """Read imagery, heights and crowns for one stand and summarise each date."""
    key = f"{stand_id}_{'-'.join(map(str, years))}_{pad_m:g}_{min_side_m:g}"
    cache = CACHE_DIR / f"{key}_v{CACHE_VERSION}.pkl"
    if use_cache and cache.exists():
        logger.info("Reading cached %s", cache.name)
        with cache.open("rb") as fh:
            return pickle.load(fh)

    site = elkinsville()
    context = load_example_context(site)
    stand = stand_row(context.stands, stand_id)
    last_pre, first_post = disturbance_years(stand)
    pre_years = [y for y in years if y <= last_pre]
    post_years = [y for y in years if y >= first_post]
    ref_year = max(pre_years) if pre_years else None
    post_year = min(post_years)
    view = square_view(
        stand.geometry, site.chm(years[0]), pad_m=pad_m, min_side_m=min_side_m
    )

    rgb: dict[int, np.ndarray] = {}
    chm: dict[int, np.ndarray] = {}
    transform = None
    for year in years:
        block, transform = _read(site.naip(year), view, [1, 2, 3])
        rgb[year] = np.moveaxis(block, 0, -1)
        heights, _ = _read(site.chm(year), view, [1])
        chm[year] = heights[0] * np.float32(site.chm_scale)

    shape = chm[years[0]].shape
    inside = stand_mask(stand.geometry, shape, transform)
    components = classify_components(
        chm[post_year],
        inside,
        chm_ref=None if ref_year is None else chm[ref_year],
        threshold_m=GAP_THRESHOLD_M,
    )
    logger.info(
        "%s: view %.0f m, ref %s, first post %s, residual %d px, new gap %d px",
        stand_id,
        view[2] - view[0],
        ref_year,
        post_year,
        int((components == RESIDUAL).sum()),
        int((components == NEW_GAP).sum()),
    )

    crowns_by_year = {}
    for year in years:
        crowns = load_stand_crowns(site, year, stand.geometry).copy()
        crowns["component"] = crowns_by_component(crowns, components, transform)
        crowns["max_h_m"] = _crown_max_height(crowns, chm[year], transform)
        crowns_by_year[year] = crowns

    series = StandSeries(
        stand_id=stand_id,
        label=str(stand.get("dist_label", "")),
        acres=float(stand.get("Acres", float("nan"))),
        years=tuple(years),
        last_pre=float(last_pre),
        first_post=float(first_post),
        ref_year=ref_year,
        post_year=post_year,
        view=view,
        transform=transform,
        geometry=stand.geometry,
        inside=inside,
        components=components,
        rgb=rgb,
        chm=chm,
        crowns=crowns_by_year,
    )
    for year in years:
        series.stats[year] = _year_stats(series, year)
    for year in years:
        series.stats[year].stage = stage_for(series, year)
        _log_year(series, year)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as fh:
        pickle.dump(series, fh)
    return series


def _year_stats(series: StandSeries, year: int) -> YearStats:
    heights = series.chm[year]
    valid = np.isfinite(heights)
    comp = series.components
    gap = heights[(comp == NEW_GAP) & valid]
    residual = heights[(comp == RESIDUAL) & valid]
    whole = heights[series.inside & valid]
    crowns = series.crowns[year]
    in_gap = crowns["component"].to_numpy() == NEW_GAP
    diam = crowns["crown_diam_m"].to_numpy(dtype=float)[in_gap]
    max_h = crowns["max_h_m"].to_numpy(dtype=float)[in_gap]
    enough = diam.size >= 10
    return YearStats(
        year=year,
        gap_q=_quantiles(gap),
        residual_q=_quantiles(residual),
        whole_q=_quantiles(whole),
        gap_fraction=float(np.mean(whole < GAP_THRESHOLD_M)),
        n_crowns=len(crowns),
        n_gap_crowns=int(in_gap.sum()),
        gap_crown_diam_median=float(np.median(diam)) if enough else float("nan"),
        gap_crown_diam_q25=float(np.percentile(diam, 25)) if enough else float("nan"),
        gap_crown_diam_q75=float(np.percentile(diam, 75)) if enough else float("nan"),
        gap_crown_cv=float(diam.std() / diam.mean()) if enough else float("nan"),
        gap_crown_height_median=float(np.median(max_h)) if enough else float("nan"),
        gap_crown_height_sd=float(max_h.std()) if enough else float("nan"),
    )


def _log_year(series: StandSeries, year: int) -> None:
    s = series.stats[year]
    logger.info(
        "%s %d %-3s gap median %5.2f m (P10 %5.2f, P90 %5.2f) | residual median "
        "%5.2f | gap fraction %.2f | crowns %d (gap %d) | gap crown diam %.1f m, "
        "CV %.2f, max-height SD %.2f",
        series.stand_id,
        year,
        s.stage,
        s.gap_q["p50"],
        s.gap_q["p10"],
        s.gap_q["p90"],
        s.residual_q["p50"],
        s.gap_fraction,
        s.n_crowns,
        s.n_gap_crowns,
        s.gap_crown_diam_median,
        s.gap_crown_cv,
        s.gap_crown_height_sd,
    )


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------


def harmonize_rgb(
    series: StandSeries,
    *,
    ref_year: int = 2016,
    gamma: float = 1.15,
    percentiles: tuple[float, float] = (0.5, 99.5),
) -> dict[int, np.ndarray]:
    """Display-only colour matching across dates.

    Each band of each date is shifted and scaled so that the forest outside the
    stand has the mean and spread it has in ``ref_year``. That forest did not
    change, so what is removed is haze and sensor difference between flights,
    not ecology. One shared stretch then maps every date to 0 to 1.
    """
    ref_year = ref_year if ref_year in series.rgb else series.years[0]
    outside = ~series.inside
    ref = series.rgb[ref_year]
    out = {}
    for year, block in series.rgb.items():
        matched = np.empty_like(block)
        for b in range(3):
            src = block[..., b]
            sel = outside & np.isfinite(src)
            mu, sd = src[sel].mean(), src[sel].std()
            rmu, rsd = ref[..., b][sel].mean(), ref[..., b][sel].std()
            matched[..., b] = (src - mu) / max(sd, 1e-6) * rsd + rmu
        out[year] = matched
    lo, hi = np.nanpercentile(out[ref_year], list(percentiles))
    for year, block in out.items():
        scaled = np.clip((block - lo) / (hi - lo), 0.0, 1.0)
        out[year] = np.power(scaled, 1.0 / gamma)
    return out


def image_extent(series: StandSeries) -> tuple[float, float, float, float]:
    """imshow extent in pixel coordinates matching ``series.transform``."""
    rows, cols = series.inside.shape
    return (-0.5, cols - 0.5, rows - 0.5, -0.5)


def draw_outline(ax: Axes, series: StandSeries, *, color="white", lw=1.4) -> None:
    """Stand boundary with a dark halo, in pixel coordinates."""
    import matplotlib.patheffects as pe

    inverse = ~series.transform
    geom = series.geometry
    parts = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
    for part in parts:
        for ring in [part.exterior, *part.interiors]:
            xs, ys = ring.xy
            px = [inverse * (x, y) for x, y in zip(xs, ys, strict=True)]
            ax.plot(
                [p[0] for p in px],
                [p[1] for p in px],
                color=color,
                lw=lw,
                solid_capstyle="round",
                path_effects=[pe.withStroke(linewidth=lw + 1.6, foreground=INK)],
                zorder=6,
            )


def bare_axis(ax: Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def highlight(ax: Axes, *, on: bool, dim: bool, lw: float = 4.0) -> None:
    """Frame the current chip, or fade it when another date is current."""
    if on:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(NOW_COLOR)
            spine.set_linewidth(lw)
            spine.set_zorder(10)
    elif dim:
        ax.add_patch(
            Rectangle(
                (0, 0),
                1,
                1,
                transform=ax.transAxes,
                color="white",
                alpha=0.30,
                lw=0,
                zorder=9,
            )
        )


def stage_tag(ax: Axes, stage: str, *, fontsize: float = 8.5, y: float = 0.03):
    """Stage name in its colour, on the bottom edge of a chip."""
    dark = stage in ("MA", "UR", "OG", "LSE")
    ax.text(
        0.5,
        y,
        STAGE_LABELS[stage],
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="white" if dark else INK,
        zorder=11,
        bbox={
            "boxstyle": "round,pad=0.28",
            "fc": STAGE_FILL[stage],
            "ec": "none",
            "alpha": 0.95,
        },
    )


def crown_paths(series: StandSeries, crowns):
    """Matplotlib paths for crown polygons in pixel coordinates."""
    from matplotlib.path import Path as MPath

    inverse = ~series.transform
    paths = []
    for geom in crowns.geometry:
        parts = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
        verts, codes = [], []
        for part in parts:
            for ring in [part.exterior, *part.interiors]:
                pts = [inverse * xy for xy in ring.coords]
                verts.extend(pts)
                codes.extend(
                    [MPath.MOVETO] + [MPath.LINETO] * (len(pts) - 2) + [MPath.CLOSEPOLY]
                )
        paths.append(MPath(np.asarray(verts), codes))
    return paths


def sqrt_axis(ax: Axes) -> None:
    ax.set_yscale(
        "function",
        functions=(lambda v: np.sqrt(np.clip(v, 0, None)), np.square),
    )


__all__ = [
    "HEIGHT_STAGES",
    "NEW_GAP",
    "NOW_COLOR",
    "PREEXISTING_GAP",
    "REGROWTH_COLOR",
    "RESIDUAL",
    "RESIDUAL_COLOR",
    "STAGE_FILL",
    "STAGE_LABELS",
    "StandSeries",
    "YEARS",
    "bare_axis",
    "crown_paths",
    "draw_outline",
    "harmonize_rgb",
    "highlight",
    "image_extent",
    "load_stand_series",
    "sqrt_axis",
    "stage_tag",
]
