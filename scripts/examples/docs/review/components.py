"""Figures for the reply on tracking residual canopy and gaps separately.

Reviewers proposed splitting a post-disturbance stand into residual canopy and
disturbance-created gaps and tracking recovery inside the gaps. These figures
make two points with Elkinsville stands: the two components can follow
different paths inside one polygon, and whole-stand metrics still add
information that neither component carries alone.

Variants:
    sketch       One stand cut before the record (U16-0-0a). Components come
                 from the first date alone. Imagery, component state, crowns
                 by component, then height P90 and CV by component.
    divergent    A clearcut with reserves (U9-0-0) with a pre-disturbance
                 reference. Imagery, canopy height, component state, then
                 canopy fraction and canopy height P90 by component.
    interaction  Gap patches by fill, gap height against distance to residual
                 canopy, patch size against fill, and the variance split.
    multiples    Five stands, one row each: component map, canopy fraction,
                 and canopy height P90 by component.

Run:
    .micromamba/envs/CSDV/bin/python scripts/examples/docs/review/components.py --variant all
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import PathCollection  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "metrics"))

from _lib import (  # noqa: E402
    CANOPY_COLOR,
    CANOPY_THRESHOLD_M,
    GAP_COLOR,
    REPO,
    ExampleContext,
    ExampleSite,
    _polygon_path,
    band_panel,
    check_value,
    class_key,
    class_panel,
    column_year_labels,
    configure_logging,
    disturbance_years,
    elkinsville,
    finish,
    load_example_context,
    load_stand_crowns,
    row_label,
    stand_mask,
    stand_metric_series,
    stand_row,
)

from csdv_core.viz.maps import (  # noqa: E402
    chm_panel,
    padded_bounds,
    read_band_window,
    rgb_panel,
)
from csdv_core.viz.panels import metric_panel  # noqa: E402
from csdv_core.viz.style import (  # noqa: E402
    ACCENT,
    GRID,
    HIGHLIGHT,
    INK,
    MUTED,
    STAGE_COLORS,
    setup_style,
)
from csdv_core.zonal.components import (  # noqa: E402
    NEW_GAP,
    OUTSIDE,
    PREEXISTING_GAP,
    RESIDUAL,
    ComponentSeries,
    classify_components,
    component_series,
    crowns_by_component,
    distance_to_residual,
    gap_patches,
    height_by_distance,
    variance_decomposition,
)

logger = logging.getLogger(__name__)

OUT_DIR = REPO / "results" / "figures" / "review_response"
NAIP_YEARS = (2012, 2014, 2016, 2018, 2020, 2022)

#: State codes for the drawn maps. The first three match the component codes.
LOST = 4
FILLED = 5
STATE_COLORS = {
    OUTSIDE: "#ffffff",
    RESIDUAL: CANOPY_COLOR,
    NEW_GAP: GAP_COLOR,
    PREEXISTING_GAP: MUTED,
    LOST: HIGHLIGHT,
    FILLED: STAGE_COLORS["LSI"],
}
STATE_LABELS = {
    RESIDUAL: "Residual canopy",
    NEW_GAP: "New gap",
    PREEXISTING_GAP: "Pre-existing gap",
    LOST: "Residual lost",
    FILLED: "Gap filled",
}
LINE_COLORS = {"whole": INK, "residual": ACCENT, "new_gap": HIGHLIGHT}
LINE_LABELS = {
    "whole": "Whole stand",
    "residual": "Residual canopy",
    "new_gap": "New gaps",
}

CHIP_IN = 2.1
SERIES_IN = 2.4
PATCH_MIN_PX = 25


@dataclass(frozen=True)
class Variant:
    """One figure: which stands, which image columns, where it lands."""

    stands: tuple[str, ...]
    years: tuple[int, ...]
    slug: str


VARIANTS = {
    "sketch": Variant(("ELKNE-U16-0-0a",), (2012, 2016, 2022), "sketch_u16a"),
    "divergent": Variant(("ELKNE-U9-0-0",), (2016, 2018, 2022), "divergent_u9"),
    "interaction": Variant(("ELKNE-U9-0-0",), (2018, 2020, 2022), "interaction_u9"),
    "multiples": Variant(
        (
            "ELKNE-U16-0-0a",
            "ELKNE-U9-0-0",
            "ELKNE-U4-0-0",
            "ELKNE-U13-0-0",
            "ELKNE-U12-0-0",
        ),
        (),
        "multiples",
    ),
}


@dataclass(frozen=True)
class StandCase:
    """One stand with its component split computed at full resolution."""

    stand_id: str
    stand: object
    ref_year: int | None
    first_post_year: int
    last_pre: float | None
    first_post: float | None
    series: ComponentSeries
    table: pd.DataFrame

    @property
    def geometry(self):
        return self.stand.geometry

    @property
    def label(self) -> str:
        return f"{self.stand_id}\n{self.stand.get('dist_label', '')}"


def stand_case(context: ExampleContext, site: ExampleSite, stand_id: str) -> StandCase:
    """Resolve the reference and first post-disturbance dates and split the stand."""
    stand = stand_row(context.stands, stand_id)
    last_pre, first_post = disturbance_years(stand)
    ref = max((y for y in NAIP_YEARS if last_pre and y <= last_pre), default=None)
    post = min(y for y in NAIP_YEARS if first_post and y >= first_post)
    series = component_series(
        {y: site.chm(y) for y in NAIP_YEARS},
        stand.geometry,
        first_post_year=post,
        ref_year=ref,
        scale=site.chm_scale,
        stand_id=stand_id,
    )
    table = stand_metric_series(context.metrics, stand_id)
    whole = series.frame[series.frame["component"] == "whole"].set_index("year")
    for year in NAIP_YEARS:
        expected = table.set_index("year").loc[year]
        check_value(
            f"{stand_id} {year} crown_fraction",
            whole.loc[year, "canopy_fraction"],
            expected["crown_fraction"],
            tolerance=1e-6,
        )
    logger.info(
        "%s: %s, reference %s, first post-disturbance date %d",
        stand_id,
        stand.get("dist_label", "unlabelled"),
        ref,
        post,
    )
    return StandCase(stand_id, stand, ref, post, last_pre, first_post, series, table)


def display_block(site: ExampleSite, case: StandCase, bounds, year: int, max_px: int):
    """Read one year's canopy height over the chip bounds."""
    return read_band_window(site.chm(year), bounds, scale=site.chm_scale, max_px=max_px)


def display_components(site: ExampleSite, case: StandCase, bounds, max_px: int):
    """Component codes on the displayed block, built the way the series was."""
    post, transform = display_block(site, case, bounds, case.first_post_year, max_px)
    ref = None
    if case.ref_year is not None:
        ref, _ = display_block(site, case, bounds, case.ref_year, max_px)
    inside = stand_mask(case.geometry, post.shape, transform)
    codes = classify_components(post, inside, chm_ref=ref)
    return codes, transform, inside


def state_codes(codes: np.ndarray, chm: np.ndarray) -> np.ndarray:
    """Add the fill and loss states for one later date to the component codes."""
    state = codes.copy()
    canopy = chm >= CANOPY_THRESHOLD_M
    state[(codes == NEW_GAP) & canopy] = FILLED
    state[(codes == RESIDUAL) & ~canopy & np.isfinite(chm)] = LOST
    return state


def state_panel(ax, state: np.ndarray, transform, geometry, invalid=None):
    colors = [STATE_COLORS[code] for code in range(FILLED + 1)]
    return class_panel(
        ax, state, transform, colors=colors, geometry=geometry, invalid=invalid
    )


def state_key(cax, codes: list[int]) -> None:
    class_key(
        cax,
        [STATE_COLORS[c] for c in codes],
        [STATE_LABELS[c] for c in codes],
        fontsize=6.5,
    )


def image_panel(ax, site: ExampleSite, year: int, bounds, geometry, max_px: int) -> str:
    """NAIP when it is on disk, otherwise the canopy height model."""
    try:
        rgb_panel(
            ax, site.naip(year), bounds, geometry=geometry, gamma=1.2, max_px=max_px
        )
        return "NAIP"
    except FileNotFoundError:
        logger.warning("%d: NAIP missing, drawing canopy height instead", year)
        chm_panel(
            ax,
            site.chm(year),
            bounds,
            geometry=geometry,
            scale=site.chm_scale,
            max_px=max_px,
        )
        return "Canopy height"


def component_lines(
    ax,
    case: StandCase,
    metric: str,
    *,
    label: str,
    ylim: tuple[float, float] | None,
    components: tuple[str, ...] = ("whole", "residual", "new_gap"),
    legend: bool = False,
) -> None:
    """One metric against time, one line per component."""
    frame = case.series.frame
    native = case.table.set_index("year")["native_res_m"]
    for component in components:
        rows = frame[frame["component"] == component].sort_values("year")
        if rows.empty:
            continue
        metric_panel(
            ax,
            rows["year"],
            rows[metric],
            label=label,
            ylim=ylim,
            native_res_m=native.loc[rows["year"]].to_numpy(),
            last_pre=case.last_pre,
            first_post=case.first_post,
            color=LINE_COLORS[component],
            series_label=LINE_LABELS[component],
        )
    ax.set_xticks(list(NAIP_YEARS))
    if legend:
        ax.legend(loc="best", fontsize=7)


def crowns_panel(
    ax, site, case, year, bounds, codes, transform, max_px
) -> dict[str, int]:
    """Imagery with crowns filled by the component under their centroid."""
    image_panel(ax, site, year, bounds, case.geometry, max_px)
    crowns = load_stand_crowns(site, year, case.geometry)
    membership = crowns_by_component(crowns, codes, transform)
    inverse = ~transform
    paths = [_polygon_path(geom, inverse) for geom in crowns.geometry]
    colors = [
        LINE_COLORS["residual"] if m == RESIDUAL else LINE_COLORS["new_gap"]
        for m in membership
    ]
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.add_collection(
        PathCollection(
            paths,
            facecolors=colors,
            edgecolors="white",
            linewidths=0.4,
            alpha=0.6,
            zorder=4,
        )
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    counts = {
        "residual": int((membership == RESIDUAL).sum()),
        "new_gap": int((membership == NEW_GAP).sum()),
    }
    logger.info("%d crowns: %s", year, counts)
    return counts


def crown_key(cax) -> None:
    class_key(
        cax,
        [LINE_COLORS["residual"], LINE_COLORS["new_gap"]],
        ["Crown in residual", "Crown in gap"],
        fontsize=6.5,
    )


# --------------------------------------------------------------------------- variants


def figure_sketch(site, context, variant: Variant, args) -> Path:
    case = stand_case(context, site, variant.stands[0])
    years = list(variant.years)
    n = len(years)
    bounds = padded_bounds(
        case.geometry,
        pad_fraction=args.pad_fraction,
        min_pad_m=args.min_pad_m,
        square=True,
    )
    codes, transform, inside = display_components(site, case, bounds, args.max_px)

    fig = plt.figure(
        figsize=(CHIP_IN * n + 1.5, CHIP_IN * 3 + SERIES_IN), constrained_layout=True
    )
    fig.get_layout_engine().set(h_pad=0.03, w_pad=0.03, hspace=0.02, wspace=0.02)
    grid = fig.add_gridspec(
        4,
        n + 1,
        width_ratios=[1.0] * n + [0.06],
        height_ratios=[1, 1, 1, SERIES_IN / CHIP_IN],
    )
    image_axes = [fig.add_subplot(grid[0, c]) for c in range(n)]
    state_axes = [fig.add_subplot(grid[1, c]) for c in range(n)]
    crown_axes = [fig.add_subplot(grid[2, c]) for c in range(n)]

    image_label = "NAIP"
    for column, year in enumerate(years):
        image_label = image_panel(
            image_axes[column], site, year, bounds, case.geometry, args.max_px
        )
        chm, _ = display_block(site, case, bounds, year, args.max_px)
        state = codes if year == case.first_post_year else state_codes(codes, chm)
        state_panel(
            state_axes[column],
            state,
            transform,
            case.geometry,
            invalid=inside & ~np.isfinite(chm),
        )
        crowns_panel(
            crown_axes[column], site, case, year, bounds, codes, transform, args.max_px
        )

    column_year_labels(image_axes, years)
    row_label(image_axes[0], image_label)
    row_label(state_axes[0], f"Components from {case.first_post_year}")
    row_label(crown_axes[0], "Crowns by component")
    state_key(fig.add_subplot(grid[1, n]), [RESIDUAL, NEW_GAP, LOST, FILLED])
    crown_key(fig.add_subplot(grid[2, n]))

    sub = grid[3, :].subgridspec(1, 2, wspace=0.28)
    component_lines(
        fig.add_subplot(sub[0, 0]),
        case,
        "height_p90",
        label="Height P90, all pixels (m)",
        ylim=(0, 32),
        legend=True,
    )
    component_lines(
        fig.add_subplot(sub[0, 1]),
        case,
        "height_cv",
        label="Height CV, all pixels",
        ylim=None,
    )
    return finish(
        fig, args.out_dir / f"{variant.slug}.png", dpi=args.dpi, optimize=args.optimize
    )


def figure_divergent(site, context, variant: Variant, args) -> Path:
    case = stand_case(context, site, variant.stands[0])
    years = list(variant.years)
    n = len(years)
    bounds = padded_bounds(
        case.geometry,
        pad_fraction=args.pad_fraction,
        min_pad_m=args.min_pad_m,
        square=True,
    )
    codes, transform, inside = display_components(site, case, bounds, args.max_px)

    fig = plt.figure(
        figsize=(CHIP_IN * n + 1.5, CHIP_IN * 3 + SERIES_IN), constrained_layout=True
    )
    fig.get_layout_engine().set(h_pad=0.03, w_pad=0.03, hspace=0.02, wspace=0.02)
    grid = fig.add_gridspec(
        4,
        n + 1,
        width_ratios=[1.0] * n + [0.06],
        height_ratios=[1, 1, 1, SERIES_IN / CHIP_IN],
    )
    image_axes = [fig.add_subplot(grid[0, c]) for c in range(n)]
    chm_axes = [fig.add_subplot(grid[1, c]) for c in range(n)]
    state_axes = [fig.add_subplot(grid[2, c]) for c in range(n)]

    image_label, chm_image = "NAIP", None
    for column, year in enumerate(years):
        image_label = image_panel(
            image_axes[column], site, year, bounds, case.geometry, args.max_px
        )
        chm_image, _ = chm_panel(
            chm_axes[column],
            site.chm(year),
            bounds,
            geometry=case.geometry,
            scale=site.chm_scale,
            max_px=args.max_px,
        )
        chm, _ = display_block(site, case, bounds, year, args.max_px)
        if case.ref_year is not None and year <= case.ref_year:
            state = np.where(
                inside & (chm >= CANOPY_THRESHOLD_M),
                RESIDUAL,
                np.where(inside, PREEXISTING_GAP, OUTSIDE),
            ).astype(np.uint8)
        elif year == case.first_post_year:
            state = codes
        else:
            state = state_codes(codes, chm)
        state_panel(
            state_axes[column],
            state,
            transform,
            case.geometry,
            invalid=inside & ~np.isfinite(chm),
        )

    column_year_labels(image_axes, years)
    row_label(image_axes[0], image_label)
    row_label(chm_axes[0], "Canopy height")
    row_label(
        state_axes[0], f"Components from {case.ref_year} to {case.first_post_year}"
    )
    fig.colorbar(chm_image, cax=fig.add_subplot(grid[1, n]), label="Canopy height (m)")
    state_key(
        fig.add_subplot(grid[2, n]), [RESIDUAL, NEW_GAP, PREEXISTING_GAP, LOST, FILLED]
    )

    sub = grid[3, :].subgridspec(1, 2, wspace=0.28)
    component_lines(
        fig.add_subplot(sub[0, 0]),
        case,
        "canopy_fraction",
        label=f"Fraction ≥ {CANOPY_THRESHOLD_M:g} m",
        ylim=(0, 1.02),
        legend=True,
    )
    component_lines(
        fig.add_subplot(sub[0, 1]),
        case,
        "canopy_height_p90",
        label=f"Height P90 of pixels ≥ {CANOPY_THRESHOLD_M:g} m",
        ylim=(0, 32),
    )
    return finish(
        fig, args.out_dir / f"{variant.slug}.png", dpi=args.dpi, optimize=args.optimize
    )


def figure_interaction(site, context, variant: Variant, args) -> Path:
    case = stand_case(context, site, variant.stands[0])
    series = case.series
    pixel = series.window.pixel_size_m
    years = list(variant.years)
    last_year = years[-1]

    # Distance is measured to any canopy on the first post-disturbance date,
    # inside or outside the polygon. A gap pixel at the stand edge sits next to
    # intact forest whether or not that forest is part of the stand, so the
    # block is read over the padded bounds at native resolution.
    bounds = padded_bounds(
        case.geometry,
        pad_fraction=args.pad_fraction,
        min_pad_m=args.min_pad_m,
        square=True,
    )

    def block(year: int) -> tuple[np.ndarray, object]:
        return read_band_window(
            site.chm(year), bounds, scale=site.chm_scale, max_px=None
        )

    post, transform = block(case.first_post_year)
    inside = stand_mask(case.geometry, post.shape, transform)
    ref = None if case.ref_year is None else block(case.ref_year)[0]
    codes = classify_components(post, inside, chm_ref=ref)
    canopy_anywhere = np.where(post >= CANOPY_THRESHOLD_M, RESIDUAL, OUTSIDE)
    distance = distance_to_residual(canopy_anywhere.astype(np.uint8), pixel)
    chm_by_year = {y: block(y)[0] for y in years}
    bins = [0, 3, 6, 9, 12, 18, 24, 36]
    by_distance = {
        y: height_by_distance(chm_by_year[y], codes, distance, bins_m=bins)
        for y in years
    }
    variance = pd.DataFrame(
        {
            y: variance_decomposition(series.chm_by_year[y], series.components)
            for y in NAIP_YEARS
        }
    ).T
    labels = gap_patches(codes, min_area_px=PATCH_MIN_PX)
    logger.info("%d gap patches of at least %d pixels", int(labels.max()), PATCH_MIN_PX)

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.6), constrained_layout=True)

    # (a) height inside the new gaps at the last date, everything else grey.
    height = chm_by_year[last_year].copy()
    height[codes != NEW_GAP] = np.nan
    ax = axes[0, 0]
    mappable, _ = band_panel(
        ax, height, transform, cmap="YlGn", vmin=0, vmax=10, geometry=case.geometry
    )
    fig.colorbar(
        mappable, ax=ax, shrink=0.8, label=f"Height in new gaps, {last_year} (m)"
    )
    ax.set_title("New gaps only, residual canopy grey", fontsize=8)

    # (b) gap height against distance to canopy, one line per year.
    shades = [GRID, ACCENT, INK][-len(years) :]
    ax = axes[0, 1]
    for shade, year in zip(shades, years, strict=True):
        table = by_distance[year]
        mid = (table["bin_lo"] + table["bin_hi"]) / 2
        ax.plot(
            mid, table["height_p90"], "-o", color=shade, ms=4, lw=1.4, label=str(year)
        )
    ax.set_xlabel(f"Distance to canopy in {case.first_post_year} (m)")
    ax.set_ylabel("Gap height P90, all pixels (m)")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.35)

    # (c) gap fill against the same distance.
    ax = axes[1, 0]
    for shade, year in zip(shades, years, strict=True):
        table = by_distance[year]
        mid = (table["bin_lo"] + table["bin_hi"]) / 2
        ax.plot(
            mid,
            table["canopy_fraction"],
            "-o",
            color=shade,
            ms=4,
            lw=1.4,
            label=str(year),
        )
    ax.set_xlabel(f"Distance to canopy in {case.first_post_year} (m)")
    ax.set_ylabel(f"Gap fraction ≥ {CANOPY_THRESHOLD_M:g} m")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.35)

    # (d) variance split per date.
    ax = axes[1, 1]
    parts = [
        ("within_residual", LINE_COLORS["residual"], "Within residual"),
        ("within_new_gap", LINE_COLORS["new_gap"], "Within new gaps"),
        ("within_preexisting_gap", MUTED, "Within pre-existing gaps"),
        ("between", GRID, "Between components"),
    ]
    bottom = np.zeros(len(variance))
    for column, color, name in parts:
        if column not in variance:
            continue
        values = variance[column].fillna(0).to_numpy()
        ax.bar(
            variance.index,
            values,
            bottom=bottom,
            color=color,
            width=1.4,
            label=name,
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += values
    ax.set_xticks(list(NAIP_YEARS))
    ax.set_ylabel("Whole-stand height variance (m²)")
    ax.legend(fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.grid(alpha=0.35, axis="y")
    twin = ax.twinx()
    twin.plot(variance.index, variance["cv"], "-o", color=INK, ms=4, lw=1.2)
    twin.set_ylabel("Whole-stand height CV", color=INK)
    twin.set_ylim(0, max(2.5, float(variance["cv"].max()) * 1.1))
    return finish(
        fig, args.out_dir / f"{variant.slug}.png", dpi=args.dpi, optimize=args.optimize
    )


def figure_multiples(site, context, variant: Variant, args) -> Path:
    cases = [stand_case(context, site, sid) for sid in variant.stands]
    n = len(cases)
    fig = plt.figure(figsize=(8.2, 1.9 * n + 0.4), constrained_layout=True)
    grid = fig.add_gridspec(n, 4, width_ratios=[1.0, 0.07, 1.5, 1.5])
    for row, case in enumerate(cases):
        bounds = padded_bounds(
            case.geometry,
            pad_fraction=args.pad_fraction,
            min_pad_m=args.min_pad_m,
            square=True,
        )
        codes, transform, inside = display_components(site, case, bounds, args.max_px)
        ax_map = fig.add_subplot(grid[row, 0])
        state_panel(ax_map, codes, transform, case.geometry)
        ax_map.set_ylabel(case.label, fontsize=7, color=MUTED)
        ax_map.set_title(
            f"Components from {case.first_post_year}", fontsize=7, color=MUTED
        )
        ax_a = fig.add_subplot(grid[row, 2])
        ax_b = fig.add_subplot(grid[row, 3])
        component_lines(
            ax_a,
            case,
            "canopy_fraction",
            label=f"Fraction ≥ {CANOPY_THRESHOLD_M:g} m",
            ylim=(0, 1.02),
            legend=(row == 0),
        )
        component_lines(
            ax_b, case, "height_p90", label="Height P90, all pixels (m)", ylim=(0, 34)
        )
        if row < n - 1:
            for ax in (ax_a, ax_b):
                ax.set_xticklabels([])
    state_key(fig.add_subplot(grid[n // 2, 1]), [RESIDUAL, NEW_GAP, PREEXISTING_GAP])
    return finish(
        fig, args.out_dir / f"{variant.slug}.png", dpi=args.dpi, optimize=args.optimize
    )


BUILDERS = {
    "sketch": figure_sketch,
    "divergent": figure_divergent,
    "interaction": figure_interaction,
    "multiples": figure_multiples,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variant", default="all", choices=["all", *VARIANTS])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--pad-fraction", type=float, default=0.12)
    parser.add_argument("--min-pad-m", type=float, default=25.0)
    parser.add_argument("--max-px", type=int, default=500)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")
    return parser


def main() -> int:
    configure_logging()
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    site = elkinsville()
    context = load_example_context(site)
    names = list(VARIANTS) if args.variant == "all" else [args.variant]
    for name in names:
        written = BUILDERS[name](site, context, VARIANTS[name], args)
        logger.info("Wrote %s", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
