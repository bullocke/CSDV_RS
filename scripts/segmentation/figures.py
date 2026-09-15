"""Build every figure in the segmentation optimization report.

Each function makes one figure and returns its path, so a single figure can be
rebuilt without redoing the rest. All of them read tables written by the
analysis scripts, so a figure never recomputes a number the report cites.

Figure conventions follow ``csdv_core.viz.style`` and the rest of the project:
axis labels, a legend where one is needed, and nothing else. Values, caveats
and interpretation live in the report text.

Run:
    .micromamba/envs/CSDV/bin/python scripts/segmentation/figures.py
    .micromamba/envs/CSDV/bin/python scripts/segmentation/figures.py --only f5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    FIGURE_DIR,
    REFERENCE_YEAR,
    configure_logging,
    crown_width_open_grown,
    crown_width_popescu,
    elkinsville_chm,
    load_tiles,
    params_from_row,
    read_table,
    read_window,
    segment_tile,
)

from csdv_core.segmentation.chm_watershed import segment_crowns  # noqa: E402
from csdv_core.segmentation.params import (  # noqa: E402
    DEFAULT_PARAMS,
    LEGACY_AS_SHIPPED_PARAMS,
    SegmentationParams,
)
from csdv_core.viz.style import (  # noqa: E402
    ACCENT,
    GRID,
    HIGHLIGHT,
    INK,
    MUTED,
    STAGE_COLORS,
    save_fig,
    setup_style,
)

logger = logging.getLogger("figures")

#: The set the report settles on. The sweep table alone would return the
#: rule's winner, which the transfer sites rejected, so the figures use the
#: production default instead.
CHOSEN_FALLBACK = DEFAULT_PARAMS


def chosen_params() -> SegmentationParams:
    """The production parameter set the report settles on."""
    return DEFAULT_PARAMS


def rule_winner() -> SegmentationParams:
    """What the pre-registered rule picked from the tuning tiles alone."""
    try:
        scored = read_table("sweep_tune_scored.parquet")
    except FileNotFoundError:
        return DEFAULT_PARAMS
    passing = scored[scored["passes"]]
    return params_from_row(passing.iloc[0]) if len(passing) else DEFAULT_PARAMS


# ---------------------------------------------------------------------------
def f1_the_bug() -> Path:
    """Two peaks collapsing, and what that did to tree-top density."""
    from skimage.feature import peak_local_max

    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))

    ax = axes[0]
    separations = np.arange(3, 15)
    collapse = []
    for d in separations:
        yy, xx = np.mgrid[0:80, 0:80]
        img = np.maximum(
            10.0 - 0.5 * np.hypot(yy - 40, xx - 25),
            9.5 - 0.5 * np.hypot(yy - 40, xx - 25 - d),
        )
        img[img < 0] = 0
        first = next(
            (
                m
                for m in range(1, 16)
                if len(peak_local_max(img, min_distance=m, exclude_border=False)) == 1
            ),
            np.nan,
        )
        collapse.append(first)
    # The measured curve lies exactly on 1:1, which is the whole point, so the
    # reference is drawn wide and underneath rather than hidden by the data.
    ax.plot(separations, separations, "-", color=GRID, lw=5.0, zorder=1)
    ax.plot(separations, collapse, "o-", color=ACCENT, ms=4, lw=1.4, zorder=2)
    ax.set_xlabel("Separation between two peaks (pixels)")
    ax.set_ylabel("min_distance that merges them")
    ax.legend(
        handles=[
            Line2D([], [], color=ACCENT, marker="o", ms=4, label="measured"),
            Line2D([], [], color=GRID, lw=5.0, label="1:1"),
        ],
        frameon=False,
        loc="upper left",
    )

    ax = axes[1]
    labels = ["as shipped", "same window,\ncorrect semantics", "chosen"]
    values = [10.3, 39.3, np.nan]
    try:
        scored = read_table("sweep_tune_scored.parquet")
        row = scored[scored["key"] == chosen_params().key]
        if len(row):
            values[2] = float(row.iloc[0]["density_per_ha"])
    except FileNotFoundError:
        pass
    colours = [HIGHLIGHT, ACCENT, STAGE_COLORS["LSE"]]
    bars = ax.bar(labels, values, color=colours, width=0.6)
    for bar, value in zip(bars, values, strict=True):
        if np.isfinite(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 3,
                f"{value:.0f}",
                ha="center",
                fontsize=8,
                color=INK,
            )
    ax.axhspan(75, 200, color=GRID, alpha=0.5, zorder=0)
    ax.set_ylabel("Tree tops per hectare")
    ax.tick_params(axis="x", labelsize=7.5)
    return save_fig(fig, FIGURE_DIR / "f1_window_semantics.png")


def f2_sweep_surface() -> Path:
    """Crown density across the swept grid."""
    setup_style()
    scored = read_table("sweep_tune_scored.parquet")
    windows = list(dict.fromkeys(scored["window"]))
    fig, axes = plt.subplots(
        1, len(windows), figsize=(2.1 * len(windows) + 1.0, 2.9), sharey=True
    )
    axes = np.atleast_1d(axes)
    grids = []
    for window, ax in zip(windows, axes, strict=True):
        sub = scored[scored["window"] == window]
        grid = sub.pivot_table(
            index="smooth_radius_m", columns="th_cr", values="density_per_ha"
        )
        grids.append((ax, window, grid))
    vmax = max(float(np.nanmax(g.to_numpy())) for _, _, g in grids)
    for ax, window, grid in grids:
        im = ax.imshow(
            grid.to_numpy(),
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=0,
            vmax=vmax,
        )
        ax.set_xticks(range(len(grid.columns)))
        ax.set_xticklabels([f"{c:g}" for c in grid.columns], fontsize=7)
        ax.set_yticks(range(len(grid.index)))
        ax.set_yticklabels([f"{i:g}" for i in grid.index], fontsize=7)
        ax.set_title(window, fontsize=8, color=INK)
        ax.set_xlabel("th_cr")
        # Contour the plausible band so the reachable region is visible.
        vals = grid.to_numpy()
        if np.nanmin(vals) < 200 < np.nanmax(vals) or np.nanmin(vals) < 75:
            ax.contour(
                vals,
                levels=[75, 200],
                colors=["white"],
                linewidths=1.0,
                linestyles=["--", "-"],
            )
    axes[0].set_ylabel("Smoothing radius (m)")
    fig.colorbar(im, ax=axes.tolist(), label="Crowns per hectare", pad=0.02)
    return save_fig(fig, FIGURE_DIR / "f2_sweep_surface.png")


def f3_allometry() -> Path:
    """Observed crown width against tree-top height, three ways."""
    setup_style()
    tiles = load_tiles("tune")
    tiles = tiles[tiles["stratum"].str.startswith("undisturbed")]
    chm = elkinsville_chm(REFERENCE_YEAR)
    chosen = chosen_params()
    variants = [
        ("Legacy window, reconstructed", LEGACY_AS_SHIPPED_PARAMS),
        ("Chosen", chosen),
    ]

    fig, axes = plt.subplots(1, len(variants), figsize=(3.5 * len(variants), 3.3))
    axes = np.atleast_1d(axes)
    for ax, (name, params) in zip(axes, variants, strict=True):
        heights, widths = [], []
        for _, tile in tiles.iterrows():
            bounds = (tile["minx"], tile["miny"], tile["maxx"], tile["maxy"])
            crowns, _ = segment_tile(chm, bounds, params)
            if len(crowns):
                heights.append(crowns["apex_h_m"].to_numpy())
                widths.append(crowns["crown_diam_m"].to_numpy())
        if not heights:
            continue
        h = np.concatenate(heights)
        w = np.concatenate(widths)
        ax.hexbin(h, w, gridsize=38, cmap="Blues", mincnt=1, linewidths=0)
        grid = np.linspace(2, 38, 100)
        ax.plot(
            grid, crown_width_popescu(grid), color=HIGHLIGHT, lw=1.6, label="Popescu"
        )
        ax.plot(
            grid,
            crown_width_open_grown(grid),
            color=INK,
            lw=1.4,
            ls="--",
            label="open-grown ceiling",
        )
        ax.set_xlabel("Tree-top height (m)")
        ax.set_title(name, fontsize=9, color=INK)
        ax.set_ylim(0, 45)
        ax.set_xlim(2, 38)
    axes[0].set_ylabel("Crown diameter (m)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=7.5)
    return save_fig(fig, FIGURE_DIR / "f3_allometry.png")


def f5_nulls() -> Path:
    """Crown diameter CV for the real data and every null model."""
    setup_style()
    from nulls import POISSON_VORONOI_DIAMETER_CV

    frame = read_table("nulls.parquet")
    order = frame.groupby("model")["diam_cv"].mean().sort_values().index.tolist()
    # The scrambled surface is a sanity control, not a comparison. It sits far
    # off the scale and would flatten everything else.
    order = [m for m in order if "scrambled" not in m]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    data = [frame[frame["model"] == m]["diam_cv"].dropna().to_numpy() for m in order]
    parts = ax.boxplot(
        data,
        vert=False,
        widths=0.6,
        patch_artist=True,
        medianprops={"color": INK},
        flierprops={"marker": ".", "ms": 3, "mfc": MUTED, "mec": MUTED},
    )
    for patch, model in zip(parts["boxes"], order, strict=True):
        patch.set_facecolor(
            STAGE_COLORS["LSE"] if "real CHM" in model else STAGE_COLORS["ESI"]
        )
        patch.set_edgecolor(INK)
    ax.axvline(
        POISSON_VORONOI_DIAMETER_CV,
        color=HIGHLIGHT,
        ls="--",
        lw=1.4,
        label="Poisson-Voronoi theory",
    )
    ax.set_yticklabels(order, fontsize=7.5)
    ax.set_xlabel("Crown diameter CV")
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    return save_fig(fig, FIGURE_DIR / "f5_nulls.png")


def f6_sensitivity() -> Path:
    """Measured crown CV against the CV that was planted."""
    setup_style()
    frame = read_table("sensitivity.parquet")
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    markers = {"0.6 m native": "o", "degraded to 1.0 m": "s"}
    colours = {"unbounded": MUTED, "bounded": ACCENT}
    for (variant, res), grp in frame.groupby(["variant", "resolution"]):
        ok = grp.dropna(subset=["measured_cv"])
        if ok.empty:
            continue
        agg = ok.groupby("planted_cv")["measured_cv"].mean()
        ax.plot(
            agg.index,
            agg.to_numpy(),
            marker=markers.get(res, "o"),
            ms=4,
            lw=1.3,
            color=colours.get(variant, INK),
            ls="-" if res == "0.6 m native" else "--",
            label=f"{variant}, {res}",
        )
    lim = [0, max(frame["planted_cv"].max(), frame["measured_cv"].max()) * 1.05]
    ax.plot(lim, lim, color=INK, lw=1.0, ls=":", label="1:1")
    ax.set_xlabel("Planted crown size CV")
    ax.set_ylabel("Measured crown diameter CV")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    return save_fig(fig, FIGURE_DIR / "f6_sensitivity.png")


def f7_stability() -> Path:
    """Six-year series on undisturbed tiles, and the degradation control."""
    setup_style()
    frame = read_table("stability.parquet")
    observed = frame[frame["source"] == "observed"]
    degraded = frame[frame["source"] != "observed"]
    metrics = ["density_per_ha", "diam_mean", "diam_cv"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.0 * len(metrics), 3.0))
    for ax, metric in zip(axes, metrics, strict=True):
        base = observed.groupby("tile_id")[metric].transform("mean")
        dev = observed.assign(dev=observed[metric] - base)
        for _tile_id, grp in dev.groupby("tile_id"):
            grp = grp.sort_values("year")
            ax.plot(grp["year"], grp["dev"], color=GRID, lw=0.9, zorder=1)
        mean_dev = dev.groupby("year")["dev"].mean()
        ax.plot(mean_dev.index, mean_dev.to_numpy(), color=ACCENT, lw=1.8, zorder=3)
        coarse = dev[dev["native_res_m"] == 1.0]
        ax.scatter(coarse["year"], coarse["dev"], s=14, color=HIGHLIGHT, zorder=4)
        if len(degraded):
            merged = degraded.merge(
                observed[observed["year"] == REFERENCE_YEAR][["tile_id", metric]],
                on="tile_id",
                suffixes=("_d", "_o"),
            )
            offset = (merged[f"{metric}_d"] - merged[f"{metric}_o"]).mean()
            ax.scatter(
                [REFERENCE_YEAR],
                [offset],
                marker="D",
                s=34,
                facecolor="none",
                edgecolor=INK,
                lw=1.3,
                zorder=5,
            )
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(metric, fontsize=8.5, color=INK)
        ax.set_xlabel("Year")
    axes[0].set_ylabel("Deviation from tile mean")
    axes[-1].legend(
        handles=[
            Line2D([], [], color=ACCENT, lw=1.8, label="mean over tiles"),
            Line2D([], [], marker="o", ls="", color=HIGHLIGHT, label="1.0 m native"),
            Line2D(
                [],
                [],
                marker="D",
                ls="",
                mfc="none",
                mec=INK,
                label="2016 degraded to 1.0 m",
            ),
        ],
        frameon=False,
        fontsize=6.5,
        loc="best",
    )
    return save_fig(fig, FIGURE_DIR / "f7_stability.png")


def f8_support() -> Path:
    """How wide the crown_cv interval is at each sample size."""
    setup_style()
    frame = read_table("support.parquet")
    band = float(frame["band_width"].iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.1))

    ax = axes[0]
    ax.fill_between(
        frame["n_crowns"],
        frame["ci90_low"],
        frame["ci90_high"],
        color=STAGE_COLORS["ESE"],
        alpha=0.45,
    )
    ax.plot(frame["n_crowns"], frame["cv_mean"], color=INK, lw=1.4)
    ax.set_xscale("log")
    ax.set_xlabel("Crowns in the sample")
    ax.set_ylabel("crown_cv")

    ax = axes[1]
    ax.plot(
        frame["n_crowns"], frame["ci90_width"], color=ACCENT, lw=1.6, label="bootstrap"
    )
    ax.plot(
        frame["n_crowns"],
        3.29 * frame["analytic_se"],
        color=MUTED,
        ls="--",
        lw=1.2,
        label="analytic",
    )
    ax.axhline(band, color=HIGHLIGHT, lw=1.4, label="narrowest stage band")
    fits = frame[frame["fits_in_band"]]
    if len(fits):
        ax.axvline(float(fits["n_crowns"].iloc[0]), color=INK, ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Crowns in the sample")
    ax.set_ylabel("Width of the 90% interval")
    ax.legend(frameon=False, fontsize=7)
    return save_fig(fig, FIGURE_DIR / "f8_support.png")


def f9_transfer() -> Path:
    """Crown width distributions at all three sites under one parameter set."""
    setup_style()
    from _lib import transfer_sites

    chosen = chosen_params()
    tiles = load_tiles()
    sources = {"ElkinsvilleNE": elkinsville_chm(REFERENCE_YEAR)}
    for site in transfer_sites():
        sources[site.name] = site.chm

    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    palette = {
        "ElkinsvilleNE": ACCENT,
        "SCBI": HIGHLIGHT,
        "HARV": STAGE_COLORS["MA_OW"],
    }
    for site, chm in sources.items():
        sub = tiles[tiles["site"] == site]
        if site == "ElkinsvilleNE":
            sub = sub[sub["stratum"].str.startswith("undisturbed")]
        pooled = []
        for _, tile in sub.iterrows():
            bounds = (tile["minx"], tile["miny"], tile["maxx"], tile["maxy"])
            crowns, _ = segment_tile(chm, bounds, chosen)
            if len(crowns):
                pooled.append(crowns["crown_diam_m"].to_numpy())
        if not pooled:
            continue
        d = np.concatenate(pooled)
        ax.hist(
            d,
            bins=np.arange(0, 40, 1.0),
            density=True,
            histtype="step",
            lw=1.6,
            color=palette.get(site, INK),
            label=f"{site} (n={d.size})",
        )
    ax.set_xlabel("Crown diameter (m)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=7.5)
    return save_fig(fig, FIGURE_DIR / "f9_transfer.png")


def f10_before_after() -> Path:
    """The same ground segmented under the old and the chosen parameters."""
    setup_style()
    tiles = load_tiles("tune")
    tiles = tiles[tiles["stratum"] == "undisturbed_tall"]
    tile = tiles.iloc[0]
    # A 150 m detail, or the crowns are too small to see.
    cx = (tile["minx"] + tile["maxx"]) / 2
    cy = (tile["miny"] + tile["maxy"]) / 2
    half = 75.0
    bounds = (cx - half, cy - half, cx + half, cy + half)
    arr, transform, crs = read_window(elkinsville_chm(REFERENCE_YEAR), bounds)

    # The left panel reads the crowns the old pipeline actually wrote, rather
    # than reconstructing them, so the comparison is against the real product.
    import geopandas as gpd

    from csdv_core.io.paths import project_paths

    archived = (
        project_paths().results_root
        / "stands"
        / "ElkinsvilleNE"
        / "crowns"
        / f"crowns_{REFERENCE_YEAR}.gpkg"
    )
    old = (
        gpd.read_file(archived, bbox=bounds)
        if archived.exists()
        else segment_crowns(arr, transform, crs, params=LEGACY_AS_SHIPPED_PARAMS)
    )
    new = segment_crowns(arr, transform, crs, params=chosen_params())
    variants = [("As shipped", old), ("Chosen", new)]

    fig, axes = plt.subplots(1, len(variants), figsize=(3.4 * len(variants), 3.5))
    for ax, (name, crowns) in zip(axes, variants, strict=True):
        ax.imshow(
            arr,
            cmap="viridis",
            extent=(bounds[0], bounds[2], bounds[1], bounds[3]),
            vmin=0,
            vmax=35,
        )
        crowns.boundary.plot(ax=ax, color="white", linewidth=0.7)
        ax.set_title(name, fontsize=8.5, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
    return save_fig(fig, FIGURE_DIR / "f10_before_after.png")


FIGURES = {
    "f1": f1_the_bug,
    "f2": f2_sweep_surface,
    "f3": f3_allometry,
    "f5": f5_nulls,
    "f6": f6_sensitivity,
    "f7": f7_stability,
    "f8": f8_support,
    "f9": f9_transfer,
    "f10": f10_before_after,
}


def main() -> int:
    """Build the requested figures, skipping any whose table is missing."""
    configure_logging()
    logging.getLogger("csdv_core.segmentation.chm_watershed").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None, choices=list(FIGURES))
    args = parser.parse_args()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    wanted = args.only or list(FIGURES)
    for name in wanted:
        try:
            path = FIGURES[name]()
            logger.info("%s -> %s", name, path.name)
        except FileNotFoundError as exc:
            logger.warning("%s skipped: %s", name, exc)
        except Exception:
            logger.exception("%s failed", name)
        finally:
            plt.close("all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
