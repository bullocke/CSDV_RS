"""Figures for docs/guides/crown_segmentation/, from the comparison tables.

Run after ``fetch_units.py``, ``before_after.py`` and ``compare.py``::

    $P scripts/segmentation/compare_lidr/figures.py           # all figures
    $P scripts/segmentation/compare_lidr/figures.py speed     # one figure

Figures are written to ``docs/guides/crown_segmentation/figures/`` as PNG.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _lib as L
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import strata  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402

log = L.setup_logging("figures")

DOCS_FIG = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "guides"
    / "crown_segmentation"
    / "figures"
)
WORK = L.OUT_DIR / "work"
SURFACE, INK, INK_2, INK_3, GRID = (
    strata.SURFACE,
    strata.INK,
    strata.INK_2,
    strata.INK_3,
    strata.GRID,
)
#: Categorical slots 1 to 4 of the reference palette: one per grower.
GROWER_COLORS = {
    "flood_apex": "#2a78d6",
    "lidr": "#eb6834",
    "post_trim": "#1baf7a",
    "flood_mean": "#eda100",
}
GROWER_LABELS = {
    "flood_apex": "Python flood, apex rule",
    "flood_mean": "Python flood, mean rule",
    "post_trim": "Python post-trim (earlier)",
    "lidr": "lidR dalponte2016",
}
GROWER_ORDER = ("flood_apex", "flood_mean", "post_trim", "lidr")
CHM_CMAP = strata.CHM_CMAP
DPI = 120


def save(fig, name: str, dpi: int = DPI) -> Path:
    DOCS_FIG.mkdir(parents=True, exist_ok=True)
    out = DOCS_FIG / f"{name}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    shrink(out)
    log.info("%s (%.0f KB)", out.name, out.stat().st_size / 1024)
    return out


def shrink(path: Path) -> None:
    """Quantize a PNG to 256 colours when it is over 300 KB.

    Raster maps compress badly as RGB PNG. A palette keeps them legible on
    GitHub at a fraction of the size.
    """
    if path.stat().st_size <= 300_000:
        return
    from PIL import Image

    with Image.open(path) as im:
        im.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(
            path, optimize=True
        )


def clean(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _crown_overlay(ax, h, labels, color, extent):
    ax.imshow(np.nan_to_num(h), cmap=CHM_CMAP, vmin=0, vmax=np.nanmax(h), extent=extent,
              interpolation="nearest")  # fmt: skip
    edge = find_boundaries(labels, mode="inner") & (labels > 0)
    ax.imshow(np.ma.masked_where(labels == 0, labels > 0), cmap=ListedColormap([color]), alpha=0.18,
              extent=extent, interpolation="nearest")  # fmt: skip
    ax.imshow(np.ma.masked_where(~edge, edge), cmap=ListedColormap([color]), extent=extent,
              interpolation="nearest")  # fmt: skip
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
def fig_synthetic() -> Path:
    """Five constructed scenes, four growers, same tree tops."""
    files = sorted((WORK / "synthetic").glob("*.npz"))
    order = ["Unequal_neighbours", "Isolated_dome", "Small_tree_beside_a_hedge", "Wide_and_narrow_crown",
             "Crown_wider_than_the_cap"]  # fmt: skip
    files = [
        WORK / "synthetic" / f"{n}.npz"
        for n in order
        if (WORK / "synthetic" / f"{n}.npz").exists()
    ]
    fig, axes = plt.subplots(
        len(files), 5, figsize=(12.5, 2.6 * len(files)), constrained_layout=True
    )
    for i, f in enumerate(files):
        d = np.load(f)
        h = d["h"]
        ext = (0, h.shape[1] * 0.6, 0, h.shape[0] * 0.6)
        ax = axes[i, 0]
        im = ax.imshow(
            np.nan_to_num(h), cmap=CHM_CMAP, vmin=0, vmax=np.nanmax(h), extent=ext
        )
        ax.plot((d["cols"] + 0.5) * 0.6, ext[3] - (d["rows"] + 0.5) * 0.6, "o", ms=4, color=INK,
                markeredgecolor=SURFACE)  # fmt: skip
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(f.stem.replace("_", " "), fontsize=9, color=INK)
        fig.colorbar(im, ax=ax, shrink=0.75, label="m")
        if i == 0:
            ax.set_title("CHM and tree tops", loc="left", fontsize=9)
        for j, g in enumerate(GROWER_ORDER, start=1):
            _crown_overlay(axes[i, j], h, d[g], GROWER_COLORS[g], ext)
            if i == 0:
                axes[i, j].set_title(GROWER_LABELS[g], loc="left", fontsize=9)
    return save(fig, "synthetic_scenes")


# ---------------------------------------------------------------------------
def fig_detection() -> Path:
    """Share of tree tops both detectors find, and why the rest differ."""
    t = pd.read_csv(L.TABLE_DIR / "detect.csv")
    cats = [c for c in t.columns if c.startswith(("py_only_", "lidr_only_"))]
    g = t.groupby(["metric", "kind", "window"])[
        ["n_python", "n_lidr", "n_match", *cats]
    ].sum()
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 3.6), constrained_layout=True, sharey=True
    )
    windows = ["fixed_3m", "fixed_5m", "fixed_8m", "tallo"]
    for ax, kind, title in zip(
        axes, ("naip", "lidar"), ("NAIP-CHM, 0.6 m", "NEON lidar CHM, 1 m"), strict=True
    ):
        x = np.arange(len(windows))
        for k, (metric, color, marker) in enumerate(
            (("pixel", "#1baf7a", "D"), ("exact", "#2a78d6", "o"))
        ):
            sub = g.loc[metric].loc[kind].reindex(windows)
            share = sub["n_match"] / np.maximum(sub["n_python"], sub["n_lidr"])
            ax.plot(x + (k - 0.5) * 0.12, share, marker, color=color, ms=7, markeredgecolor=SURFACE,
                    label=f"Python window radius {'rounded to pixels' if metric == 'pixel' else 'exact'}")  # fmt: skip
        ax.set_xticks(x, ["3 m", "5 m", "8 m", "Tallo"])
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylim(0.85, 1.002)
        ax.set_xlabel("Search window")
        clean(ax)
    axes[0].set_ylabel("Tree tops found by both engines\n(share of the larger set)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    return save(fig, "detection_agreement")


# ---------------------------------------------------------------------------
def fig_grow_maps() -> Path:
    """Same tree tops, four growers, on three NAIP-CHM blocks and two lidar blocks."""
    picks = [("default", "naip", "eastern_hardwood_0"), ("default", "naip", "urban_atlanta_0"),
             ("default", "naip", "pinyon_juniper_0"), ("tallo", "lidar", "SCBI"), ("tallo", "lidar", "WREF")]  # fmt: skip
    size_m = 120.0
    fig, axes = plt.subplots(
        len(picks), 4, figsize=(12, 3.1 * len(picks)), constrained_layout=True
    )
    for i, (set_name, kind, name) in enumerate(picks):
        tag = f"{set_name}_{kind}_{name}"
        d = np.load(WORK / "labels" / f"{tag}.npz")
        import rasterio

        with rasterio.open(WORK / "surfaces" / f"{tag}.tif") as src:
            h = src.read(1)
            px = abs(src.transform.a)
        lab = {g: d[g] for g in GROWER_ORDER[:3]}
        with rasterio.open(WORK / "labels" / f"{tag}_lidr.tif") as src:
            lab["lidr"] = src.read(1)
        n = int(size_m / px)
        r0 = h.shape[0] // 2 - n // 2
        c0 = h.shape[1] // 2 - n // 2
        win = (slice(r0, r0 + n), slice(c0, c0 + n))
        ext = (0, size_m, 0, size_m)
        for j, g in enumerate(GROWER_ORDER):
            ax = axes[i, j]
            _crown_overlay(ax, h[win], lab[g][win], GROWER_COLORS[g], ext)
            if i == 0:
                ax.set_title(GROWER_LABELS[g], loc="left", fontsize=9)
            if j == 0:
                nice = (
                    strata.STRATUM_NAMES.get(name.rsplit("_", 1)[0], name)
                    if kind == "naip"
                    else f"NEON {name} lidar"
                )
                ax.set_ylabel(
                    f"{nice}\n{'default' if set_name == 'default' else 'Tallo'} parameters",
                    fontsize=9,
                )
    return save(fig, "growth_maps", dpi=90)


def fig_grow_metrics() -> Path:
    """Canopy assigned, crown size, boundary height and agreement, by grower."""
    t = pd.read_csv(L.TABLE_DIR / "grow.csv")
    metrics = [
        ("canopy_assigned", "Share of canopy in a crown"),
        ("median_cd", "Median crown diameter (m)"),
        ("valley", "Boundary height / shorter apex"),
        ("iou_vs_flood_apex", "IoU with the flood apex crown"),
    ]
    fig, axes = plt.subplots(
        2, len(metrics), figsize=(13, 6.2), constrained_layout=True
    )
    for row, (set_name, set_label) in enumerate(
        (("default", "Default parameters"), ("tallo", "Tallo parameters"))
    ):
        sub = t[t["set"] == set_name]
        for col, (m, label) in enumerate(metrics):
            ax = axes[row, col]
            for k, kind in enumerate(("naip", "lidar")):
                for gi, g in enumerate(GROWER_ORDER):
                    v = sub[(sub["kind"] == kind) & (sub["grower"] == g)][m].dropna()
                    if v.empty:
                        continue
                    x = k * 5 + gi
                    ax.scatter(np.full(len(v), x) + np.linspace(-0.25, 0.25, len(v)), v, s=8,
                               color=GROWER_COLORS[g], alpha=0.45, linewidth=0)  # fmt: skip
                    ax.plot(
                        [x - 0.35, x + 0.35],
                        [v.median()] * 2,
                        color=GROWER_COLORS[g],
                        linewidth=2.2,
                    )
            ax.set_xticks([1.5, 6.5], ["NAIP-CHM", "Lidar"])
            ax.set_title(label, loc="left", fontsize=9)
            clean(ax)
        axes[row, 0].set_ylabel(set_label, fontsize=10)
    handles = [
        Line2D([], [], color=GROWER_COLORS[g], linewidth=2.2, label=GROWER_LABELS[g])
        for g in GROWER_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    return save(fig, "growth_metrics")


# ---------------------------------------------------------------------------
def fig_speed() -> Path:
    t = pd.read_csv(L.TABLE_DIR / "speed.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), constrained_layout=True)
    size = t[t["radius_m"] == 12.0].groupby("size").median(numeric_only=True)
    mpx = size.index.to_numpy() ** 2 / 1e6
    ax = axes[0]
    series = [
        ("py_seeds", "Python tree tops", "#2a78d6", "--"),
        ("py_flood_apex", "Python flood growth", "#2a78d6", "-"),
        ("lidr_lmf", "lidR lmf", "#eb6834", "--"),
        ("lidr_grow", "lidR dalponte2016", "#eb6834", "-"),
    ]
    for col, label, color, ls in series:
        ax.plot(
            mpx,
            size[col],
            ls,
            marker="o",
            color=color,
            label=label,
            linewidth=1.8,
            ms=5,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Block size (million pixels)")
    ax.set_ylabel("Seconds, one thread (median of 3 to 5)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Run time against block size, 12 m cap", loc="left", fontsize=9)
    clean(ax)
    ax = axes[1]
    rad = t[t["size"] == 1024].groupby("radius_m").median(numeric_only=True)
    ax.plot(
        rad.index,
        rad["lidr_grow"],
        "-o",
        color="#eb6834",
        label="lidR dalponte2016",
        ms=5,
    )
    ax.plot(
        rad.index,
        rad["py_flood_apex"],
        "-o",
        color="#2a78d6",
        label="Python flood",
        ms=5,
    )
    ax.plot(
        rad.index,
        rad["py_post_trim"],
        "--o",
        color="#1baf7a",
        label="Python post-trim",
        ms=5,
    )
    ax.set_xlabel("Crown radius cap (m)")
    ax.set_ylabel("Seconds, 1024 x 1024 block")
    ax.set_title("Growth time against the crown cap", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    clean(ax)
    ax = axes[2]
    passes = L.TABLE_DIR / "dalponte_passes.csv"
    if passes.exists():
        p = pd.read_csv(passes)
        ax.plot(p["max_cr_px"] * 0.6, p["passes"], "-o", color="#eb6834", ms=5)
        ax.set_xlabel("max_cr (m at 0.6 m pixels)")
        ax.set_ylabel("Full-image passes of C_dalponte2016")
        ax.set_title(
            "Synchronous growth: passes against the cap", loc="left", fontsize=9
        )
        clean(ax)
    else:
        ax.set_visible(False)
    return save(fig, "speed")


# ---------------------------------------------------------------------------
def fig_before_after() -> list[Path]:
    """The strata canopy dot plot and two maps from before_after.py."""
    tag, engines = "before_after", ["legacy", "flood", "lidr"]
    table = pd.read_csv(L.TABLE_DIR / f"strata_{tag}.csv")
    out = [strata.canopy_dotplot(table, engines, DOCS_FIG / "before_after_canopy.png")]
    for s in ("midwest_mosaic", "urban_atlanta"):
        out.append(
            strata.stratum_figure(tag, s, engines, DOCS_FIG / f"before_after_{s}.png")
        )
    out.append(
        strata.cdf_grid(
            tag,
            engines,
            "crown_diam_m",
            "Crown diameter (m)",
            DOCS_FIG / "before_after_cdf_diameter.png",
        )
    )
    for path in out:
        shrink(path)
    return out


# ---------------------------------------------------------------------------
def fig_agb() -> Path:
    scores = pd.read_csv(L.TABLE_DIR / "agb_scores.csv")
    per = pd.read_parquet(L.TABLE_DIR / "agb_units.parquet")
    units = pd.read_parquet(L.TABLE_DIR / "neon_units.parquet")[
        ["unit_id", "split", "agbd_Mgha"]
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), constrained_layout=True)
    colors = {"python": "#2a78d6", "lidr": "#eb6834"}
    names = {"python": "Python flood", "lidr": "lidR dalponte2016"}
    for ax, source, title in zip(
        axes[:2], ("naip", "neon"), ("NAIP-CHM", "NEON lidar CHM"), strict=True
    ):
        for engine in ("python", "lidr"):
            d = per[
                (per.source == source) & (per.set == "tallo") & (per.engine == engine)
            ].merge(units, on="unit_id")
            d = d[d.split == "val"]
            ax.scatter(d["agbd_Mgha"], d["agbd"], s=12, color=colors[engine], alpha=0.6, linewidth=0,
                       label=names[engine])  # fmt: skip
        lim = float(np.nanpercentile(per[per.source == source]["agbd"], 99.5))
        lim = max(lim, float(units["agbd_Mgha"].max()))
        ax.plot([0, lim], [0, lim], color=INK_3, linewidth=0.8)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Field AGBD (Mg/ha)")
        ax.set_ylabel("Tallo crown AGBD, no fitting (Mg/ha)")
        ax.set_title(
            f"{title}, Tallo parameters, held-out units", loc="left", fontsize=9
        )
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        clean(ax)
    ax = axes[2]
    sub = scores[scores["set"] == "tallo"]
    y = 0
    labels = []
    for source in ("naip", "neon"):
        for engine in ("python", "lidr"):
            r = sub[(sub.source == source) & (sub.engine == engine)].iloc[0]
            ax.errorbar(r.val_mean_agbd, y, xerr=[[r.val_mean_agbd - r.val_mean_lo], [r.val_mean_hi - r.val_mean_agbd]],
                        fmt="o", color=colors[engine], capsize=3, ms=6)  # fmt: skip
            labels.append(f"{'NAIP' if source == 'naip' else 'Lidar'}, {names[engine]}")
            y += 1
    field = float(sub["field_val_mean"].iloc[0])
    ax.axvline(field, color=INK_3, linewidth=1, linestyle="--")
    ax.set_ylim(-0.6, y - 0.4)
    ax.text(
        field, -0.5, "field mean ", color=INK_2, fontsize=8, ha="right", va="bottom"
    )
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Mean held-out AGBD (Mg/ha), 95% coefficient interval")
    ax.set_title(
        "Engine difference against allometric uncertainty", loc="left", fontsize=9
    )
    clean(ax)
    return save(fig, "agb")


FIGURES = {
    "synthetic": fig_synthetic,
    "detection": fig_detection,
    "grow_maps": fig_grow_maps,
    "grow_metrics": fig_grow_metrics,
    "speed": fig_speed,
    "before_after": fig_before_after,
    "agb": fig_agb,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", choices=[*FIGURES, []], default=[])
    args = ap.parse_args()
    for name in args.names or FIGURES:
        FIGURES[name]()


if __name__ == "__main__":
    main()
