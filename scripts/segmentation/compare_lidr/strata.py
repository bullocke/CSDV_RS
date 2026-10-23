"""Run Python parameter sets and lidR on the strata blocks, then map and compare.

Used by ``before_after.py strata`` and ``compare.py``. For each block the
haloed CHM is segmented by every Python set and by lidR ``dalponte2016``
(through ``crown_segmentation.R``), crowns are kept when their centroid lies in
the block interior, and the crowns are saved per stratum and engine under
``results_root/segmentation_compare/crowns/<tag>/``.

Summary columns per stratum and engine: crown count, share of interior canopy
(raw CHM at or above 2 m) inside a crown, median and 90th percentile crown
diameter, Kolmogorov-Smirnov distance to lidR for apex height and diameter,
and compute seconds.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import _lib as L
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402
from shapely.geometry import box  # noqa: E402

from csdv_core.national.benchmark import interior_crowns  # noqa: E402
from csdv_core.segmentation.chm_watershed import segment_crowns  # noqa: E402
from csdv_core.segmentation.lidr_bridge import run_lidr_segmentation  # noqa: E402
from csdv_core.segmentation.params import SegmentationParams  # noqa: E402
from csdv_core.zonal.crowns import default_overlap_px  # noqa: E402

log = L.setup_logging("strata")

#: Light-mode categorical slots 1 to 3 of the reference palette. Legacy is also
#: dashed, because aqua sits below 3:1 against the surface.
COLORS = {"flood": "#2a78d6", "lidr": "#eb6834", "legacy": "#1baf7a"}
LABELS = {
    "flood": "Python, flood bound",
    "lidr": "lidR dalponte2016",
    "legacy": "Python, post-trim (earlier)",
}
SHORT = {"flood": "flood", "lidr": "lidR", "legacy": "post-trim"}
STYLES = {"flood": "-", "lidr": "-", "legacy": "--"}
SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
CHM_CMAP = ListedColormap(plt.get_cmap("Greys")(np.linspace(0.06, 0.88, 256)))
STRATUM_NAMES = {
    "eastern_hardwood": "Eastern hardwood (KY)",
    "eastern_hardwood_pair": "Eastern hardwood, neighbour (KY)",
    "southern_pine": "Southern pine (SC)",
    "northern_mixed": "Northern mixed (ME)",
    "pnw_conifer": "PNW conifer (WA)",
    "sierra_conifer": "Sierra conifer (CA)",
    "rocky_conifer": "Rocky Mountain conifer (MT)",
    "pinyon_juniper": "Pinyon-juniper (UT)",
    "oak_woodland": "Oak woodland (CA)",
    "midwest_mosaic": "Midwest mosaic (OH)",
    "plains_cropland": "Plains cropland (SD)",
    "urban_atlanta": "Urban, Atlanta (GA)",
}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": INK_3,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.titlecolor": INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def crown_dir(tag: str) -> Path:
    return L.OUT_DIR / "crowns" / tag


def segment_block(
    block: L.Block, params: SegmentationParams
) -> tuple[gpd.GeoDataFrame, float]:
    """All crowns of the haloed block, and the compute seconds."""
    t0 = time.perf_counter()
    crowns = segment_crowns(block.chm, block.transform, block.crs, params=params)
    return crowns, time.perf_counter() - t0


def lidr_block(
    block: L.Block, params: SegmentationParams, work: Path
) -> tuple[gpd.GeoDataFrame, float]:
    tif, gpkg = work / "block.tif", work / "block.gpkg"
    profile = {
        "driver": "GTiff", "width": block.chm.shape[1], "height": block.chm.shape[0],
        "count": 1, "dtype": "float32", "crs": block.crs, "transform": block.transform,
        "nodata": np.nan,
    }  # fmt: skip
    with rasterio.open(tif, "w", **profile) as dst:
        dst.write(block.chm.astype("float32"), 1)
    timings: dict[str, float] = {}
    try:
        run_lidr_segmentation(tif, gpkg, params=params, timings=timings)
    except RuntimeError:
        return gpd.GeoDataFrame(geometry=[], crs=block.crs), np.nan
    crowns = gpd.read_file(gpkg)
    seconds = sum(
        v
        for k, v in timings.items()
        if k in ("smooth", "mask", "seeds", "grow", "vectorize")
    )
    return crowns, seconds


def run(
    sets: dict[str, SegmentationParams],
    *,
    lidr_params: SegmentationParams | None,
    tag: str,
    strata: list[str] | None = None,
) -> pd.DataFrame:
    """Segment every block with every engine, save crowns, write the summary."""
    blocks = L.strata_blocks()
    if strata:
        blocks = blocks[blocks["stratum"].isin(strata)]
    out = crown_dir(tag)
    out.mkdir(parents=True, exist_ok=True)
    engines = list(sets) + (["lidr"] if lidr_params is not None else [])
    halo = max(
        default_overlap_px(p, 0.6)
        for p in list(sets.values()) + ([lidr_params] if lidr_params else [])
    )
    rows = []
    with tempfile.TemporaryDirectory(dir=L.OUT_DIR) as tmp:
        for stratum, grp in blocks.groupby("stratum", sort=False):
            kept = {e: [] for e in engines}
            secs = {e: 0.0 for e in engines}
            canopy = {e: [0, 0] for e in engines}
            for row in grp.itertuples(index=False):
                block = L.read_block(pd.Series(row._asdict()), halo)
                inner_canopy = np.zeros(block.chm.shape, dtype=bool)
                inner_canopy[block.inner] = np.nan_to_num(block.chm[block.inner]) >= 2.0
                for name in engines:
                    if name == "lidr":
                        crowns, s = lidr_block(block, lidr_params, Path(tmp))
                    else:
                        crowns, s = segment_block(block, sets[name])
                    # Canopy share counts every crown of the haloed block, so
                    # crowns centred just outside the interior still count.
                    lab = L.crown_label_raster(crowns, block.chm.shape, block.transform)
                    canopy[name][0] += int((lab[inner_canopy] > 0).sum())
                    canopy[name][1] += int(inner_canopy.sum())
                    kept[name].append(interior_crowns(crowns, block.interior))
                    secs[name] += s
            frames = {}
            for name in engines:
                parts = [k for k in kept[name] if len(k)]
                g = (
                    pd.concat(parts, ignore_index=True)
                    if parts
                    else gpd.GeoDataFrame(geometry=[], crs=block.crs)
                )
                if len(g):
                    g = gpd.GeoDataFrame(g, geometry="geometry", crs=block.crs)
                    g.to_parquet(out / f"{stratum}_{name}.parquet")
                frames[name] = g
            ref = frames.get("lidr")
            for name, g in frames.items():
                r = {
                    "stratum": stratum,
                    "engine": name,
                    "n_crowns": len(g),
                    "canopy_assigned": canopy[name][0] / max(canopy[name][1], 1),
                    "median_cd": g["crown_diam_m"].median() if len(g) else np.nan,
                    "p90_cd": g["crown_diam_m"].quantile(0.9) if len(g) else np.nan,
                    "median_h": g["apex_h_m"].median() if len(g) else np.nan,
                    "seconds": secs[name],
                }
                if ref is not None and len(ref) and len(g) and name != "lidr":
                    r["ks_h_vs_lidr"] = ks_2samp(
                        g["apex_h_m"], ref["apex_h_m"]
                    ).statistic
                    r["ks_cd_vs_lidr"] = ks_2samp(
                        g["crown_diam_m"], ref["crown_diam_m"]
                    ).statistic
                rows.append(r)
            log.info(
                "%s: %s",
                stratum,
                {
                    e: (
                        len(frames[e]),
                        round(canopy[e][0] / max(canopy[e][1], 1), 3),
                        round(secs[e], 1),
                    )
                    for e in engines
                },
            )
    table = pd.DataFrame(rows)
    L.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(L.TABLE_DIR / f"strata_{tag}.csv", index=False)
    return table


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def load_crowns(
    tag: str, stratum: str, engines: list[str]
) -> dict[str, gpd.GeoDataFrame]:
    out = {}
    for e in engines:
        path = crown_dir(tag) / f"{stratum}_{e}.parquet"
        if path.exists():
            out[e] = gpd.read_parquet(path)
    return out


def map_window(
    stratum: str, size_m: float
) -> tuple[Path, tuple[float, float, float, float]]:
    """A window centred in the stratum's median-canopy block."""
    blocks = L.strata_blocks()
    sub = blocks[blocks["stratum"] == stratum].sort_values("canopy_px")
    mid = sub.iloc[len(sub) // 2]
    path = L.tile_path(mid["chm_url"])
    side = int(mid["block_px"])
    with rasterio.open(path) as src:
        cx, cy = src.transform * (mid["col0"] + side / 2, mid["row0"] + side / 2)
    half = size_m / 2
    return path, (cx - half, cy - half, cx + half, cy + half)


def read_window(path: Path, bounds) -> np.ndarray:
    from rasterio.windows import from_bounds

    with rasterio.open(path) as src:
        data = src.read(1, window=from_bounds(*bounds, src.transform), masked=True)
    return data.astype("float32").filled(np.nan) * np.float32(L.NAIP_SCALE)


def draw_map(ax, chm, bounds, crowns: gpd.GeoDataFrame, engine: str, vmax: float):
    x0, y0, x1, y1 = bounds
    im = ax.imshow(np.nan_to_num(chm), cmap=CHM_CMAP, vmin=0, vmax=vmax,
                   extent=(0, x1 - x0, 0, y1 - y0), interpolation="nearest")  # fmt: skip
    g = crowns[crowns.intersects(box(*bounds))]
    if len(g):
        g = g.translate(-x0, -y0)
        g.boundary.plot(ax=ax, color=COLORS[engine], linewidth=0.8,
                        linestyle=STYLES[engine])  # fmt: skip
    ax.set_xlim(0, x1 - x0)
    ax.set_ylim(0, y1 - y0)
    ax.set_aspect("equal")
    ax.set_xlabel("m")
    ax.set_ylabel("m")
    ax.tick_params(labelsize=8)
    return im


def ecdf(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.sort(np.asarray(v))
    return v, np.arange(1, len(v) + 1) / len(v)


def cdf_axes(ax, xlabel: str) -> None:
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.spines[["top", "right"]].set_visible(False)


def stratum_figure(
    tag: str, stratum: str, engines: list[str], out: Path, size_m: float = 150.0
) -> Path:
    crowns = load_crowns(tag, stratum, engines)
    path, bounds = map_window(stratum, size_m)
    chm = read_window(path, bounds)
    vmax = float(np.nanpercentile(chm, 99.5)) if np.isfinite(chm).any() else 1.0
    fig = plt.figure(figsize=(4.2 * len(engines), 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, len(engines), height_ratios=[1.15, 1])
    map_axes = []
    for i, e in enumerate(engines):
        ax = fig.add_subplot(gs[0, i])
        im = draw_map(
            ax, chm, bounds, crowns.get(e, gpd.GeoDataFrame(geometry=[])), e, vmax
        )
        n_win = int(crowns[e].intersects(box(*bounds)).sum()) if e in crowns else 0
        ax.set_title(f"{LABELS[e]}: {n_win} crowns", loc="left", fontsize=10)
        map_axes.append(ax)
    fig.colorbar(
        im, ax=map_axes, shrink=0.8, label="NAIP-CHM height (m)", location="right"
    )
    half = len(engines) // 2 or 1
    ax_cd = fig.add_subplot(gs[1, :half])
    ax_h = fig.add_subplot(gs[1, half:])
    for e in engines:
        g = crowns.get(e)
        if g is None or g.empty:
            continue
        for ax, col in ((ax_cd, "crown_diam_m"), (ax_h, "apex_h_m")):
            x, y = ecdf(g[col])
            ax.plot(x, y, color=COLORS[e], linestyle=STYLES[e], linewidth=1.8,
                    label=f"{SHORT[e]}: n = {len(g):,}, median {np.median(g[col]):.1f} m")  # fmt: skip
    cdf_axes(ax_cd, "Crown diameter, area-equivalent (m)")
    cdf_axes(ax_h, "Apex height (m)")
    ax_cd.set_ylabel("Cumulative fraction of crowns")
    ax_cd.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.0))
    ax_h.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.0))
    fig.suptitle(f"{STRATUM_NAMES.get(stratum, stratum)}: {size_m:.0f} m window and CDFs over three 1024 px blocks",
                 x=0.01, ha="left", fontsize=11)  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def canopy_dotplot(table: pd.DataFrame, engines: list[str], out: Path) -> Path:
    """Share of canopy inside a crown, by stratum and engine."""
    t = table[table["engine"].isin(engines)].pivot(
        index="stratum", columns="engine", values="canopy_assigned"
    )
    t = t.dropna(how="all").sort_values(engines[0])
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(t) + 1.2), constrained_layout=True)
    y = np.arange(len(t))
    for e in engines:
        marker = "o" if e != "legacy" else "D"
        ax.scatter(t[e], y, s=42, color=COLORS[e], marker=marker, label=LABELS[e],
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)  # fmt: skip
    ax.set_yticks(y, [STRATUM_NAMES.get(s, s) for s in t.index])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of canopy (NAIP-CHM ≥ 2 m) inside a crown")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=len(engines),
    )
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def cdf_grid(tag: str, engines: list[str], column: str, xlabel: str, out: Path) -> Path:
    strata = [s for s in L.strata_blocks()["stratum"].unique() if s != "desert"]
    ncol = 4
    nrow = int(np.ceil(len(strata) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(12, 2.6 * nrow), constrained_layout=True, sharey=True
    )
    for ax, s in zip(axes.ravel(), strata, strict=False):
        crowns = load_crowns(tag, s, engines)
        for e in engines:
            if e in crowns and len(crowns[e]):
                x, y = ecdf(crowns[e][column])
                ax.plot(x, y, color=COLORS[e], linestyle=STYLES[e], linewidth=1.5)
        ax.set_title(STRATUM_NAMES.get(s, s), loc="left", fontsize=9)
        cdf_axes(ax, xlabel if ax in axes[-1] else "")
    for ax in axes.ravel()[len(strata) :]:
        ax.set_visible(False)
    handles = [
        Line2D(
            [], [], color=COLORS[e], linestyle=STYLES[e], linewidth=1.8, label=LABELS[e]
        )
        for e in engines
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(engines),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.supylabel("Cumulative fraction of crowns", color=INK_2, fontsize=9)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out
