"""Python against lidR: tree detection, crown growth, speed and AGB.

``detect``     ``locate_seeds`` against ``locate_trees(lmf)`` on identical
               smoothed, masked rasters (36 NAIP-CHM blocks, 16 NEON lidar
               blocks) for fixed 3, 5 and 8 m windows and the Tallo window.
               Counts the tree tops each finds, how many coincide, and why the
               rest differ. Repeats lidR at 6 threads to test determinism.
``grow``       The same Python tree tops grown by lidR ``dalponte2016`` and by
               the Python flood (apex and mean rules) and post-trim growers,
               under the default and Tallo parameter sets.
``speed``      Wall time and memory of each step against block size and
               crown radius cap, both engines on one thread, lidR also on six.
``synthetic``  Four constructed scenes that isolate each rule difference.
``agb``        Tallo crown AGB from both engines on the NEON units, NAIP-CHM
               and lidar, against field AGBD, with coefficient draws.

Run any step with ``$P scripts/segmentation/compare_lidr/compare.py <step>``.
Each writes tables to ``results_root/segmentation_compare/tables/`` and
figures to ``figures/compare/``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import _lib as L
import numpy as np
import pandas as pd
import surfaces as S

from csdv_core.segmentation.params import (
    DEFAULT_PARAMS,
    WINDOW_FUNCTIONS,
    SegmentationParams,
    WindowFunction,
    tallo_params,
    tallo_window,
)

log = L.setup_logging("compare")

WORK = L.OUT_DIR / "work"
BENCH_R = Path(__file__).with_name("bench_lidr.R")
FIXED = {
    "fixed_3m": WindowFunction(a=3.0, lo=3.0, hi=3.0, name="fixed_3m"),
    "fixed_5m": WINDOW_FUNCTIONS["fixed_5m"],
    "fixed_8m": WINDOW_FUNCTIONS["fixed_8m"],
}
MANIFEST_COLUMNS = [
    "job_id", "chm", "min_height", "ws_form", "ws_a", "ws_b", "ws_c", "ws_k",
    "ws_lo", "ws_hi", "th_seed", "th_cr", "max_cr_px", "seeds", "threads",
    "reps", "grow", "out_tops", "out_labels", "profile",
]  # fmt: skip


def window_for(name: str, biome: str) -> WindowFunction:
    return tallo_window(biome) if name == "tallo" else FIXED[name]


def lidr_job(
    job_id: str, chm: Path, params: SegmentationParams, px: float, **kw
) -> dict:
    """One manifest row; the window and thresholds come from ``params``."""
    w = params.window
    row = {
        "job_id": job_id,
        "chm": str(chm),
        "min_height": params.min_height_m,
        "ws_form": w.form, "ws_a": w.a, "ws_b": w.b, "ws_c": w.c, "ws_k": w.k,
        "ws_lo": w.lo, "ws_hi": w.hi,
        "th_seed": params.th_seed or 0.45,
        "th_cr": params.th_cr,
        # Same conversion as crown_segmentation.R.
        "max_cr_px": max(1, round((params.max_crown_radius_m or 1000.0) / px)),
        "seeds": "", "threads": 1, "reps": 1, "grow": 0,
        "out_tops": "", "out_labels": "", "profile": 0,
    }  # fmt: skip
    row.update(kw)
    return row


#: Set by ``--reuse``: read an earlier lidR run of the same step if present.
REUSE = False
#: Set by ``--limit``: keep only the first few surfaces, for a quick test.
LIMIT: int | None = None


def run_lidr(jobs: list[dict], name: str) -> pd.DataFrame:
    """Run a manifest through bench_lidr.R in one R session."""
    WORK.mkdir(parents=True, exist_ok=True)
    manifest = WORK / f"{name}_manifest.csv"
    out = WORK / f"{name}_lidr.csv"
    if REUSE and out.exists():
        return pd.read_csv(out)
    pd.DataFrame(jobs, columns=MANIFEST_COLUMNS).to_csv(manifest, index=False)
    rscript = shutil.which("Rscript")
    t0 = time.perf_counter()
    proc = subprocess.run(
        [rscript, str(BENCH_R), f"--manifest={manifest}", f"--out={out}"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"bench_lidr.R failed:\n{proc.stderr[-3000:]}")
    log.info("lidR %s: %d jobs in %.0f s", name, len(jobs), time.perf_counter() - t0)
    return pd.read_csv(out)


def warm_up() -> None:
    """Compile the Numba kernels once, so no timing includes compilation."""
    from csdv_core.segmentation.chm_watershed import locate_seeds
    from csdv_core.segmentation.flood import grow_crowns

    h = np.add.outer(np.arange(20.0), np.arange(20.0))
    rows, cols = locate_seeds(h, np.ones(h.shape, bool), 1.0, FIXED["fixed_3m"])
    grow_crowns(h, rows, cols, h[rows, cols], th_cr=0.5)
    grow_crowns(h, rows, cols, h[rows, cols], th_cr=0.5, rule="mean")


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def _is_max(h: np.ndarray, r: int, c: int, radius_px: float, quantized: bool) -> bool:
    """Whether (r, c) has no higher pixel in its window under one engine's rule."""
    rad = int(np.ceil(radius_px)) + 1
    r0, r1 = max(0, r - rad), min(h.shape[0], r + rad + 1)
    c0, c1 = max(0, c - rad), min(h.shape[1], c + rad + 1)
    yy, xx = np.ogrid[r0 - r : r1 - r, c0 - c : c1 - c]
    d2 = (yy * yy + xx * xx).astype("float64")
    lim = radius_px * radius_px + (1e-9 if quantized else 1e-6)
    win = np.where(np.isnan(h[r0:r1, c0:c1]), -np.inf, h[r0:r1, c0:c1])
    return not np.any(win[d2 <= lim] > h[r, c])


def classify(
    s: S.Surface, tops: np.ndarray, other: set, window: WindowFunction, params
) -> pd.Series:
    """Why each top found by one engine is missing from the other."""
    h = s.smoothed
    floor_px = max(1, int(round(params.min_separation_m / s.px)))
    other_arr = np.array(sorted(other)) if other else np.zeros((0, 2), dtype=int)
    out = []
    for r, c in tops:
        if (r, c) in other:
            out.append("match")
            continue
        if len(other_arr) and np.min(np.abs(other_arr - (r, c)).max(axis=1)) <= 1:
            near = True
        else:
            near = False
        plateau = S.plateau_of(h, r, c)
        if len(plateau) > 1 and any((int(a), int(b)) in other for a, b in plateau):
            out.append("plateau")
            continue
        exact = float(window(h[r, c])) / 2.0 / s.px
        if params.window_metric == "pixel":
            quant = max(float(np.rint(exact)), floor_px)
        else:
            quant = max(exact, 1.0, params.min_separation_m / s.px)
        if _is_max(h, r, c, exact, False) != _is_max(h, r, c, quant, True):
            out.append("radius")
        elif len(plateau) > 1:
            out.append("plateau")
        else:
            out.append("near" if near else "other")
    return pd.Series(out).value_counts()


def cmd_detect() -> None:
    params = DEFAULT_PARAMS
    surfaces = S.naip_surfaces(lambda b: params) + S.lidar_surfaces(lambda b: params)
    surfaces = surfaces[:LIMIT] if LIMIT else surfaces
    warm_up()
    jobs, py = [], {}
    for s in surfaces:
        S.write_surface(
            s, WORK / "surfaces" / f"{s.kind}_{s.name.replace('#', '_')}.tif"
        )
        for wname in (*FIXED, "tallo"):
            p = params.replace(window=window_for(wname, s.biome))
            for metric in ("exact", "pixel"):
                pm = p.replace(window_metric=metric)
                t0 = time.perf_counter()
                rows, cols = S.python_seeds(s, pm)
                py[(s.name, wname, metric)] = (rows, cols, time.perf_counter() - t0)
            tops = WORK / "tops" / f"{s.kind}_{s.name.replace('#', '_')}_{wname}.csv"
            tops.parent.mkdir(parents=True, exist_ok=True)
            jobs.append(
                lidr_job(f"{s.name}|{wname}|1", s.path, p, s.px, out_tops=str(tops))
            )
            if wname == "fixed_5m":
                t6 = tops.with_name(tops.stem + "_t6.csv")
                jobs.append(
                    lidr_job(
                        f"{s.name}|{wname}|6",
                        s.path,
                        p,
                        s.px,
                        threads=6,
                        out_tops=str(t6),
                    )
                )
    lidr = run_lidr(jobs, "detect").set_index("job_id")

    rows_out = []
    by_name = {s.name: s for s in surfaces}
    for (name, wname, metric), (r, c, t_py) in py.items():
        s = by_name[name]
        p = params.replace(window=window_for(wname, s.biome), window_metric=metric)
        tops_path = WORK / "tops" / f"{s.kind}_{name.replace('#', '_')}_{wname}.csv"
        lt = pd.read_csv(tops_path)
        lr, lc = S.xy_to_rc(s, lt["x"].to_numpy(), lt["y"].to_numpy())
        inner = s.interior
        py_set = {(int(a), int(b)) for a, b in zip(r, c, strict=True) if inner[a, b]}
        li_set = {(int(a), int(b)) for a, b in zip(lr, lc, strict=True) if inner[a, b]}
        py_only = classify(s, np.array(sorted(py_set - li_set)), li_set, p.window, p)
        li_only = classify(s, np.array(sorted(li_set - py_set)), py_set, p.window, p)
        job = lidr.loc[f"{name}|{wname}|1"]
        row = {
            "surface": name, "kind": s.kind, "biome": s.biome, "window": wname,
            "metric": metric,
            "n_python": len(py_set), "n_lidr": len(li_set), "n_match": len(py_set & li_set),
            "t_python": t_py, "t_lidr": job["t_lmf"], "n_valid": job["n_valid"],
        }  # fmt: skip
        for k, v in py_only.items():
            row[f"py_only_{k}"] = int(v)
        for k, v in li_only.items():
            row[f"lidr_only_{k}"] = int(v)
        if wname == "fixed_5m":
            t6 = pd.read_csv(tops_path.with_name(tops_path.stem + "_t6.csv"))
            r6, c6 = S.xy_to_rc(s, t6["x"].to_numpy(), t6["y"].to_numpy())
            s6 = {(int(a), int(b)) for a, b in zip(r6, c6, strict=True) if inner[a, b]}
            row["lidr_t6_n"] = len(s6)
            row["lidr_t6_differs"] = len(s6 ^ li_set)
            row["t_lidr_t6"] = lidr.loc[f"{name}|{wname}|6", "t_lmf"]
        rows_out.append(row)
    table = pd.DataFrame(rows_out).fillna(0)
    table.to_csv(L.TABLE_DIR / "detect.csv", index=False)
    summary = table.groupby(["metric", "kind", "window"]).sum(numeric_only=True)
    summary["match_share_py"] = summary["n_match"] / summary["n_python"]
    summary["match_share_lidr"] = summary["n_match"] / summary["n_lidr"]
    summary["speedup"] = summary["t_lidr"] / summary["t_python"]
    log.info("\n%s", summary.round(3).T.to_string())


# ---------------------------------------------------------------------------
# Growth from identical seeds
# ---------------------------------------------------------------------------
GROWERS = ("flood_apex", "flood_mean", "post_trim", "lidr")
#: Surfaces whose label maps are kept for the figures.
MAP_SURFACES = (
    "eastern_hardwood#0",
    "urban_atlanta#0",
    "pinyon_juniper#0",
    "SCBI",
    "WREF",
    "SRER",
)


def param_sets() -> dict[str, object]:
    """Name -> function of biome giving the parameter set."""
    return {
        "default": lambda biome: DEFAULT_PARAMS,
        "tallo": lambda biome: tallo_params(biome),
    }


def grow_metrics(s, labels, rows, cols, ref=None) -> dict:
    table = S.crown_table(s, labels, rows, cols)
    apex_by_label = np.concatenate([[np.inf], s.smoothed[rows, cols]])
    out = {
        "n_crowns": len(table),
        "canopy_assigned": S.canopy_assigned(s, labels),
        "median_cd": table["crown_diam_m"].median(),
        "p90_cd": table["crown_diam_m"].quantile(0.9),
        "valley": S.valley_score(s, labels, apex_by_label),
        "straight": S.straight_edge_share(s, labels),
        "compact": S.compactness(labels, table),
    }
    if ref is not None:
        out["iou_vs_flood_apex"] = S.matched_iou(labels, ref, table["label"].to_numpy())
    return out


def cmd_grow() -> None:
    warm_up()
    sets = param_sets()
    rows_out, jobs, pending = [], [], []
    keep = WORK / "labels"
    keep.mkdir(parents=True, exist_ok=True)
    for set_name, params_for in sets.items():
        surfaces = S.naip_surfaces(params_for) + S.lidar_surfaces(params_for)
        surfaces = surfaces[:LIMIT] if LIMIT else surfaces
        for s in surfaces:
            p = params_for(s.biome)
            tag = f"{set_name}_{s.kind}_{s.name.replace('#', '_')}"
            S.write_surface(s, WORK / "surfaces" / f"{tag}.tif")
            rows, cols = S.python_seeds(s, p)
            seeds_csv = WORK / "seeds" / f"{tag}.csv"
            seeds_csv.parent.mkdir(parents=True, exist_ok=True)
            S.seeds_to_xy(s, rows, cols).to_csv(seeds_csv, index=False)
            labels = {}
            for g in GROWERS[:3]:
                t0 = time.perf_counter()
                labels[g] = S.python_labels(s, rows, cols, p, g)
                rows_out.append(
                    {"set": set_name, "surface": s.name, "kind": s.kind, "biome": s.biome,
                     "grower": g, "seconds": time.perf_counter() - t0,
                     **grow_metrics(s, labels[g], rows, cols, labels["flood_apex"])}
                )  # fmt: skip
            if s.name in MAP_SURFACES:
                np.savez_compressed(keep / f"{tag}.npz", rows=rows, cols=cols, **labels)
            out_labels = keep / f"{tag}_lidr.tif"
            jobs.append(
                lidr_job(f"{tag}", s.path, p, s.px, seeds=str(seeds_csv), grow=1,
                         out_labels=str(out_labels))  # fmt: skip
            )
            pending.append(
                (
                    set_name,
                    s,
                    rows,
                    cols,
                    labels["flood_apex"],
                    labels["flood_mean"],
                    out_labels,
                )
            )
            log.info("%s: %d seeds", tag, rows.size)
    lidr = run_lidr(jobs, "grow").set_index("job_id")
    for set_name, s, rows, cols, ref, mean_rule, out_labels in pending:
        tag = f"{set_name}_{s.kind}_{s.name.replace('#', '_')}"
        lab = S.read_labels(out_labels)
        m = grow_metrics(s, lab, rows, cols, ref)
        table = S.crown_table(s, lab, rows, cols)
        m["iou_vs_flood_mean"] = S.matched_iou(
            lab, mean_rule, table["label"].to_numpy()
        )
        rows_out.append(
            {"set": set_name, "surface": s.name, "kind": s.kind, "biome": s.biome,
             "grower": "lidr", "seconds": lidr.loc[tag, "t_grow"], **m}
        )  # fmt: skip
        if s.name not in MAP_SURFACES:
            out_labels.unlink(missing_ok=True)
    table = pd.DataFrame(rows_out)
    table.to_csv(L.TABLE_DIR / "grow.csv", index=False)
    cols = ["canopy_assigned", "median_cd", "p90_cd", "valley", "straight", "compact",
            "iou_vs_flood_apex", "iou_vs_flood_mean", "seconds"]  # fmt: skip
    summary = table.groupby(["set", "kind", "grower"])[cols].median()
    log.info("\n%s", summary.round(3).to_string())


# ---------------------------------------------------------------------------
# Speed and memory
# ---------------------------------------------------------------------------
SPEED_STRATA = ("eastern_hardwood", "pnw_conifer", "pinyon_juniper")
SPEED_SIZES = (256, 512, 1024, 2048)
SPEED_RADII_M = (5.0, 10.0, 15.0, 25.0)


def speed_surface(stratum: str, size: int, params: SegmentationParams) -> S.Surface:
    """A size x size crop centred in the stratum's first tile, prepared."""
    import rasterio
    from rasterio.windows import Window
    from rasterio.windows import transform as win_transform

    row = L.strata_blocks().query("stratum == @stratum").iloc[0]
    with rasterio.open(L.tile_path(row["chm_url"])) as src:
        r0 = (src.height - size) // 2
        c0 = (src.width - size) // 2
        win = Window(c0, r0, size, size)
        data = src.read(1, window=win, masked=True)
        chm = data.astype("float32").filled(np.nan) * np.float32(L.NAIP_SCALE)
        transform, crs = win_transform(win, src.transform), src.crs
    full = (slice(0, size), slice(0, size))
    return S.Surface(f"{stratum}_{size}", "naip", row["biome"], S.prepare(chm, 0.6, params),
                     transform, crs, full)  # fmt: skip


def vmhwm_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return float(line.split()[1]) / 1024.0
    return float("nan")


def speed_worker(path: str, max_radius_m: float, reps: int) -> dict:
    """Time the Python steps on one prepared surface. Runs in its own process.

    Memory is the process high-water mark (``VmHWM``), which a new process
    starts afresh. ``ru_maxrss`` is not used: Linux carries it across fork and
    exec, so a worker would report its parent's peak.
    """
    import rasterio
    from rasterio.features import shapes

    from csdv_core.segmentation.chm_watershed import locate_seeds

    base_rss = vmhwm_mb()
    warm_up()
    jit_rss = vmhwm_mb()
    with rasterio.open(path) as src:
        h = src.read(1)
        transform = src.transform
    s = S.Surface("x", "naip", "", h, transform, None, (slice(None), slice(None)))
    p = DEFAULT_PARAMS.replace(max_crown_radius_m=max_radius_m)
    valid = ~np.isnan(h)
    t = {"seeds": [], "flood_apex": [], "post_trim": [], "vectorize": []}
    for _ in range(reps):
        t0 = time.perf_counter()
        rows, cols = locate_seeds(
            h, valid, s.px, p.window, window_metric=p.window_metric
        )
        t["seeds"].append(time.perf_counter() - t0)
        for g in ("flood_apex", "post_trim"):
            t0 = time.perf_counter()
            lab = S.python_labels(s, rows, cols, p, g)
            t[g].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        _ = sum(
            1 for _ in shapes(lab, mask=lab > 0, transform=transform, connectivity=4)
        )
        t["vectorize"].append(time.perf_counter() - t0)
    out = {k: float(np.median(v)) for k, v in t.items()}
    out["n_tops"] = int(rows.size)
    out["peak_rss_mb"] = vmhwm_mb()
    out["base_rss_mb"] = base_rss
    out["jit_rss_mb"] = jit_rss
    return out


def cmd_speed() -> None:
    import json
    import sys

    params = DEFAULT_PARAMS
    configs = []  # (stratum, size, radius)
    for st in SPEED_STRATA:
        configs += [(st, n, 12.0) for n in SPEED_SIZES]
        configs += [(st, 1024, r) for r in SPEED_RADII_M]
    rows_out = []
    for st in SPEED_STRATA:
        jobs = []
        for size in SPEED_SIZES:
            s = speed_surface(st, size, params)
            path = S.write_surface(s, WORK / "speed" / f"{st}_{size}.tif")
            reps = 3 if size >= 1024 else 5
            for radius in sorted({r for (a, n, r) in configs if a == st and n == size}):
                p = params.replace(max_crown_radius_m=radius)
                job_id = f"{st}|{size}|{radius:g}"
                jobs.append(lidr_job(job_id, path, p, 0.6, grow=1, reps=reps,
                                     profile=int(size == 1024)))  # fmt: skip
                if size == 1024 and radius == 12.0:
                    jobs.append(
                        lidr_job(job_id + "|t6", path, p, 0.6, threads=6, reps=reps)
                    )
                proc = subprocess.run(
                    [sys.executable, __file__, "speed-worker", str(path), str(radius), str(reps)],
                    capture_output=True, text=True,
                )  # fmt: skip
                if proc.returncode != 0:
                    raise RuntimeError(proc.stderr[-2000:])
                py = json.loads(proc.stdout.strip().splitlines()[-1])
                rows_out.append({"stratum": st, "size": size, "radius_m": radius,
                                 "n_valid": int(np.isfinite(s.smoothed).sum()), **{f"py_{k}": v for k, v in py.items()}})  # fmt: skip
                log.info(
                    "python %s: %s", job_id, {k: round(v, 3) for k, v in py.items()}
                )
        lidr = run_lidr(jobs, f"speed_{st}").set_index("job_id")
        for r in rows_out:
            if r["stratum"] != st:
                continue
            j = lidr.loc[f"{st}|{r['size']}|{r['radius_m']:g}"]
            r.update({"lidr_lmf": j["t_lmf"], "lidr_grow": j["t_grow"], "lidr_cpp": j["t_cpp"],
                      "lidr_tops": j["n_tops"], "lidr_vmhwm_mb": j["vmhwm_mb"]})  # fmt: skip
            t6 = f"{st}|{r['size']}|{r['radius_m']:g}|t6"
            if t6 in lidr.index:
                r["lidr_lmf_t6"] = lidr.loc[t6, "t_lmf"]
    table = pd.DataFrame(rows_out)
    table.to_csv(L.TABLE_DIR / "speed.csv", index=False)
    show = ["stratum", "size", "radius_m", "py_seeds", "py_flood_apex", "py_post_trim",
            "lidr_lmf", "lidr_lmf_t6", "lidr_grow", "lidr_cpp", "py_peak_rss_mb", "lidr_vmhwm_mb"]  # fmt: skip
    log.info(
        "\n%s", table[[c for c in show if c in table]].round(3).to_string(index=False)
    )


# ---------------------------------------------------------------------------
# Synthetic scenes
# ---------------------------------------------------------------------------
def _dome(shape, centre, height, sigma_px):
    rr, cc = np.indices(shape, dtype="float64")
    d2 = (rr - centre[0]) ** 2 + (cc - centre[1]) ** 2
    return height * np.exp(-d2 / (2.0 * sigma_px**2))


def synthetic_scenes() -> dict[str, np.ndarray]:
    """Constructed CHMs (0.6 m pixels, 60 m across) that isolate one rule each.

    Designed for ``th_cr = 0.55`` and ``th_seed = 0.45``, refAGB's values.
    """
    shape = (100, 100)
    rr, cc = np.indices(shape, dtype="float64")
    hedge = np.where((np.abs(rr - 50) <= 3) & (cc >= 28) & (cc <= 95), 4.0, 0.0)
    plateau = np.clip(
        20.0 * (1.0 - np.maximum(np.hypot(rr - 50, cc - 50) - 22, 0) / 6.0), 0, 20
    )
    plateau += 0.4 * _dome(shape, (50, 50), 1.0, 12)
    scenes = {
        # The crossing of the two domes (about 10.6 m) lies below the tall
        # tree's floor (13.8 m) and above the short tree's (8.8 m).
        "Unequal neighbours": np.maximum(
            _dome(shape, (50, 40), 25.0, 9), _dome(shape, (50, 60), 16.0, 9)
        ),
        "Isolated dome": _dome(shape, (50, 50), 20.0, 11),
        # The 4 m hedge is below the small tree's apex floor (4.4 m) and above
        # its th_seed floor (3.6 m), so only a mean-relative rule enters it.
        "Small tree beside a hedge": np.maximum(_dome(shape, (50, 24), 8.0, 4), hedge),
        "Wide and narrow crown": np.maximum(
            _dome(shape, (50, 35), 20.0, 18), _dome(shape, (50, 60), 20.0, 6)
        ),
        "Crown wider than the cap": plateau,
    }
    return {
        k: np.where(v >= 2.0, v, np.nan).astype("float32") for k, v in scenes.items()
    }


def cmd_synthetic() -> None:
    from rasterio.transform import from_origin

    warm_up()
    params = DEFAULT_PARAMS.replace(smooth_radius_m=0.0, th_cr=0.55, th_seed=0.45)
    transform = from_origin(500_000.0, 4_000_000.0, 0.6, 0.6)
    jobs, out = [], {}
    for i, (name, h) in enumerate(synthetic_scenes().items()):
        s = S.Surface(
            name, "naip", "", h, transform, "EPSG:32617", (slice(None), slice(None))
        )
        S.write_surface(s, WORK / "synthetic" / f"scene{i}.tif")
        rows, cols = S.python_seeds(s, params)
        seeds = WORK / "synthetic" / f"scene{i}_seeds.csv"
        S.seeds_to_xy(s, rows, cols).to_csv(seeds, index=False)
        labels = {g: S.python_labels(s, rows, cols, params, g) for g in GROWERS[:3]}
        lab_path = WORK / "synthetic" / f"scene{i}_lidr.tif"
        jobs.append(lidr_job(f"scene{i}", s.path, params, 0.6, seeds=str(seeds), grow=1,
                             out_labels=str(lab_path)))  # fmt: skip
        out[name] = (s, rows, cols, labels, lab_path)
    run_lidr(jobs, "synthetic")
    rows_out = {}
    for name, (s, rows, cols, labels, lab_path) in out.items():
        labels["lidr"] = S.read_labels(lab_path)
        np.savez_compressed(WORK / "synthetic" / f"{name.replace(' ', '_')}.npz",
                            h=s.smoothed, rows=rows, cols=cols, **labels)  # fmt: skip
        rows_out[name] = {g: int((labels[g] > 0).sum()) for g in GROWERS}
    log.info(
        "\nAssigned pixels by scene and grower:\n%s",
        pd.DataFrame(rows_out).T.to_string(),
    )


# ---------------------------------------------------------------------------
# AGB on the NEON units
# ---------------------------------------------------------------------------
AGB_ENGINES = ("python", "lidr")


def _unit_crowns(
    s: S.Surface, labels: np.ndarray, rows, cols, region: np.ndarray, min_area: float
) -> pd.DataFrame:
    """Crowns whose tree top lies in the unit, with apex height and diameter."""
    n = rows.size
    area = np.bincount(labels.ravel(), minlength=n + 1)[1 : n + 1] * s.px * s.px
    keep = region[rows, cols] & (area >= min_area)
    return pd.DataFrame(
        {
            "apex_h_m": s.smoothed[rows, cols][keep],
            "crown_diam_m": 2.0 * np.sqrt(area[keep] / np.pi),
        }
    )


def cmd_agb() -> None:
    import geopandas as gpd
    import rasterio

    from csdv_core.biomass.allometry import tallo_model

    warm_up()
    table = pd.read_parquet(L.TABLE_DIR / "neon_units.parquet")
    units = L.neon_units()[["unit_id", "geometry"]].merge(table, on="unit_id")
    units = units[
        units["biome_long"].isin(
            [b for b in units["biome_long"].unique() if _has_tallo(b)]
        )
    ]
    sets = param_sets()
    crowns: dict[tuple, pd.DataFrame] = {}
    jobs, pending = [], []
    work = WORK / "agb"
    for source in ("naip", "neon"):
        for unit in units.itertuples():
            chip = L.unit_chip(source, unit.unit_id)
            if not chip.exists():
                continue
            chm, transform, crs = L.read_chip(chip)
            px = float(abs(transform.a))
            poly = gpd.GeoSeries([unit.geometry], crs=units.crs).to_crs(crs).iloc[0]
            region = L.unit_mask(poly, chm.shape, transform)
            for set_name, params_for in sets.items():
                p = params_for(unit.biome_long)
                s = S.Surface(unit.unit_id, source, unit.biome_long, S.prepare(chm, px, p), transform, crs,
                              (slice(None), slice(None)))  # fmt: skip
                tag = f"{source}_{set_name}_{unit.unit_id.replace(':', '_')}"
                if np.isfinite(s.smoothed).any():
                    rows, cols = S.python_seeds(s, p)
                    lab = S.python_labels(s, rows, cols, p, "flood_apex")
                    crowns[(source, set_name, "python", unit.unit_id)] = _unit_crowns(
                        s, lab, rows, cols, region, p.min_crown_area_m2
                    )
                    S.write_surface(s, work / "surfaces" / f"{tag}.tif")
                    tops = work / "tops" / f"{tag}.csv"
                    labels = work / "labels" / f"{tag}.tif"
                    tops.parent.mkdir(parents=True, exist_ok=True)
                    labels.parent.mkdir(parents=True, exist_ok=True)
                    jobs.append(
                        lidr_job(
                            tag,
                            s.path,
                            p,
                            px,
                            grow=1,
                            out_tops=str(tops),
                            out_labels=str(labels),
                        )
                    )
                    pending.append(
                        (source, set_name, unit.unit_id, s, region, p, tops, labels)
                    )
                else:
                    for e in AGB_ENGINES:
                        crowns[(source, set_name, e, unit.unit_id)] = pd.DataFrame(
                            columns=["apex_h_m", "crown_diam_m"]
                        )
        log.info("%s: Python done", source)
    run_lidr(jobs, "agb")
    for source, set_name, uid, s, region, p, tops, labels in pending:
        if not tops.exists() or not labels.exists():
            crowns[(source, set_name, "lidr", uid)] = pd.DataFrame(
                columns=["apex_h_m", "crown_diam_m"]
            )
            continue
        t = pd.read_csv(tops)
        with rasterio.open(labels) as src:
            lab = src.read(1).astype("int64")
        r, c = S.xy_to_rc(s, t["x"].to_numpy(), t["y"].to_numpy())
        # Reorder so label k sits at index k - 1, as the Python labels do.
        order = np.argsort(t["treeID"].to_numpy())
        r, c = r[order], c[order]
        ids = np.sort(t["treeID"].to_numpy())
        lab = np.where(lab > 0, np.searchsorted(ids, lab) + 1, 0)
        crowns[(source, set_name, "lidr", uid)] = _unit_crowns(
            s, lab, r, c, region, p.min_crown_area_m2
        )

    # Unit AGBD, point estimate and coefficient draws.
    rng_draws = {
        b: tallo_model(b).draws(1000, seed=7) for b in units["biome_long"].unique()
    }
    area_ha = units.set_index("unit_id")["area_m2"] / 1e4
    biome = units.set_index("unit_id")["biome_long"]
    rows_out, draws_out = [], []
    for (source, set_name, engine, uid), cr in crowns.items():
        model = tallo_model(biome[uid])
        h, cd = cr["apex_h_m"].to_numpy(float), cr["crown_diam_m"].to_numpy(float)
        point = (
            np.nansum(model.agb_kg(h, cd)) / 1000.0 / area_ha[uid] if len(cr) else 0.0
        )
        draw = (np.nansum(model.agb_kg(h, cd, coefs=rng_draws[biome[uid]]), axis=1) / 1000.0 / area_ha[uid]
                if len(cr) else np.zeros(1000))  # fmt: skip
        rows_out.append({"source": source, "set": set_name, "engine": engine, "unit_id": uid,
                         "n_crowns": len(cr), "agbd": point,
                         "mean_cd": float(np.mean(cd)) if len(cr) else np.nan})  # fmt: skip
        draws_out.append(draw)
    per_unit = pd.DataFrame(rows_out)
    per_unit.to_parquet(L.TABLE_DIR / "agb_units.parquet", index=False)
    np.save(L.TABLE_DIR / "agb_unit_draws.npy", np.vstack(draws_out))
    log.info("\n%s", agb_scores(per_unit, units).round(3).to_string(index=False))


def _has_tallo(biome: str) -> bool:
    from csdv_core.biomass.allometry import tallo_biomes

    return biome in tallo_biomes()


def agb_scores(per_unit: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    """Held-out skill of crown AGB per surface, parameter set and engine."""
    draws = np.load(L.TABLE_DIR / "agb_unit_draws.npy")
    per_unit = per_unit.reset_index(drop=True)
    u = units[["unit_id", "split", "agbd_Mgha"]]
    rows = []
    for (source, set_name, engine), grp in per_unit.groupby(
        ["source", "set", "engine"]
    ):
        d = grp.merge(u, on="unit_id")
        cal, val = d[d["split"] == "cal"], d[d["split"] == "val"]
        coef = L.line_fit(cal["agbd"].to_numpy(), cal["agbd_Mgha"].to_numpy())
        line = L.fit_metrics(
            val["agbd_Mgha"].to_numpy(), L.line_predict(val["agbd"].to_numpy(), coef)
        )
        raw = L.fit_metrics(val["agbd_Mgha"].to_numpy(), val["agbd"].to_numpy())
        k = cal["agbd_Mgha"].sum() / max(cal["agbd"].sum(), 1e-9)
        val_idx = grp.index[grp["unit_id"].isin(val["unit_id"])]
        mean_draws = draws[val_idx].mean(axis=0)
        rows.append({
            "source": source, "set": set_name, "engine": engine, "n_val": line["n"],
            "r2_line": line["r2"], "rmse_line": line["rmse"],
            "r2_raw": raw["r2"], "bias_raw_pct": raw["bias_pct"], "K": k,
            "val_mean_agbd": float(val["agbd"].mean()),
            "val_mean_lo": float(np.percentile(mean_draws, 2.5)),
            "val_mean_hi": float(np.percentile(mean_draws, 97.5)),
            "field_val_mean": float(val["agbd_Mgha"].mean()),
        })  # fmt: skip
    for source, base in mean_height_baseline(units).groupby("source"):
        d = base.merge(u, on="unit_id")
        cal, val = d[d["split"] == "cal"], d[d["split"] == "val"]
        coef = L.line_fit(cal["mean_h"].to_numpy(), cal["agbd_Mgha"].to_numpy())
        line = L.fit_metrics(
            val["agbd_Mgha"].to_numpy(), L.line_predict(val["mean_h"].to_numpy(), coef)
        )
        rows.append({"source": source, "set": "none", "engine": "mean height", "n_val": line["n"],
                     "r2_line": line["r2"], "rmse_line": line["rmse"],
                     "field_val_mean": float(val["agbd_Mgha"].mean())})  # fmt: skip
    out = pd.DataFrame(rows)
    out.to_csv(L.TABLE_DIR / "agb_scores.csv", index=False)
    return out


def mean_height_baseline(units: pd.DataFrame) -> pd.DataFrame:
    """Plot mean of the raw CHM (NaN and below-ground as 0) on each source."""
    import geopandas as gpd

    geom = L.neon_units()[["unit_id", "geometry"]]
    g = geom.merge(units[["unit_id"]], on="unit_id")
    rows = []
    for source in ("naip", "neon"):
        for unit in g.itertuples():
            chip = L.unit_chip(source, unit.unit_id)
            if not chip.exists():
                continue
            chm, transform, crs = L.read_chip(chip)
            poly = gpd.GeoSeries([unit.geometry], crs=g.crs).to_crs(crs).iloc[0]
            h = np.nan_to_num(chm[L.unit_mask(poly, chm.shape, transform)], nan=0.0)
            rows.append(
                {
                    "source": source,
                    "unit_id": unit.unit_id,
                    "mean_h": float(np.mean(np.maximum(h, 0.0))),
                }
            )
    return pd.DataFrame(rows)


def cmd_agb_score() -> None:
    """Rescore saved AGB results, adding the mean-height baseline."""
    table = pd.read_parquet(L.TABLE_DIR / "neon_units.parquet")
    per_unit = pd.read_parquet(L.TABLE_DIR / "agb_units.parquet")
    log.info("\n%s", agb_scores(per_unit, table).round(3).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "step",
        choices=[
            "detect",
            "grow",
            "speed",
            "speed-worker",
            "synthetic",
            "agb",
            "agb-score",
        ],
    )
    ap.add_argument("worker_args", nargs="*", help=argparse.SUPPRESS)
    ap.add_argument("--reuse", action="store_true", help="Reuse an earlier lidR run.")
    ap.add_argument("--limit", type=int, default=None, help="First N surfaces only.")
    args = ap.parse_args()
    global REUSE, LIMIT
    REUSE, LIMIT = args.reuse, args.limit
    L.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if args.step == "speed-worker":
        import json

        path, radius, reps = args.worker_args
        print(json.dumps(speed_worker(path, float(radius), int(reps))))
        return
    steps = {
        "detect": cmd_detect,
        "grow": cmd_grow,
        "speed": cmd_speed,
        "synthetic": cmd_synthetic,
        "agb": cmd_agb,
        "agb-score": cmd_agb_score,
    }
    steps[args.step]()


if __name__ == "__main__":
    main()
