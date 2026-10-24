"""Before and after the flood bound: choose its defaults, then compare engines.

``select``  Grid over ``th_cr`` and ``min_crown_depth_m`` under the flood
            bound, scored on the calibration units of the refAGB split. Every
            other parameter is the optimization study's set. The rule below was
            fixed before any grid result was read.

            Primary: maximise the median share of unit canopy (NAIP-CHM at or
            above 2 m) assigned to a crown, subject to the median ratio of mean
            crown diameter to mean field canopy-tree crown diameter being no
            larger than under the earlier post-trim bound. Fallback, if no set
            meets the constraint: the highest calibration R² of a line on Tallo
            crown AGBD among sets that assign at least 80 percent of the
            canopy. Ties within 0.005 go to the higher ``th_cr``, then to no
            minimum depth.

            Units count as forest when they hold at least three field canopy
            trees.

``units``   Legacy post-trim, the chosen flood set, and the refAGB Tallo set on
            all 626 units: crown counts against field canopy trees, crown size,
            canopy assigned, and held-out AGB skill on the refAGB split.

``strata``  Legacy, flood and lidR ``dalponte2016`` on the 36 strata blocks
            with the same parameters: canopy assigned, crown counts and size,
            run time, maps and CDFs.

Run::

    export REFAGB_METHODS_DIR=<refAGB methods-paper repo>
    P=.micromamba/envs/CSDV/bin/python
    $P scripts/segmentation/compare_lidr/before_after.py select   # ~4 min
    $P scripts/segmentation/compare_lidr/before_after.py units    # ~2 min
    $P scripts/segmentation/compare_lidr/before_after.py strata   # ~10 min
"""

from __future__ import annotations

import argparse
import json
import time

import _lib as L
import geopandas as gpd
import numpy as np
import pandas as pd

from csdv_core.segmentation.chm_watershed import segment_crowns
from csdv_core.segmentation.params import (
    OPTIMIZED_POST_TRIM_PARAMS,
    SegmentationParams,
    tallo_params,
)

log = L.setup_logging("before_after")

TH_CR_GRID = (0.50, 0.55, 0.60, 0.70)
DEPTH_GRID = (None, 2.0, 3.0)
MIN_FOREST_TREES = 3
SELECTION_JSON = L.TABLE_DIR / "flood_default_selection.json"


def grid() -> dict[str, SegmentationParams]:
    base = OPTIMIZED_POST_TRIM_PARAMS.replace(bound_mode="flood")
    sets = {"legacy": OPTIMIZED_POST_TRIM_PARAMS}
    for th in TH_CR_GRID:
        for depth in DEPTH_GRID:
            name = f"flood_th{th:.2f}_d{0 if depth is None else depth:g}"
            sets[name] = base.replace(th_cr=th, min_crown_depth_m=depth)
    return sets


# ---------------------------------------------------------------------------
# NEON units
# ---------------------------------------------------------------------------
def load_units(source: str = "naip") -> gpd.GeoDataFrame:
    table = pd.read_parquet(L.TABLE_DIR / "neon_units.parquet")
    geom = L.neon_units()[["unit_id", "geometry"]]
    units = geom.merge(table, on="unit_id")
    have = units["unit_id"].map(lambda u: L.unit_chip(source, u).exists())
    return units[have].reset_index(drop=True)


def evaluate_units(
    units: gpd.GeoDataFrame,
    params: SegmentationParams | dict[str, SegmentationParams],
    source: str = "naip",
) -> pd.DataFrame:
    """Per-unit crown summaries for one parameter set, or one set per biome."""
    rows = []
    for unit in units.itertuples():
        chm, transform, crs = L.read_chip(L.unit_chip(source, unit.unit_id))
        p = params[unit.biome_long] if isinstance(params, dict) else params
        poly = gpd.GeoSeries([unit.geometry], crs=units.crs).to_crs(crs).iloc[0]
        region = L.unit_mask(poly, chm.shape, transform)
        t0 = time.perf_counter()
        crowns = segment_crowns(chm, transform, crs, params=p)
        seconds = time.perf_counter() - t0
        assigned = L.canopy_assigned(crowns, chm, transform, region=region)
        if not crowns.empty:
            seeds = gpd.points_from_xy(crowns.seed_x, crowns.seed_y, crs=crs)
            crowns = crowns[seeds.within(poly)]
        agb = L.crown_agb_kg(crowns, unit.biome_long)
        area_ha = poly.area / 1e4
        h = chm[region]
        rows.append(
            {
                "unit_id": unit.unit_id,
                "canopy_assigned": assigned,
                "n_crowns": len(crowns),
                "crown_diam": crowns["crown_diam_m"].mean() if len(crowns) else np.nan,
                "crown_apex_h": crowns["apex_h_m"].mean() if len(crowns) else np.nan,
                "tallo_agbd": np.nansum(agb["tallo"]) / 1000.0 / area_ha,
                "jucker_agbd": np.nansum(agb["jucker"]) / 1000.0 / area_ha,
                "mean_h": float(np.nanmean(np.nan_to_num(h, nan=0.0))),
                "seconds": seconds,
            }
        )
    return pd.DataFrame(rows)


def summarise(per_unit: pd.DataFrame, units: pd.DataFrame) -> dict[str, float]:
    u = units.merge(per_unit, on="unit_id")
    forest = u[u["n_canopy"].fillna(0) >= MIN_FOREST_TREES]
    cal = u[u["split"] == "cal"]
    out = {
        "units": len(u),
        "forest_units": len(forest),
        "canopy_assigned_median": float(forest["canopy_assigned"].median()),
        "canopy_assigned_mean": float(forest["canopy_assigned"].mean()),
        "crowns_per_canopy_tree": float(
            forest["n_crowns"].sum() / forest["n_canopy"].sum()
        ),
        "diam_ratio_median": float(
            (forest["crown_diam"] / forest["field_canopy_cd"]).median()
        ),
        "spearman_count": float(
            forest[["n_crowns", "n_canopy"]].corr("spearman").iloc[0, 1]
        ),
        "seconds_per_unit": float(u["seconds"].mean()),
    }
    for col in ("tallo_agbd", "jucker_agbd", "mean_h"):
        coef = L.line_fit(cal[col].to_numpy(), cal["agbd_Mgha"].to_numpy())
        pred = L.line_predict(cal[col].to_numpy(), coef)
        out[f"cal_r2_{col}"] = L.fit_metrics(cal["agbd_Mgha"].to_numpy(), pred)["r2"]
    return out


def apply_rule(scores: pd.DataFrame) -> tuple[str, str]:
    """The pre-registered rule in the module docstring. Returns (name, path)."""
    legacy = scores.loc["legacy"]
    flood = scores.drop(index="legacy").copy()
    flood["th_cr"] = [float(n.split("_th")[1].split("_")[0]) for n in flood.index]
    flood["depth"] = [float(n.split("_d")[1]) for n in flood.index]
    ok = flood[flood["diam_ratio_median"] <= legacy["diam_ratio_median"]]
    if len(ok):
        key, path = "canopy_assigned_median", "primary"
    else:
        ok = flood[flood["canopy_assigned_median"] >= 0.80]
        key, path = "cal_r2_tallo_agbd", "fallback"
    best = ok[key].max()
    tied = ok[ok[key] >= best - 0.005].sort_values(
        ["th_cr", "depth"], ascending=[False, True]
    )
    return str(tied.index[0]), path


def cmd_select() -> None:
    units = load_units()
    rows, per_set = {}, {}
    for name, params in grid().items():
        t0 = time.perf_counter()
        per_unit = evaluate_units(units, params)
        per_set[name] = per_unit.assign(param_set=name)
        rows[name] = summarise(per_unit, units)
        log.info(
            "%-22s %s (%.0f s)",
            name,
            {k: round(v, 3) for k, v in rows[name].items()},
            time.perf_counter() - t0,
        )
    scores = pd.DataFrame(rows).T
    scores.to_csv(L.TABLE_DIR / "flood_grid_scores.csv")
    pd.concat(per_set.values()).to_parquet(
        L.TABLE_DIR / "flood_grid_units.parquet", index=False
    )
    chosen, path = apply_rule(scores)
    params = grid()[chosen]
    SELECTION_JSON.write_text(
        json.dumps(
            {
                "chosen": chosen,
                "rule_path": path,
                "params": params.as_dict(),
                "key": params.key,
            },
            indent=2,
        )
    )
    log.info("Chosen: %s by the %s rule (%s)", chosen, path, params.describe())


def chosen_params() -> SegmentationParams:
    """The selected flood set, with exact window distances as the engine now uses.

    The grid was scored with radii rounded to pixels, the detector of the
    earlier engine, so that only the bound differed from the legacy set.
    """
    sel = json.loads(SELECTION_JSON.read_text())
    return grid()[sel["chosen"]].replace(window_metric="exact")


def cmd_units() -> None:
    units = load_units()
    sets = {
        "legacy": OPTIMIZED_POST_TRIM_PARAMS,
        "flood": chosen_params(),
        "tallo_flood": {b: tallo_params(b) for b in units["biome_long"].unique()},
        "tallo_flood_mean": {
            b: tallo_params(b, th_cr_ref="mean") for b in units["biome_long"].unique()
        },
    }
    per = []
    for name, params in sets.items():
        per.append(evaluate_units(units, params).assign(param_set=name))
        log.info("%s done", name)
    per_unit = pd.concat(per)
    per_unit.to_parquet(L.TABLE_DIR / "units_before_after.parquet", index=False)
    log.info("\n%s", gate_d(per_unit, units).round(3).to_string(index=False))


def gate_d(per_unit: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    """Held-out skill on the refAGB split, per parameter set and predictor."""
    rows = []
    for name, grp in per_unit.groupby("param_set", sort=False):
        u = units.merge(grp, on="unit_id")
        cal, val = u[u["split"] == "cal"], u[u["split"] == "val"]
        for col in ("tallo_agbd", "jucker_agbd", "mean_h"):
            coef = L.line_fit(cal[col].to_numpy(), cal["agbd_Mgha"].to_numpy())
            m = L.fit_metrics(
                val["agbd_Mgha"].to_numpy(), L.line_predict(val[col].to_numpy(), coef)
            )
            rows.append({"param_set": name, "model": f"line on {col}", **m})
        for col in ("tallo_agbd", "jucker_agbd"):
            m = L.fit_metrics(val["agbd_Mgha"].to_numpy(), val[col].to_numpy())
            rows.append({"param_set": name, "model": f"{col}, no fitting", **m})
            k = cal["agbd_Mgha"].sum() / cal[col].sum()
            m = L.fit_metrics(val["agbd_Mgha"].to_numpy(), k * val[col].to_numpy())
            rows.append({"param_set": name, "model": f"{col} x K ({k:.2f})", **m})
    out = pd.DataFrame(rows)
    out.to_csv(L.TABLE_DIR / "units_gate_d.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Strata blocks
# ---------------------------------------------------------------------------
def cmd_strata(figures_only: bool = False) -> None:
    import strata

    tag, engines = "before_after", ["legacy", "flood", "lidr"]
    table_path = L.TABLE_DIR / f"strata_{tag}.csv"
    if figures_only and table_path.exists():
        table = pd.read_csv(table_path)
    else:
        table = strata.run(
            {"legacy": OPTIMIZED_POST_TRIM_PARAMS, "flood": chosen_params()},
            lidr_params=OPTIMIZED_POST_TRIM_PARAMS,
            tag=tag,
        )
        log.info("\n%s", table.round(3).to_string(index=False))
    figs = L.FIG_DIR / tag
    figs.mkdir(parents=True, exist_ok=True)
    strata.canopy_dotplot(table, engines, figs / "canopy_assigned.png")
    strata.cdf_grid(
        tag, engines, "crown_diam_m", "Crown diameter (m)", figs / "cdf_diameter.png"
    )
    strata.cdf_grid(
        tag, engines, "apex_h_m", "Apex height (m)", figs / "cdf_height.png"
    )
    for s in table["stratum"].unique():
        if s != "desert":
            strata.stratum_figure(tag, s, engines, figs / f"map_{s}.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["select", "units", "strata"])
    ap.add_argument(
        "--figures-only", action="store_true", help="strata: redraw from saved crowns"
    )
    args = ap.parse_args()
    L.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if args.step == "strata":
        cmd_strata(args.figures_only)
    else:
        {"select": cmd_select, "units": cmd_units}[args.step]()


if __name__ == "__main__":
    main()
