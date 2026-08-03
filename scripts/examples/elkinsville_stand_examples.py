# 1

import sys
from pathlib import Path

# Locate the repository whether Jupyter starts in the root or notebooks/.
working_dir = Path.cwd().resolve()
print(working_dir)
REPO = next(
    (
        path
        for path in (working_dir, *working_dir.parents)
        if (path / "pyproject.toml").is_file()
    ),
    None,
)

if REPO is None:
    raise RuntimeError("Could not locate the CSDV repository root.")
src_path = REPO / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
print(f"Using source package from {src_path}")


# 2

# from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from csdv_core.config import load_stages, load_trajectories
from csdv_core.examples.screen import ScreenCriteria, screen_stands
from csdv_core.io.stands import cover_midpoint, module_footprint, read_ais_stands
from csdv_core.stages.stand import classify_stand_sequence
from csdv_core.trajectories.sequences import build_sequence
from csdv_core.trajectories.stand import blocking_report, classify_stand_trajectory
from csdv_core.viz import cover_class_agreement, save_fig, setup_style
from csdv_core.zonal.compute import (
    DateInputs,
    assert_common_grid,
    compute_module_metrics,
)
from csdv_core.zonal.crowns import segment_scene_crowns

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
setup_style()
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
SITE = "ElkinsvilleNE"
GDB = (
    REPO
    / "data/calibration/Indiana-ElkinsvilleNE_revised.gdb/Indiana-ElkinsvilleNE_revised.gdb"
)
NAIP_DIR = REPO / "data/naip" / SITE
CHM_DIR = REPO / "data/naip_chm" / SITE
RESULTS = REPO / "results/stands" / SITE
RESULTS.mkdir(parents=True, exist_ok=True)

METRICS_PARQUET = RESULTS / "stand_metrics.parquet"
CROWNS_DIR = RESULTS / "crowns"
CROWNS_DIR.mkdir(parents=True, exist_ok=True)
print("Repository:", REPO)

# 3. Stands and which carry an example
stands = read_ais_stands(GDB)
footprint = module_footprint(GDB)
print(f"{len(stands)} disturbance polygons over {footprint.area / 1e6:.1f} km2")
print()
print(
    stands.groupby("dist_label")
    .agg(
        n=("stand_id", "size"),
        acres=("area_m2", lambda s: (s / 4046.856).sum()),
    )
    .sort_values("n", ascending=False)
    .round(1)
    .to_string()
)

# 4.
# Imagery years actually on disk. Falls back to the Planetary Computer set.
years = sorted(int(p.name) for p in CHM_DIR.glob("[12][0-9][0-9][0-9]")) or [
    2012,
    2014,
    2016,
    2018,
    2020,
    2022,
]

screen = screen_stands(stands, ScreenCriteria(years=years))
print(f"{int(screen['passes'].sum())} of {len(screen)} stands pass\n")
print(
    screen[
        [
            "stand_id",
            "dist_label",
            "acres",
            "bbox_fill",
            "n_pre_dates",
            "n_post_dates",
            "passes",
            "fails",
        ]
    ].to_string(index=False, float_format=lambda v: f"{v:6.2f}")
)
screen.to_csv(RESULTS / "screen.csv", index=False)

# 5. Choose example and why it was chosen (gated examples can still be good)
EXAMPLES = {
    "A": ("ELKNE-U2-0-0", "A large windthrow event"),
    "B": ("ELKNE-U44-0-0", "A selection harvest with a large footprint"),
    "C": ("ELKNE-U9-0-0", "A clearcut with 3 images before and after"),
    "D": ("ELKNE-U33-0-0", "Tree mortality"),
    "E": ("ELKNE-U13-0-0", "A small clearcut"),
    "F": ("ELKNE-U23-0-0", "A selterwood establishment cut"),
    "G": ("ELKNE-U48-0-0", "normal development after a clearcut"),
    "H": ("ELKNE-U16-0-0a", "two outcomes inside one disturbance footprint"),
    "H2": ("ELKNE-U16-0-0b", "the other half of the same footprint"),
    "I": ("ELKNE-U17-0-0", "windthrow followed by salvage"),
}
chosen = screen.set_index("stand_id").loc[[sid for sid, _ in EXAMPLES.values()]]
print(
    chosen[
        [
            "dist_label",
            "acres",
            "bbox_fill",
            "n_pre_dates",
            "n_post_dates",
            "passes",
            "fails",
        ]
    ].to_string()
)

# 6. Imagery
manifest_path = CHM_DIR / "manifest.json"
if manifest_path.exists():
    manifest = {int(k): v for k, v in json.loads(manifest_path.read_text()).items()}
else:
    manifest = {}
    print("No manifest; falling back to filename conventions.")


def _one(directory: Path, pattern: str) -> Path | None:
    hits = sorted(directory.glob(pattern))
    return hits[0] if hits else None


print("")
dates = []
for year in years:
    chm = _one(CHM_DIR / str(year), "*.tif")
    naip = _one(NAIP_DIR / str(year), "*.tif")
    if chm is None:
        print(f"{year}: no canopy height model, skipping")
        continue
    entry = manifest.get(year, {})
    tag = entry.get("date_tag") or "".join(c for c in chm.stem if c.isdigit())[-8:]
    dates.append(
        DateInputs(
            year=year,
            date=f"{tag[:4]}-{tag[4:6]}-{tag[6:8]}",
            native_res_m=float(entry.get("native_res_m", 1.0 if year < 2016 else 0.6)),
            chm_path=chm,
            naip_path=naip,
        )
    )
    print(f"{year}: {chm.name}  native {dates[-1].native_res_m} m")

assert dates, "No canopy height models found. Run the Colab notebook first."
assert_common_grid([d.chm_path for d in dates])

# 7. Crown segmentation (slow part)
import rasterio

crowns_by_year = {}
for date in dates:
    cached = CROWNS_DIR / f"crowns_{date.year}.gpkg"
    if cached.exists():
        crowns_by_year[date.year] = gpd.read_file(cached)
        print(f"{date.year}: {len(crowns_by_year[date.year]):6d} crowns (cached)")
        continue
    with rasterio.open(date.chm_path) as src:
        chm = src.read(1, masked=True).filled(np.nan).astype("float32")
        transform, crs = src.transform, src.crs
    found = segment_scene_crowns(chm, transform, crs, block_px=2048, overlap_px=64)
    found.to_file(cached, driver="GPKG")
    crowns_by_year[date.year] = found
    print(
        f"{date.year}: {len(found):6d} crowns, "
        f"mean diameter {found['crown_diam_m'].mean():.2f} m"
    )


# 8. Stand metrics
if METRICS_PARQUET.exists():
    metrics = pd.read_parquet(METRICS_PARQUET)
    print(f"Loaded {len(metrics)} cached stand-date records")
else:
    metrics = compute_module_metrics(stands, dates, crowns_by_year=crowns_by_year)
    metrics.to_parquet(METRICS_PARQUET, index=False)
    print(f"Computed {len(metrics)} stand-date records")

metrics.head(12)

# What could not be computed, and why. A blank in the table should always be
# traceable to a stated cause.
reasons = (
    metrics.loc[metrics["unavailable"] != "", "unavailable"]
    .str.split("; ")
    .explode()
    .value_counts()
)
print(
    f"{int((metrics['unavailable'] != '').sum())} of {len(metrics)} records "
    "have at least one unavailable metric\n"
)
print(reasons.head(20).to_string())

# Stands with substantial nodata cannot be classified and are excluded rather
# than averaged over.
bad = metrics[metrics["support_nodata_fraction"] > 0.05]
if len(bad):
    print("Excluded for nodata inside the stand:")
    print(bad[["stand_id", "year", "support_nodata_fraction"]].to_string(index=False))
    metrics = metrics[metrics["support_nodata_fraction"] <= 0.05].reset_index(drop=True)
else:
    print("No stand-date exceeds 5 percent nodata.")


# 9. External check of itnerpreters
base_year, follow_year = 2018, 2022
pairs = []
for year, tree_col in [
    (base_year, "PercentTreeBase"),
    (follow_year, "PercentTreeFollowUp"),
]:
    subset = metrics[metrics["year"] == year][["stand_id", "crown_fraction"]]
    joined = subset.merge(stands[["stand_id", tree_col]], on="stand_id")
    pairs.append(
        pd.DataFrame(
            {
                "year": year,
                "stand_id": joined["stand_id"],
                "measured": joined["crown_fraction"],
                "cover_class": joined[tree_col],
            }
        )
    )
check = pd.concat(pairs, ignore_index=True)

fig, axes = plt.subplots(
    1, 2, figsize=(10.5, 4.2), sharey=True, constrained_layout=True
)
for ax, year in zip(axes, [base_year, follow_year]):
    subset = check[check["year"] == year]
    rho, n = cover_class_agreement(ax, subset["measured"], subset["cover_class"])
    ax.set_title(f"{year}   rho = {rho:.2f}, n = {n}")
axes[1].set_ylabel("")
save_fig(fig, RESULTS / "cover_agreement.png")
# plt.show()
# How often does the measurement land inside the interpreter's own bin?
from csdv_core.io.stands import cover_bounds

inside = []
for _, row in check.iterrows():
    lo, hi = cover_bounds(row["cover_class"])
    if np.isfinite(lo) and np.isfinite(row["measured"]):
        inside.append(lo <= row["measured"] <= hi)
print(
    f"{np.mean(inside):.1%} of {len(inside)} stand-dates fall inside the "
    "interpreted cover class"
)

# 9b. Satellite metrics
"""
Per-stand Landsat NDVI series, fetched by `csdv satellite fetch` and reduced to
one value a year by `csdv satellite annual`. The rows for the six NAIP years
are joined on so the stage envelopes can read them; the full 1985 onward series
stays in satellite_annual.parquet for the figures and the trajectory rules.
"""
from csdv_core.io.satellite_io import join_satellite_metrics, read_annual

ANNUAL = RESULTS / "satellite_annual.parquet"
if ANNUAL.exists():
    before_unclassified = None
    metrics = join_satellite_metrics(metrics, read_annual(ANNUAL))
    print(
        f"Joined Landsat metrics; {int(metrics['ndvi_mean'].notna().sum())} of "
        f"{len(metrics)} stand-dates have a growing-season NDVI"
    )
else:
    print("No satellite_annual.parquet; run `csdv satellite fetch` then `annual`.")

# 10. Developmental stage
"""
Each stand is compared at each date against the provisional envelopes for every
stage, and the highest scorer wins if it clears the minimum score. The number of
metrics each score rests on is reported with it, because a score of 1.0 on two
metrics is weaker evidence than 0.8 on five.

Site type is not assigned, so every stand uses the unstratified envelope set.
That is the largest single departure from the design, which calls for envelopes
that vary by site type.
"""
stages_cfg = load_stages()
traj_cfg = load_trajectories()
ENVELOPE_METRICS = [
    "gap_fraction",
    "crown_cv",
    "glcm_texture",
    "shrub_fraction",
    "gap_persistence",
    "ndvi_mean",
    "ndvi_seasonal_amplitude",
]

stage_rows = []
for stand_id, group in metrics.groupby("stand_id"):
    group = group.sort_values("year")
    per_date = [
        {name: row[name] for name in ENVELOPE_METRICS if name in group.columns}
        for _, row in group.iterrows()
    ]
    for (_, row), result in zip(
        group.iterrows(), classify_stand_sequence(per_date, stages_cfg)
    ):
        stage_rows.append(
            {
                "stand_id": stand_id,
                "year": row["year"],
                "date": row["date"],
                "stage": result.stage,
                "score": result.score,
                "n_evaluated": result.n_evaluated,
                "runner_up": result.ranked[1][0] if len(result.ranked) > 1 else None,
                "failed": ", ".join(result.failed_metrics),
                "reason": result.reason,
            }
        )
stage_table = pd.DataFrame(stage_rows)
stage_table.to_csv(RESULTS / "stages.csv", index=False)

print(stage_table["stage"].value_counts(dropna=False).to_string())
print(
    f"\nmean metrics evaluated per assignment: {stage_table['n_evaluated'].mean():.1f}"
)

# Stage sequence per example stand.
for letter, (stand_id, note) in EXAMPLES.items():
    subset = stage_table[stage_table["stand_id"] == stand_id].sort_values("year")
    seq = " -> ".join(f"{int(r.year)}:{r.stage or '--'}" for r in subset.itertuples())
    print(f"{letter}  {stand_id:16s} {note}\n     {seq}\n")

# 11.Trajectory
"""
Each stand's stage sequence and metric series are evaluated against the 19
trajectory classes in priority order, first match wins.

Expect very few assignments. Most classes hold a threshold that has never been
filled in, or depend on a metric that has no implementation. The blocking report
below is the substantive result of this section.
"""
traj_rows = []
for stand_id, group in metrics.groupby("stand_id"):
    group = group.sort_values("year")
    if len(group) < 2:
        continue
    stage_seq = (
        stage_table[stage_table["stand_id"] == stand_id]
        .sort_values("year")["stage"]
        .tolist()
    )
    sequence = build_sequence(
        stand_id,
        group,
        stage_seq,
        metric_names=[c for c in group.columns if group[c].dtype.kind == "f"],
    )
    result = classify_stand_trajectory(sequence, traj_cfg, stages_cfg)
    traj_rows.append(
        {
            "stand_id": stand_id,
            "trajectory": result.code,
            "name": result.name,
            "n_dates": result.n_dates,
            "n_evaluable_rules": len(result.evaluable),
        }
    )
trajectories = pd.DataFrame(traj_rows)
trajectories.to_csv(RESULTS / "trajectories.csv", index=False)
print(trajectories["trajectory"].value_counts(dropna=False).to_string())

available = [
    c
    for c in metrics.columns
    if metrics[c].dtype.kind == "f" and metrics[c].notna().any()
]
blocked = blocking_report(traj_cfg, available)
order = list(traj_cfg.trajectory_order)
print(f"{len(order) - len(blocked)} of {len(order)} trajectory rules can fire\n")
for code in order:
    label = traj_cfg.trajectories[code].name
    status = "; ".join(blocked[code]) if code in blocked else "FIREABLE"
    print(f"  {code:5s} {label[:42]:44s} {status}")

pd.DataFrame(
    [
        {
            "code": c,
            "name": traj_cfg.trajectories[c].name,
            "fireable": c not in blocked,
            "blocked_by": "; ".join(blocked.get(c, ())),
        }
        for c in order
    ]
).to_csv(RESULTS / "trajectory_blocking.csv", index=False)

# 12 Figures
"""

The final figures for the classification document are produced by
`Planning/Classification_Alt/make_example_figures.py`, which reads the tables
written above so that the figures can be regenerated without rerunning this
notebook.
"""
