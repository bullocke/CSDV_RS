"""Re-segment Elkinsville with the chosen parameters and rebuild stand metrics.

Writes crowns to a directory keyed by the parameter hash, with a sidecar JSON
recording exactly what produced them. The previous cache keyed on filename
alone, so a re-run after a parameter change silently reused the old crowns, and
deleting some years but not others left a metrics table that mixed two
parameter sets across the series. Keying on the hash removes that failure.

The stand metric table is rebuilt from the new crowns. Only the crown family
changes; every other metric is read straight through from the existing table,
because nothing else depends on segmentation.

Run:
    .micromamba/envs/CSDV/bin/python scripts/segmentation/apply.py --dry-run
    .micromamba/envs/CSDV/bin/python scripts/segmentation/apply.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    NAIP_YEARS,
    SITE,
    configure_logging,
    elkinsville_chm,
    params_from_row,
    read_table,
    stands,
)

from csdv_core.io.paths import project_paths  # noqa: E402
from csdv_core.segmentation.params import (  # noqa: E402
    DEFAULT_PARAMS,
    WINDOW_FUNCTIONS,
    SegmentationParams,
)
from csdv_core.zonal.crowns import (  # noqa: E402
    crown_diameter_stats,
    crowns_in_stand,
    segment_scene_crowns,
)

logger = logging.getLogger("apply")

CROWN_METRIC_COLUMNS = (
    "crown_cv",
    "crown_p90",
    "crown_mean",
    "crown_median",
    "crown_std",
    "crown_count",
)


def results_root() -> Path:
    """Where the stand products live."""
    return project_paths().results_root / "stands" / SITE


def crown_dir(params: SegmentationParams) -> Path:
    """Crown directory for one parameter set."""
    return results_root() / "crowns" / f"seg-{params.key}"


def segment_year(year: int, params: SegmentationParams, *, force: bool = False) -> Path:
    """Segment one year, or reuse it when the sidecar proves it matches."""
    out_dir = crown_dir(params)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"crowns_{year}.gpkg"
    sidecar = out.with_suffix(".gpkg.params.json")

    if out.exists() and sidecar.exists() and not force:
        recorded = json.loads(sidecar.read_text()).get("key")
        if recorded == params.key:
            logger.info("%d: reusing %s", year, out.name)
            return out
        logger.warning("%d: sidecar key mismatch, re-segmenting", year)

    chm_path = elkinsville_chm(year)
    started = time.time()
    with rasterio.open(chm_path) as src:
        arr = src.read(1).astype("float32")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        arr[arr < -100] = np.nan
        transform, crs = src.transform, src.crs
    crowns = segment_scene_crowns(arr, transform, crs, block_px=2048, params=params)
    crowns.to_file(out, driver="GPKG")
    sidecar.write_text(
        json.dumps(
            {
                "key": params.key,
                "engine": "watershed",
                "site": SITE,
                "year": year,
                "chm": str(chm_path),
                "n_crowns": int(len(crowns)),
                "params": params.as_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    logger.info(
        "%d: %d crowns, mean diam %.2f m, %.0f s",
        year,
        len(crowns),
        float(crowns["crown_diam_m"].mean()) if len(crowns) else float("nan"),
        time.time() - started,
    )
    return out


def rebuild_stand_metrics(params: SegmentationParams, *, min_crowns: int) -> Path:
    """Recompute the crown family for every stand and year."""
    metrics_path = results_root() / "stand_metrics.parquet"
    table = pd.read_parquet(metrics_path)
    stand_gdf = stands().set_index("stand_id")
    logger.info("Rebuilding %d rows in %s", len(table), metrics_path.name)

    updates: dict[tuple[str, int], dict[str, float]] = {}
    for year in NAIP_YEARS:
        crowns = gpd.read_file(crown_dir(params) / f"crowns_{year}.gpkg")
        for stand_id, stand in stand_gdf.iterrows():
            inside = crowns_in_stand(crowns, stand.geometry)
            stats = crown_diameter_stats(inside, min_crowns=min_crowns)
            updates[(str(stand_id), year)] = stats.as_metrics()
        logger.info("%d: %d stands", year, len(stand_gdf))

    for column in CROWN_METRIC_COLUMNS:
        table[column] = [
            updates.get((row.stand_id, int(row.year)), {}).get(column, np.nan)
            for row in table.itertuples()
        ]
    # The crown deltas are recomputed from the refreshed values.
    for delta, source in (
        ("d_crown_p90", "crown_p90"),
        ("d_crown_count", "crown_count"),
    ):
        if delta in table.columns:
            table[delta] = table.sort_values("year").groupby("stand_id")[source].diff()

    backup = metrics_path.with_suffix(".parquet.pre-segmentation")
    if not backup.exists():
        metrics_path.replace(backup)
        logger.info("Kept the previous table at %s", backup.name)
    table.to_parquet(metrics_path, index=False)
    logger.info("Wrote %s", metrics_path)
    return metrics_path


def main() -> int:
    """Segment every year and rebuild the stand metric table."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-sweep",
        action="store_true",
        help=(
            "Use the pre-registered rule's winner from the tuning tiles. That "
            "set over-segments on the lidar transfer sites, so the production "
            "default is used unless this is passed deliberately."
        ),
    )
    parser.add_argument("--window", default=None)
    parser.add_argument("--smooth-radius-m", type=float, default=None)
    parser.add_argument("--th-cr", type=float, default=None)
    parser.add_argument("--max-crown-radius-m", type=float, default=None)
    parser.add_argument("--min-crowns", type=int, default=None)
    parser.add_argument(
        "--years",
        default=None,
        help=(
            "Comma separated years to segment. Segmenting the whole scene holds "
            "300k polygons in memory, which does not fit alongside another "
            "year on a small machine, so one year per process is the safe way "
            "to run this."
        ),
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Segment only. Use when running years in separate processes.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    params = None
    if args.from_sweep:
        scored = read_table("sweep_tune_scored.parquet")
        passing = scored[scored["passes"]]
        if len(passing):
            params = params_from_row(passing.iloc[0])
            logger.warning(
                "Using the rule's winner from the tuning tiles. It reaches 200 "
                "and 234 crowns per hectare on the lidar transfer sites."
            )
    if params is None:
        # The production set. Any flag given explicitly overrides one field.
        params = DEFAULT_PARAMS.replace(
            **{
                k: v
                for k, v in {
                    "smooth_radius_m": args.smooth_radius_m,
                    "th_cr": args.th_cr,
                    "max_crown_radius_m": (
                        None
                        if args.max_crown_radius_m is not None
                        and args.max_crown_radius_m < 0
                        else args.max_crown_radius_m
                    ),
                }.items()
                if v is not None
            }
        )
        if args.window:
            params = params.replace(window=WINDOW_FUNCTIONS[args.window])
    min_crowns = args.min_crowns
    if min_crowns is None:
        try:
            support = read_table("support.parquet")
            fits = support[support["fits_in_band"]]
            min_crowns = int(fits["n_crowns"].iloc[0]) if len(fits) else 3
        except FileNotFoundError:
            min_crowns = 3

    logger.info("Parameters %s: %s", params.key, params.describe())
    logger.info("Crown directory: %s", crown_dir(params))
    logger.info("MIN_CROWNS: %d", min_crowns)
    if args.dry_run:
        logger.info("Dry run, nothing written")
        return 0

    years = [int(y) for y in args.years.split(",")] if args.years else list(NAIP_YEARS)
    for year in years:
        segment_year(year, params, force=args.force)
    if args.skip_metrics:
        logger.info("Skipping the metric rebuild as asked")
        return 0
    rebuild_stand_metrics(params, min_crowns=min_crowns)
    logger.info(
        "Done. stages.csv and trajectories.csv are now stale and were not "
        "regenerated, and the crown_cv bands in config/stages.yaml were "
        "calibrated against the previous segmentation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
