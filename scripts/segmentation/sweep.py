"""Run the segmentation parameter sweep over the frozen tiles.

Two stages. A coarse screen on a few tiles drops the regions of the grid that
cannot work, then the survivors run on every tuning tile. The screen exists
only to save time, and its threshold is deliberately loose so it cannot decide
anything the decision rule should decide.

Results are one row per (parameter set, tile), written as Parquet so every
number in the report traces back to a row.

Run:
    .micromamba/envs/CSDV/bin/python scripts/segmentation/sweep.py
    .micromamba/envs/CSDV/bin/python scripts/segmentation/sweep.py --split holdout
    .micromamba/envs/CSDV/bin/python scripts/segmentation/sweep.py --stage full
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    REFERENCE_YEAR,
    apply_decision_rule,
    configure_logging,
    elkinsville_chm,
    load_tiles,
    parameter_grid,
    params_row,
    save_manifest,
    segment_tile,
    tile_stats,
    transfer_sites,
    write_table,
)

logger = logging.getLogger("sweep")

#: Tiles used for the coarse screen. One from each of the strata where the
#: parameters behave most differently.
SCREEN_STRATA = ("undisturbed_tall", "clearcut", "wind", "edge_and_gap")

#: A parameter set has to clear this on the screen tiles to earn a full run.
#: Wider than the decision rule on purpose. This step removes the hopeless, it
#: does not choose the winner.
SCREEN_DENSITY = (30.0, 400.0)


def chm_for(tile: pd.Series) -> Path:
    """Locate the CHM a tile belongs to."""
    if tile["site"] == "ElkinsvilleNE":
        return elkinsville_chm(REFERENCE_YEAR)
    for site in transfer_sites():
        if site.name == tile["site"]:
            return site.chm
    raise KeyError(f"No CHM for site {tile['site']}")


def run_grid(tiles: pd.DataFrame, grid, *, label: str) -> pd.DataFrame:
    """Segment every tile under every parameter set."""
    rows = []
    total = len(tiles) * len(grid)
    started = time.time()
    done = 0
    for _, tile in tiles.iterrows():
        chm = chm_for(tile)
        bounds = (tile["minx"], tile["miny"], tile["maxx"], tile["maxy"])
        for params in grid:
            crowns, canopy_area = segment_tile(chm, bounds, params)
            row = {
                "tile_id": tile["tile_id"],
                "site": tile["site"],
                "stratum": tile["stratum"],
                "split": tile["split"],
                "year": REFERENCE_YEAR if tile["site"] == "ElkinsvilleNE" else 2023,
                **params_row(params),
                **tile_stats(crowns, bounds, canopy_area),
            }
            rows.append(row)
            done += 1
            if done % 100 == 0:
                rate = done / max(time.time() - started, 1e-9)
                logger.info(
                    "%s: %d/%d (%.1f/s, %.0f s left)",
                    label,
                    done,
                    total,
                    rate,
                    (total - done) / max(rate, 1e-9),
                )
    return pd.DataFrame(rows)


def summarise(sweep: pd.DataFrame) -> pd.DataFrame:
    """Average each parameter set over its tiles.

    The mean is taken over tiles so that no stratum dominates by having more
    crowns. A tile is one observation regardless of how much canopy it holds.
    """
    numeric = [
        c
        for c in sweep.columns
        if sweep[c].dtype.kind in "fi"
        and c not in {"year", "smooth_radius_m", "th_cr", "max_crown_radius_m"}
    ]
    grouped = sweep.groupby(
        ["key", "smooth_radius_m", "window", "th_cr", "max_crown_radius_m"],
        dropna=False,
    )
    out = grouped[numeric].mean().reset_index()
    out["n_tiles"] = grouped.size().to_numpy()
    return out


def main() -> int:
    """Screen the grid, then run the survivors on the tuning tiles."""
    configure_logging()
    # The engine logs one line per call, and the sweep makes thousands.
    logging.getLogger("csdv_core.segmentation.chm_watershed").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="tune", help="tune, holdout, or a site")
    parser.add_argument(
        "--stage",
        default="both",
        choices=["screen", "full", "both"],
        help="Which stage to run.",
    )
    args = parser.parse_args()

    grid = parameter_grid()
    logger.info("Full grid: %d parameter sets", len(grid))

    survivors = grid
    if args.stage in {"screen", "both"}:
        tiles = load_tiles("tune")
        screen_tiles = (
            tiles[tiles["stratum"].isin(SCREEN_STRATA)]
            .drop_duplicates("stratum")
            .reset_index(drop=True)
        )
        logger.info("Screening on %d tiles", len(screen_tiles))
        screen = run_grid(screen_tiles, grid, label="screen")
        write_table(screen, "sweep_screen.parquet")

        keep = (
            screen.groupby("key")["density_per_ha"]
            .mean()
            .between(*SCREEN_DENSITY)
            .pipe(lambda s: set(s[s].index))
        )
        survivors = [p for p in grid if p.key in keep]
        logger.info(
            "Screen kept %d of %d parameter sets (density %g to %g per ha)",
            len(survivors),
            len(grid),
            *SCREEN_DENSITY,
        )
        if args.stage == "screen":
            return 0

    tiles = load_tiles(args.split)
    logger.info(
        "Full run: %d sets on %d %s tiles", len(survivors), len(tiles), args.split
    )
    sweep = run_grid(tiles, survivors, label=f"full-{args.split}")
    write_table(sweep, f"sweep_{args.split}.parquet")

    summary = summarise(sweep)
    scored = apply_decision_rule(summary)
    write_table(scored, f"sweep_{args.split}_scored.parquet")
    save_manifest(
        {
            "split": args.split,
            "n_parameter_sets_total": len(grid),
            "n_parameter_sets_run": len(survivors),
            "n_tiles": len(tiles),
            "screen_density_band": SCREEN_DENSITY,
        },
        f"sweep_{args.split}_manifest.json",
    )

    passing = scored[scored["passes"]]
    logger.info("%d of %d sets pass every filter", len(passing), len(scored))
    cols = [
        "window",
        "smooth_radius_m",
        "th_cr",
        "max_crown_radius_m",
        "density_per_ha",
        "diam_mean",
        "diam_cv",
        "assigned_fraction",
        "capped_fraction",
        "allo_slope",
        "allo_rmse",
    ]
    if len(passing):
        print("\nTop passing parameter sets:")
        print(passing.head(10)[cols].to_string(index=False, float_format="%.3f"))
    else:
        print("\nNo parameter set passes every filter. Nearest misses:")
        fails = [c for c in scored.columns if c.startswith("fail_")]
        scored["n_fail"] = scored[fails].sum(axis=1)
        print(
            scored.sort_values(["n_fail", "allo_rmse"])
            .head(10)[cols + fails]
            .to_string(index=False, float_format="%.3f")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
