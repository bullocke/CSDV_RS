#!/usr/bin/env python
"""Write the batch files of a block of quads inside one state.

Inputs: a bounding box in degrees, a state (postal code) and the outlines of
the states (``data/reference/cb_2023_us_state_20m.zip``, Census Bureau,
1:20,000,000, hence the margin). Outputs: ``config/grid/<name>.csv`` or its
parts ``<name>_1.csv``, ``<name>_2.csv``, ..., and a small map of the block as
a PNG beside ``--map``.

Run from the repository root, seconds:
    python scripts/grid/block_csv.py blue_ridge --state NC \
        --bbox -84.0 35.0 -81.5 36.25 --part-size 40
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from csdv_core.grid.blocks import block_units, write_batches
from csdv_core.grid.units import read_batch

log = logging.getLogger("block_csv")
ROOT = Path(__file__).resolve().parents[2]
STATES = ROOT / "data/reference/cb_2023_us_state_20m.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("name", help="Name of the block, e.g. blue_ridge")
    parser.add_argument("--state", required=True, help="Postal code, e.g. NC")
    parser.add_argument(
        "--bbox", nargs=4, type=float, required=True, metavar=("W", "S", "E", "N")
    )
    parser.add_argument("--margin-km", type=float, default=2.0)
    parser.add_argument("--part-size", type=int, default=40)
    parser.add_argument(
        "--skip-batch", nargs="*", default=[], help="Batch names whose quads to skip"
    )
    parser.add_argument("--states", type=Path, default=STATES)
    parser.add_argument("--map", type=Path, default=None, help="PNG of the block")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import geopandas as gpd

    states = gpd.read_file(f"zip://{args.states}")
    outline = states.loc[states["STUSPS"] == args.state.upper(), "geometry"]
    if outline.empty:
        raise SystemExit(f"No state {args.state!r} in {args.states}")
    skip = {
        unit.unit_id
        for batch in args.skip_batch
        for unit in read_batch(ROOT / "config/grid" / f"{batch}.csv")
    }
    units = block_units(
        tuple(args.bbox),
        outline.iloc[0],
        margin_km=args.margin_km,
        skip=skip,
        prefix=f"{args.state.upper()}_",
    )
    log.info(
        "%s: %d quads inside %s, about %d GB", args.name, len(units), args.state,
        len(units) * 12,
    )  # fmt: skip
    if args.map is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from shapely.geometry import box

        fig, ax = plt.subplots(figsize=(7, 5))
        w, s, e, n = args.bbox
        states.cx[w - 1 : e + 1, s - 1 : n + 1].boundary.plot(
            ax=ax, color="0.4", lw=0.8
        )
        gpd.GeoSeries([box(*u.bbox_4326) for u in units]).plot(
            ax=ax, facecolor="#4c9f70", edgecolor="white", lw=0.3
        )
        ax.set_xlim(w - 0.6, e + 0.6)
        ax.set_ylim(s - 0.5, n + 0.5)
        ax.set_title(f"{args.name}: {len(units)} quads in {args.state.upper()}")
        args.map.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.map, dpi=130, bbox_inches="tight")
        log.info("Wrote %s", args.map)
    if args.dry_run:
        return
    w, s, e, n = args.bbox
    note = (
        f"Block {args.name}: the 7.5' quads of {w} to {e} E and {s} to {n} N\n"
        f"that lie wholly inside {args.state.upper()} "
        f"(margin {args.margin_km:g} km), north to south.\n"
        "Written by scripts/grid/block_csv.py. Submit with\n"
        f"scripts/chpc/region.sh block {args.name}."
    )
    write_batches(
        units, ROOT / "config/grid", args.name, part_size=args.part_size, note=note
    )


if __name__ == "__main__":
    main()
