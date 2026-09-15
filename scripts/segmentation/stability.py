"""Temporal stability of crown metrics, and the resolution artefact.

The NAIP record changes resolution mid-series. 2012 and 2014 were flown at
1.0 m and resampled onto the common 0.6 m grid; 2016 onward are 0.6 m native. A
trajectory product cannot tell a real change from an instrument change unless
that step is measured, so this does two things.

First, it computes the six-year series on the undisturbed tiles and expresses
each year as a deviation from that tile's own mean. Real forest change on an
undisturbed tile over ten years is small, so a large deviation is the pipeline
talking rather than the forest.

Second, it degrades the 2016 CHM from 0.6 to 1.0 m and back, re-segments, and
marks where that lands. The decision rule is fixed in advance:

    reproduces 70 percent or more of the 2012 and 2014 deviation
        the offset is a resolution artefact
    below 30 percent
        the offset is real change, phenology, or model drift
    in between
        report both and adjust nothing

One caveat has to travel with the result. Degrading a finished raster is a
lower bound on the true effect. The 2012 and 2014 CHMs came out of a model
predicting height from blurrier imagery, and most of the artefact lives in that
inference rather than in the resampling. Reproducing it properly means
re-running inference on degraded NAIP, which needs the GPU model and is out of
scope here.

Sensitivity to the degradation is a ranking criterion in its own right. For a
multi-date product, the parameter set that moves least is the better one.

Run:
    .micromamba/envs/CSDV/bin/python scripts/segmentation/stability.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    CANOPY_FLOOR_M,
    NAIP_YEARS,
    NATIVE_RES_M,
    REFERENCE_YEAR,
    configure_logging,
    degrade,
    elkinsville_chm,
    load_tiles,
    params_from_row,
    read_table,
    read_window,
    tile_stats,
    write_table,
)

from csdv_core.segmentation.chm_watershed import segment_crowns  # noqa: E402
from csdv_core.segmentation.params import (  # noqa: E402
    WINDOW_FUNCTIONS,
    SegmentationParams,
)

logger = logging.getLogger("stability")

METRICS = ("density_per_ha", "diam_mean", "diam_cv", "diam_p90")

#: Fractions of the 1 m deviation the degradation must reproduce before the
#: offset is called an artefact. Set before any result was read.
ARTEFACT_THRESHOLD = 0.70
REAL_CHANGE_THRESHOLD = 0.30


def measure(arr, transform, crs, bounds, params) -> dict:
    """Segment one array and summarise it the way the sweep does."""
    px = float(abs(transform.a))
    canopy_area = float(np.sum(np.isfinite(arr) & (arr >= CANOPY_FLOOR_M)) * px * px)
    crowns = segment_crowns(arr, transform, crs, params=params)
    return tile_stats(crowns, bounds, canopy_area)


def chosen_params(args) -> SegmentationParams:
    """The parameter set under test, from the sweep result or the flags."""
    if args.from_sweep:
        scored = read_table("sweep_tune_scored.parquet")
        passing = scored[scored["passes"]]
        if len(passing):
            return params_from_row(passing.iloc[0])
        logger.warning("No passing set in the sweep, falling back to the flags")
    return SegmentationParams(
        smooth_radius_m=args.smooth_radius_m,
        window=WINDOW_FUNCTIONS[args.window],
        th_cr=args.th_cr,
        max_crown_radius_m=(
            None if args.max_crown_radius_m < 0 else args.max_crown_radius_m
        ),
    )


def main() -> int:
    """Six-year series plus the degradation experiment."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-sweep", action="store_true")
    parser.add_argument("--window", default="popescu_linear")
    parser.add_argument("--smooth-radius-m", type=float, default=0.6)
    parser.add_argument("--th-cr", type=float, default=0.55)
    parser.add_argument("--max-crown-radius-m", type=float, default=-1.0)
    parser.add_argument(
        "--all-strata",
        action="store_true",
        help="Include disturbed tiles, where change is real.",
    )
    args = parser.parse_args()

    params = chosen_params(args)
    logger.info("Stability under %s", params.describe())

    tiles = load_tiles("tune")
    tiles = tiles[tiles["site"] == "ElkinsvilleNE"]
    if not args.all_strata:
        tiles = tiles[tiles["stratum"].str.startswith("undisturbed")]
    logger.info("%d tiles", len(tiles))

    rows = []
    for _, tile in tiles.iterrows():
        bounds = (tile["minx"], tile["miny"], tile["maxx"], tile["maxy"])
        for year in NAIP_YEARS:
            arr, transform, crs = read_window(elkinsville_chm(year), bounds)
            stats = measure(arr, transform, crs, bounds, params)
            rows.append(
                {
                    "tile_id": tile["tile_id"],
                    "stratum": tile["stratum"],
                    "year": year,
                    "native_res_m": NATIVE_RES_M[year],
                    "source": "observed",
                    **stats,
                }
            )
        # The degradation control, on the reference year only.
        arr, transform, crs = read_window(elkinsville_chm(REFERENCE_YEAR), bounds)
        coarse = degrade(arr, 1.0 / 0.6)
        stats = measure(coarse, transform, crs, bounds, params)
        rows.append(
            {
                "tile_id": tile["tile_id"],
                "stratum": tile["stratum"],
                "year": REFERENCE_YEAR,
                "native_res_m": 1.0,
                "source": "degraded to 1.0 m",
                **stats,
            }
        )
        logger.info("%s done", tile["tile_id"])

    frame = pd.DataFrame(rows)
    write_table(frame, "stability.parquet")

    observed = frame[frame["source"] == "observed"]
    print("\nSix-year series, mean over tiles")
    print(
        observed.groupby(["year", "native_res_m"])[list(METRICS)]
        .mean()
        .to_string(float_format="%.3f")
    )

    print("\nDeviation from each tile's own mean, and what degradation explains")
    verdicts = []
    for metric in METRICS:
        base = observed.groupby("tile_id")[metric].transform("mean")
        dev = observed.assign(dev=observed[metric] - base)
        coarse_years = dev[dev["native_res_m"] == 1.0]["dev"].mean()
        fine_years = dev[dev["native_res_m"] == 0.6]["dev"].mean()
        one_m_effect = coarse_years - fine_years

        degraded = frame[frame["source"] == "degraded to 1.0 m"]
        merged = degraded.merge(
            observed[observed["year"] == REFERENCE_YEAR][["tile_id", metric]],
            on="tile_id",
            suffixes=("_degraded", "_observed"),
        )
        synth_effect = float(
            (merged[f"{metric}_degraded"] - merged[f"{metric}_observed"]).mean()
        )
        share = synth_effect / one_m_effect if abs(one_m_effect) > 1e-9 else np.nan
        if not np.isfinite(share):
            verdict = "no measurable 1 m offset"
        elif share >= ARTEFACT_THRESHOLD:
            verdict = "resolution artefact"
        elif share <= REAL_CHANGE_THRESHOLD:
            verdict = "not resolution"
        else:
            verdict = "ambiguous, report both"
        verdicts.append(
            {
                "metric": metric,
                "observed_1m_offset": one_m_effect,
                "degradation_offset": synth_effect,
                "share_explained": share,
                "verdict": verdict,
            }
        )
    verdict_frame = pd.DataFrame(verdicts)
    write_table(verdict_frame, "stability_verdicts.csv")
    print(verdict_frame.to_string(index=False, float_format="%.3f"))

    print("\nYear-to-year spread on undisturbed tiles (coefficient of variation)")
    for metric in METRICS:
        per_tile = observed.groupby("tile_id")[metric]
        rel = (per_tile.std() / per_tile.mean()).mean()
        print(f"  {metric:16s} {rel * 100:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
