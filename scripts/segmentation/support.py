"""How many crowns does a stand need before crown_cv means anything?

``MIN_CROWNS = 3`` in ``csdv_core/zonal/crowns.py`` is inherited from the
windowed implementation and its own comment concedes it is probably too low. It
has never been measured. This measures it, and ties the answer to something
external rather than to an eyeball judgement about when a curve flattens.

The tie is the classifier's own resolution. ``config/stages.yaml`` splits
``crown_cv`` into bands, and the narrowest is 0.10 wide. A measurement whose
confidence interval is wider than a band cannot place a stand in that band. So
the support floor is the crown count at which the bootstrap interval fits
inside the narrowest band.

There is an analytic expectation to check the bootstrap against. For a roughly
normal sample, ``SE(CV) = CV * sqrt((1 + 2*CV^2) / (2n))``, which at CV 0.33
gives 0.052 at n = 25 and 0.026 at n = 100.

The result matters beyond the threshold itself. At a given crown density, a
support floor converts directly into a minimum stand area, and that decides how
many of the 40 calibration stands can carry the metric at all.

Run:
    .micromamba/envs/CSDV/bin/python scripts/segmentation/support.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    REFERENCE_YEAR,
    REPO,
    configure_logging,
    elkinsville_chm,
    load_tiles,
    params_from_row,
    read_table,
    segment_tile,
    stands,
    write_table,
)

from csdv_core.segmentation.params import (  # noqa: E402
    WINDOW_FUNCTIONS,
    SegmentationParams,
)

logger = logging.getLogger("support")

SAMPLE_SIZES = (3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500)
N_BOOTSTRAP = 500
RNG_SEED = 20260803
ACRES_PER_HA = 2.47105


def crown_cv_bands() -> dict[str, tuple[float, float]]:
    """Every crown_cv envelope in ``config/stages.yaml``, keyed by stage.

    Schema is ``stages.<stage_code>.envelopes.<site_type>.<metric>``. A null
    bound means no constraint on that side.
    """
    path = REPO / "src" / "csdv_core" / "config" / "stages.yaml"
    doc = yaml.safe_load(path.read_text())
    out: dict[str, tuple[float, float]] = {}
    for stage, spec in (doc.get("stages") or {}).items():
        for envelope in (spec.get("envelopes") or {}).values():
            band = (envelope or {}).get("crown_cv")
            if not isinstance(band, dict):
                continue
            lo = band.get("min")
            hi = band.get("max")
            out[stage] = (
                -np.inf if lo is None else float(lo),
                np.inf if hi is None else float(hi),
            )
            break
    return out


def narrowest_band() -> float:
    """Width of the narrowest bounded crown_cv envelope.

    This is the finest distinction the stage classifier is asked to make with
    the metric, so it is what a crown_cv measurement has to resolve.
    """
    widths = [
        hi - lo
        for lo, hi in crown_cv_bands().values()
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo
    ]
    if not widths:
        logger.warning("No bounded crown_cv bands found, defaulting to 0.10")
        return 0.10
    return float(min(widths))


def analytic_se(cv: float, n: np.ndarray) -> np.ndarray:
    """Standard error of a coefficient of variation for a normal sample."""
    return cv * np.sqrt((1.0 + 2.0 * cv**2) / (2.0 * n))


def main() -> int:
    """Bootstrap the support curve and convert it to a minimum stand area."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-sweep", action="store_true")
    parser.add_argument("--window", default="popescu_linear")
    parser.add_argument("--smooth-radius-m", type=float, default=0.6)
    parser.add_argument("--th-cr", type=float, default=0.55)
    args = parser.parse_args()

    if args.from_sweep:
        scored = read_table("sweep_tune_scored.parquet")
        passing = scored[scored["passes"]]
        params = (
            params_from_row(passing.iloc[0])
            if len(passing)
            else SegmentationParams(
                smooth_radius_m=args.smooth_radius_m,
                window=WINDOW_FUNCTIONS[args.window],
                th_cr=args.th_cr,
            )
        )
    else:
        params = SegmentationParams(
            smooth_radius_m=args.smooth_radius_m,
            window=WINDOW_FUNCTIONS[args.window],
            th_cr=args.th_cr,
            max_crown_radius_m=None,
        )
    logger.info("Support under %s", params.describe())

    # Pool crowns from the closed-canopy tuning tiles. The support question is
    # about sample size, so the pool should be one condition rather than a mix.
    tiles = load_tiles("tune")
    tiles = tiles[tiles["stratum"] == "undisturbed_tall"]
    chm = elkinsville_chm(REFERENCE_YEAR)
    pool = []
    density = []
    for _, tile in tiles.iterrows():
        bounds = (tile["minx"], tile["miny"], tile["maxx"], tile["maxy"])
        crowns, _ = segment_tile(chm, bounds, params)
        if len(crowns):
            pool.append(crowns["crown_diam_m"].to_numpy(dtype="float64"))
            area_ha = (
                (tile["maxx"] - tile["minx"]) * (tile["maxy"] - tile["miny"]) / 1e4
            )
            density.append(len(crowns) / area_ha)
    diameters = np.concatenate(pool)
    crowns_per_ha = float(np.mean(density))
    logger.info("Pool: %d crowns, %.1f per hectare", diameters.size, crowns_per_ha)

    rng = np.random.default_rng(RNG_SEED)
    band = narrowest_band()
    rows = []
    for n in SAMPLE_SIZES:
        if n > diameters.size:
            continue
        draws = rng.choice(diameters, size=(N_BOOTSTRAP, n), replace=True)
        cvs = draws.std(axis=1, ddof=0) / draws.mean(axis=1)
        lo, hi = np.percentile(cvs, [5, 95])
        rows.append(
            {
                "n_crowns": n,
                "cv_mean": float(cvs.mean()),
                "cv_std": float(cvs.std()),
                "ci90_low": float(lo),
                "ci90_high": float(hi),
                "ci90_width": float(hi - lo),
                "analytic_se": float(analytic_se(float(cvs.mean()), np.array(n))),
                "area_ha_at_density": n / crowns_per_ha,
                "area_acres_at_density": n / crowns_per_ha * ACRES_PER_HA,
            }
        )
    frame = pd.DataFrame(rows)
    frame["band_width"] = band
    frame["fits_in_band"] = frame["ci90_width"] <= band
    write_table(frame, "support.parquet")

    print(f"\nNarrowest crown_cv band in stages.yaml: {band:.3f}")
    print("\nBootstrap support curve")
    print(
        frame[
            [
                "n_crowns",
                "cv_mean",
                "ci90_width",
                "analytic_se",
                "fits_in_band",
                "area_acres_at_density",
            ]
        ].to_string(index=False, float_format="%.3f")
    )

    ok = frame[frame["fits_in_band"]]
    if len(ok):
        floor = int(ok["n_crowns"].iloc[0])
        acres = float(ok["area_acres_at_density"].iloc[0])
        print(
            f"\nSupport floor: {floor} crowns, the first sample size whose 90 percent "
            f"interval ({float(ok['ci90_width'].iloc[0]):.3f}) fits inside the "
            f"{band:.2f} band."
        )
        print(
            f"At {crowns_per_ha:.0f} crowns per hectare that is "
            f"{floor / crowns_per_ha:.2f} ha, about {acres:.1f} acres."
        )
    else:
        floor = int(frame["n_crowns"].max())
        acres = float(frame["area_acres_at_density"].iloc[-1])
        print(
            "\nNo tested sample size gives an interval narrower than the band. "
            "crown_cv cannot place a stand in a band at any stand size present "
            "in this dataset."
        )

    # What the floor does to the calibration population.
    stand_gdf = stands()
    stand_gdf["expected_crowns"] = stand_gdf["area_m2"] / 1e4 * crowns_per_ha
    for label, threshold in (("MIN_CROWNS=3", 3), (f"floor={floor}", floor)):
        clear = int((stand_gdf["expected_crowns"] >= threshold).sum())
        print(
            f"  {label:16s} {clear:2d} of {len(stand_gdf)} calibration stands "
            f"reach the threshold"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
