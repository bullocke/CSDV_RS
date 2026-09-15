"""Can crown_cv recover a crown size distribution it is given?

The null models in ``nulls.py`` ask whether the observed CV could be geometry
alone. This asks the complementary and more useful question. Build canopy
height models whose crown size variability is known by construction, run the
real pipeline over them, and plot what comes out against what went in.

The answer decides what happens to the metric:

    slope near zero, intercept near 0.3
        The measurement is a constant. The five ``crown_cv`` bands in
        ``config/stages.yaml`` cannot separate anything, and the metric needs
        deprecating or redefining.

    slope positive but well below one
        The measurement is real and compressed. The bands need rescaling by the
        measured attenuation, which is a usable result rather than a failure.

    slope near one
        The metric works and the old parameters were the whole problem.

Crowns are paraboloids on a hexagonal lattice jittered to break the regularity,
with radii drawn lognormal at a set CV. Height scales with radius through the
same allometry the scoring uses, so taller trees carry wider crowns, which is
the relationship the segmentation is supposed to find.

Run:
    .micromamba/envs/CSDV/bin/python scripts/segmentation/sensitivity.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    configure_logging,
    degrade,
    write_table,
)

from csdv_core.segmentation.chm_watershed import segment_crowns  # noqa: E402
from csdv_core.segmentation.params import (  # noqa: E402
    WINDOW_FUNCTIONS,
    SegmentationParams,
)

logger = logging.getLogger("sensitivity")

TRUE_CVS = (0.05, 0.15, 0.30, 0.50, 0.70)
N_REPLICATES = 3
SCENE_M = 300.0
PIXEL_M = 0.6
RNG_SEED = 20260803

#: Mean crown radius in metres. Chosen so the scene reaches full canopy closure
#: at a density inside the eastern hardwood range under test.
MEAN_RADIUS_M = 4.5
TARGET_DENSITY_PER_HA = 140.0


def synth_chm(
    true_cv: float,
    rng: np.random.Generator,
    *,
    noise_m: float = 0.0,
    scene_m: float = SCENE_M,
    pixel_m: float = PIXEL_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a CHM from crowns of known size variability.

    Returns the height array and the true crown radii that produced it.
    """
    n_px = int(round(scene_m / pixel_m))
    spacing = np.sqrt(1e4 / TARGET_DENSITY_PER_HA)

    xs, ys = [], []
    row = 0
    y = spacing / 2
    while y < scene_m:
        offset = (spacing / 2) if row % 2 else 0.0
        x = offset + spacing / 2
        while x < scene_m:
            xs.append(x)
            ys.append(y)
            x += spacing
        y += spacing * np.sqrt(3) / 2
        row += 1
    centres = np.column_stack([np.array(xs), np.array(ys)])
    # Jitter, so the pattern is not a perfect lattice whose tessellation would
    # be uniform by construction.
    centres += rng.normal(0.0, spacing * 0.12, size=centres.shape)

    # Lognormal radii at the requested CV, mean held at MEAN_RADIUS_M.
    sigma = np.sqrt(np.log(1.0 + true_cv**2))
    mu = np.log(MEAN_RADIUS_M) - 0.5 * sigma**2
    radii = rng.lognormal(mu, sigma, size=len(centres))

    # Height from radius, inverting the crown-width allometry the scoring uses,
    # so the planted forest obeys the relationship being tested for.
    a, _, c = (3.09632, 0.0, 0.00895)
    widths = 2.0 * radii
    heights = np.sqrt(np.maximum(widths - a, 0.5) / c)
    heights = np.clip(heights, 4.0, 38.0)

    yy, xx = np.mgrid[0:n_px, 0:n_px]
    xx = (xx + 0.5) * pixel_m
    yy = (yy + 0.5) * pixel_m
    chm = np.zeros((n_px, n_px), dtype="float32")
    for (cx, cy), r, h in zip(centres, radii, heights, strict=True):
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        inside = d2 <= r * r
        if not inside.any():
            continue
        # Paraboloid crown: full height at the apex, zero at the edge.
        z = h * (1.0 - d2 / (r * r))
        np.maximum(chm, np.where(inside, z, 0.0).astype("float32"), out=chm)
    if noise_m > 0:
        chm = chm + rng.normal(0.0, noise_m, size=chm.shape).astype("float32")
    chm[chm < 0] = 0.0
    return chm, radii


def measure(chm: np.ndarray, params: SegmentationParams, pixel_m: float) -> dict:
    """Run the pipeline on a synthetic scene and report what it found."""
    transform = Affine(pixel_m, 0.0, 0.0, 0.0, -pixel_m, chm.shape[0] * pixel_m)
    crowns = segment_crowns(
        chm, transform, rasterio.crs.CRS.from_epsg(26916), params=params
    )
    if len(crowns) < 10:
        return {"measured_cv": float("nan"), "n_found": len(crowns)}
    # Drop crowns touching the scene edge, which are truncated by the frame.
    minx, miny, maxx, maxy = 0.0, 0.0, chm.shape[1] * pixel_m, chm.shape[0] * pixel_m
    inset = 20.0
    cx = crowns.geometry.centroid.x.to_numpy()
    cy = crowns.geometry.centroid.y.to_numpy()
    inner = (
        (cx > minx + inset)
        & (cx < maxx - inset)
        & (cy > miny + inset)
        & (cy < maxy - inset)
    )
    kept = crowns.loc[inner]
    if len(kept) < 10:
        return {"measured_cv": float("nan"), "n_found": len(kept)}
    d = kept["crown_diam_m"].to_numpy()
    return {
        "measured_cv": float(d.std() / d.mean()),
        "measured_mean_diam": float(d.mean()),
        "n_found": int(len(kept)),
    }


def main() -> int:
    """Sweep true CV and report the measured response."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", default="popescu_linear")
    parser.add_argument("--smooth-radius-m", type=float, default=0.6)
    parser.add_argument(
        "--noise-m",
        type=float,
        default=0.5,
        help="Height noise, standing in for CHM error.",
    )
    args = parser.parse_args()

    variants = {
        "unbounded": SegmentationParams(
            smooth_radius_m=args.smooth_radius_m,
            window=WINDOW_FUNCTIONS[args.window],
            th_cr=0.0,
            max_crown_radius_m=None,
        ),
        "bounded": SegmentationParams(
            smooth_radius_m=args.smooth_radius_m,
            window=WINDOW_FUNCTIONS[args.window],
            th_cr=0.55,
            max_crown_radius_m=None,
        ),
    }

    rows = []
    for true_cv in TRUE_CVS:
        for rep in range(N_REPLICATES):
            rng = np.random.default_rng(RNG_SEED + rep * 977 + int(true_cv * 1000))
            chm, radii = synth_chm(true_cv, rng, noise_m=args.noise_m)
            planted_cv = float(radii.std() / radii.mean())
            for variant, params in variants.items():
                res = measure(chm, params, PIXEL_M)
                rows.append(
                    {
                        "true_cv": true_cv,
                        "planted_cv": planted_cv,
                        "replicate": rep,
                        "variant": variant,
                        "resolution": "0.6 m native",
                        **res,
                    }
                )
            # And once through the resolution change the record contains.
            coarse = degrade(chm, 1.0 / 0.6)
            for variant, params in variants.items():
                res = measure(coarse, params, PIXEL_M)
                rows.append(
                    {
                        "true_cv": true_cv,
                        "planted_cv": planted_cv,
                        "replicate": rep,
                        "variant": variant,
                        "resolution": "degraded to 1.0 m",
                        **res,
                    }
                )
        logger.info("true CV %.2f done", true_cv)

    frame = pd.DataFrame(rows)
    write_table(frame, "sensitivity.parquet")

    print("\nMeasured crown diameter CV against planted CV")
    pivot = frame.pivot_table(
        index=["variant", "resolution"],
        columns="true_cv",
        values="measured_cv",
        aggfunc="mean",
    )
    print(pivot.to_string(float_format="%.3f"))

    print("\nResponse slope (measured on planted), by variant and resolution")
    for (variant, res), grp in frame.groupby(["variant", "resolution"]):
        ok = grp.dropna(subset=["measured_cv"])
        if len(ok) < 4:
            continue
        slope, intercept = np.polyfit(ok["planted_cv"], ok["measured_cv"], 1)
        corr = float(np.corrcoef(ok["planted_cv"], ok["measured_cv"])[0, 1])
        print(
            f"  {variant:10s} {res:18s} slope={slope:6.3f} "
            f"intercept={intercept:6.3f} r={corr:6.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
