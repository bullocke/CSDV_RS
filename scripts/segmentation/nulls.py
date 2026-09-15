"""Null models for crown diameter CV.

``crown_cv`` reads about 0.33 on every stand large enough to be stable, at
every date, at every site tested, and at every seed spacing tried. That is the
behaviour of a geometric constant, not of a forest measurement.

There is a prediction to test against. A watershed seeded by points and run to
completion is a tessellation. For a Poisson-Voronoi tessellation the cell area
distribution is close to a gamma with shape 3.57 (Ferenc and Neda 2007, Physica
A 385: 518-526), so the area CV is about ``1/sqrt(3.57) = 0.53``. Diameter goes
as the square root of area, which to first order halves the CV, giving about
0.26. Hard-core point patterns, where seeds cannot be arbitrarily close, are
more regular and sit a little lower.

Three tests, cheapest first.

Geometry null
    Keep the real seed positions, throw the canopy height model away, and
    tessellate a flat surface. Then repeat with random seeds at matched
    density, and with a hard-core pattern at matched density and matched
    minimum separation. If the real result sits among these, the metric carries
    no information about the forest.

Surface null
    Keep the real seeds and the real height histogram, but destroy the spatial
    arrangement by phase randomisation. Isolates what the canopy surface
    contributes beyond where the seeds are.

Run:
    .micromamba/envs/CSDV/bin/python scripts/segmentation/nulls.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    CANOPY_FLOOR_M,
    REFERENCE_YEAR,
    configure_logging,
    elkinsville_chm,
    load_tiles,
    read_window,
    transfer_sites,
    write_table,
)

from csdv_core.preprocess.chm import mask_below, smooth_chm  # noqa: E402
from csdv_core.segmentation.chm_watershed import (  # noqa: E402
    locate_seeds,
    smooth_kernel_px,
)
from csdv_core.segmentation.params import (  # noqa: E402
    WINDOW_FUNCTIONS,
    SegmentationParams,
)

logger = logging.getLogger("nulls")

#: Theoretical diameter CV for a Poisson-Voronoi tessellation. Area CV is
#: 1/sqrt(3.57); diameter is the square root of area, which roughly halves it.
POISSON_VORONOI_DIAMETER_CV = 0.5 / np.sqrt(3.57)

RNG_SEED = 20260803


def _diam_cv(labels: np.ndarray, pixel_area_m2: float) -> tuple[float, float, int]:
    """Diameter CV, mean diameter and count from a label array."""
    counts = np.bincount(labels.ravel())
    counts = counts[1:]
    counts = counts[counts > 0]
    if counts.size < 3:
        return float("nan"), float("nan"), int(counts.size)
    area = counts * pixel_area_m2
    diam = 2.0 * np.sqrt(area / np.pi)
    return float(diam.std() / diam.mean()), float(diam.mean()), int(diam.size)


def _tessellate(
    surface: np.ndarray, valid: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> np.ndarray:
    """Watershed ``surface`` from the given seeds, restricted to ``valid``."""
    markers = np.zeros(surface.shape, dtype="int32")
    markers[rows, cols] = np.arange(1, rows.size + 1)
    inv = np.where(valid, -surface, np.inf).astype("float32")
    return watershed(inv, markers=markers, mask=valid)


def _random_seeds(valid: np.ndarray, n: int, rng: np.random.Generator):
    """``n`` seeds placed at random inside the canopy mask."""
    idx = np.flatnonzero(valid.ravel())
    pick = rng.choice(idx, size=min(n, idx.size), replace=False)
    return np.unravel_index(pick, valid.shape)


def _hardcore_seeds(
    valid: np.ndarray, n: int, min_sep_px: float, rng: np.random.Generator
):
    """Random seeds thinned so none lie closer than ``min_sep_px``.

    A hard-core pattern is the fair geometric comparison, because the real
    detector also cannot place two tree tops arbitrarily close together.
    """
    from scipy.spatial import cKDTree

    rows, cols = _random_seeds(valid, min(n * 25, int(valid.sum())), rng)
    pts = np.column_stack([rows, cols]).astype("float64")
    order = rng.permutation(len(pts))
    pts = pts[order]
    kept: list[np.ndarray] = []
    tree = None
    for p in pts:
        if kept:
            if tree is None or len(kept) % 64 == 0:
                tree = cKDTree(np.array(kept))
            if tree.query(p)[0] < min_sep_px:
                continue
            if min(np.hypot(*(np.array(kept) - p).T)) < min_sep_px:
                continue
        kept.append(p)
        if len(kept) >= n:
            break
    arr = np.array(kept, dtype=int)
    return arr[:, 0], arr[:, 1]


def _phase_randomise(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomise the phase of a surface, preserving its power spectrum.

    The height histogram is then restored by rank matching, so the result has
    the same heights in the same proportions but a scrambled arrangement.
    """
    filled = np.nan_to_num(arr, nan=float(np.nanmean(arr)))
    spectrum = np.fft.fft2(filled)
    phase = rng.uniform(0, 2 * np.pi, size=spectrum.shape)
    shuffled = np.real(np.fft.ifft2(np.abs(spectrum) * np.exp(1j * phase)))
    # Rank match back onto the original height distribution.
    order = np.argsort(shuffled.ravel())
    donor = np.sort(filled.ravel())
    out = np.empty(shuffled.size, dtype="float64")
    out[order] = donor
    return out.reshape(arr.shape).astype("float32")


def run_tile(chm_path: Path, bounds, params: SegmentationParams, tile_id: str):
    """Every null for one tile, plus the real result."""
    rng = np.random.default_rng(RNG_SEED)
    arr, transform, _ = read_window(chm_path, bounds)
    px = float(abs(transform.a))
    pixel_area = px * px

    kernel = smooth_kernel_px(params.smooth_radius_m, px)
    smoothed = smooth_chm(arr, kernel=kernel)
    masked = mask_below(smoothed, threshold_m=CANOPY_FLOOR_M)
    valid = ~np.isnan(masked)
    if valid.sum() < 1000:
        return []

    rows, cols = locate_seeds(masked, valid, px, params.window)
    n = int(rows.size)
    if n < 20:
        return []

    # Minimum separation actually achieved, so the hard-core null matches the
    # real pattern rather than a nominal setting.
    from scipy.spatial import cKDTree

    pts = np.column_stack([rows, cols]).astype("float64")
    nn = cKDTree(pts).query(pts, k=2)[0][:, 1]
    min_sep_px = float(np.percentile(nn, 5))

    out = []

    def record(name, labels, note=""):
        cv, mean, count = _diam_cv(labels, pixel_area)
        out.append(
            {
                "tile_id": tile_id,
                "model": name,
                "diam_cv": cv,
                "diam_mean": mean,
                "n_crowns": count,
                "note": note,
            }
        )

    record("real seeds, real CHM", _tessellate(masked, valid, rows, cols))
    flat = np.zeros_like(masked)
    record(
        "real seeds, flat surface",
        _tessellate(flat, valid, rows, cols),
        "pure Voronoi on the observed point pattern",
    )
    r_rows, r_cols = _random_seeds(valid, n, rng)
    record(
        "random seeds, flat surface",
        _tessellate(flat, valid, r_rows, r_cols),
        "Poisson-Voronoi at matched density",
    )
    h_rows, h_cols = _hardcore_seeds(valid, n, min_sep_px, rng)
    record(
        "hard-core seeds, flat surface",
        _tessellate(flat, valid, h_rows, h_cols),
        f"matched density and {min_sep_px * px:.1f} m minimum separation",
    )
    record(
        "real seeds, scrambled CHM",
        _tessellate(_phase_randomise(masked, rng), valid, rows, cols),
        "same heights, destroyed arrangement",
    )

    # And the same set once more with the crown extent bound switched on, to
    # ask whether a physical stopping rule recovers any real signal.
    if params.th_cr > 0:
        bounded = _bounded_labels(masked, valid, rows, cols, params.th_cr)
        record(
            "real seeds, real CHM, bounded",
            bounded,
            f"th_cr={params.th_cr:g}",
        )
        flat_bounded = _bounded_labels(flat, valid, rows, cols, params.th_cr)
        record("real seeds, flat surface, bounded", flat_bounded, "th_cr on a flat CHM")
    return out


def _bounded_labels(surface, valid, rows, cols, th_cr):
    """Tessellate then trim each cell to ``th_cr`` of its own seed height."""
    labels = _tessellate(surface, valid, rows, cols)
    apex = surface[rows, cols]
    apex_by_label = np.concatenate([[np.inf], apex])
    keep = (labels > 0) & (surface >= th_cr * apex_by_label[labels])
    trimmed = np.where(keep, labels, 0)
    comp, _ = ndi.label(keep)  # type: ignore[misc]
    n_labels = int(labels.max())
    pair = np.asarray(comp).astype(np.int64) * (n_labels + 1) + trimmed
    seed_pair = pair[rows, cols]
    seed_pair = seed_pair[trimmed[rows, cols] > 0]
    return np.where(np.isin(pair, seed_pair), trimmed, 0)


def main() -> int:
    """Run the nulls on every tuning tile and both transfer sites."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--th-cr", type=float, default=0.55)
    parser.add_argument("--window", default="popescu_linear")
    parser.add_argument("--smooth-radius-m", type=float, default=0.6)
    args = parser.parse_args()

    params = SegmentationParams(
        smooth_radius_m=args.smooth_radius_m,
        window=WINDOW_FUNCTIONS[args.window],
        th_cr=args.th_cr,
        max_crown_radius_m=None,
    )
    logger.info("Nulls under %s", params.describe())

    tiles = load_tiles()
    chm_cache = {"ElkinsvilleNE": elkinsville_chm(REFERENCE_YEAR)}
    for site in transfer_sites():
        chm_cache[site.name] = site.chm

    rows = []
    for _, tile in tiles.iterrows():
        chm = chm_cache.get(tile["site"])
        if chm is None:
            continue
        bounds = (tile["minx"], tile["miny"], tile["maxx"], tile["maxy"])
        found = run_tile(chm, bounds, params, tile["tile_id"])
        for r in found:
            r["site"] = tile["site"]
            r["stratum"] = tile["stratum"]
        rows.extend(found)
        logger.info("%s: %d models", tile["tile_id"], len(found))

    frame = pd.DataFrame(rows)
    write_table(frame, "nulls.parquet")

    summary = (
        frame.groupby("model")["diam_cv"]
        .agg(["mean", "std", "min", "max", "count"])
        .sort_values("mean")
    )
    print("\nCrown diameter CV by model")
    print(summary.to_string(float_format="%.4f"))
    print(
        f"\nPoisson-Voronoi theoretical diameter CV: "
        f"{POISSON_VORONOI_DIAMETER_CV:.4f}"
    )
    spread = summary["mean"].max() - summary["mean"].min()
    print(
        f"Spread between the most and least informative model: {spread:.4f}\n"
        "A spread of the same order as the measurement noise means the metric "
        "is reporting tessellation geometry rather than forest structure."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
