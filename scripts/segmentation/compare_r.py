"""Compare the Python engine against lidR on matched parameters.

This is the check that the Python port is a port. Both engines take the same
window function, the same smoothing radius in metres, the same canopy floor,
the same ``th_cr`` and the same crown radius ceiling in metres, so a difference
between them is a difference in how a crown grows from a tree top and nothing
else.

Two comparisons, because they isolate different things:

    tree tops
        Depends only on the local maximum rule. This is where the old port was
        wrong. ``skimage.peak_local_max(min_distance=d)`` takes a minimum
        separation while lidR's ``lmf(ws=...)`` takes a window diameter, so
        passing the window straight through spread tree tops twice as far apart
        as the equation intended.

    crown widths
        Depends on the growing rule as well. Python watersheds and then trims
        to ``th_cr``; lidR grows regions outward under Dalponte and Coomes
        (2016), which also applies ``th_seed``. Some disagreement here is
        expected and is not a defect.

R, lidR, terra and sf are installed in the project environment but are not
declared in ``environment.yml``. The comparison is therefore reproducible on
this machine and should be treated as a one-off elsewhere, with versions
recorded in the report.

Run:
    .micromamba/envs/CSDV/bin/python scripts/segmentation/compare_r.py
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
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
    segment_crowns,
    smooth_kernel_px,
)
from csdv_core.segmentation.lidr_bridge import (  # noqa: E402
    lidr_available,
    run_lidr_segmentation,
)
from csdv_core.segmentation.params import (  # noqa: E402
    WINDOW_FUNCTIONS,
    SegmentationParams,
)

logger = logging.getLogger("compare_r")


def write_tile(arr, transform, crs, path: Path) -> Path:
    """Write a tile so R can read the identical pixels Python used."""
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": -9999.0,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.nan_to_num(arr, nan=-9999.0).astype("float32"), 1)
    return path


def r_versions() -> dict[str, str]:
    """Record what produced the R side, for the report."""
    import subprocess

    out = subprocess.run(
        [
            shutil.which("Rscript") or "Rscript",
            "-e",
            'cat(R.version.string, "|",'
            ' as.character(packageVersion("lidR")), "|",'
            ' as.character(packageVersion("terra")), "|",'
            ' as.character(packageVersion("sf")))',
        ],
        capture_output=True,
        text=True,
    )
    parts = [p.strip() for p in out.stdout.split("|")]
    keys = ["R", "lidR", "terra", "sf"]
    return dict(zip(keys, parts + [""] * 4, strict=False))


def main() -> int:
    """Run both engines on the same tiles and compare."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", default="popescu_linear")
    parser.add_argument("--smooth-radius-m", type=float, default=0.6)
    parser.add_argument("--th-cr", type=float, default=0.55)
    parser.add_argument("--max-crown-radius-m", type=float, default=8.0)
    parser.add_argument("--max-tiles", type=int, default=8)
    args = parser.parse_args()

    if not lidr_available():
        logger.error("Rscript or lidR is unavailable, so the comparison cannot run")
        return 1
    versions = r_versions()
    logger.info("R side: %s", versions)

    params = SegmentationParams(
        smooth_radius_m=args.smooth_radius_m,
        window=WINDOW_FUNCTIONS[args.window],
        th_cr=args.th_cr,
        max_crown_radius_m=args.max_crown_radius_m,
    )
    logger.info("Matched parameters: %s", params.describe())

    tiles = load_tiles()
    # A spread of conditions and both transfer sites, capped for runtime.
    tiles = (
        tiles.sort_values(["site", "stratum"])
        .groupby("stratum", group_keys=False)
        .head(2)
        .head(args.max_tiles)
    )
    chm_cache = {"ElkinsvilleNE": elkinsville_chm(REFERENCE_YEAR)}
    for site in transfer_sites():
        chm_cache[site.name] = site.chm

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for _, tile in tiles.iterrows():
            chm_path = chm_cache.get(tile["site"])
            if chm_path is None:
                continue
            bounds = (tile["minx"], tile["miny"], tile["maxx"], tile["maxy"])
            arr, transform, crs = read_window(chm_path, bounds)
            px = float(abs(transform.a))
            tile_tif = write_tile(
                arr, transform, crs, tmpdir / f"{tile['tile_id']}.tif"
            )

            py = segment_crowns(arr, transform, crs, params=params)
            kernel = smooth_kernel_px(params.smooth_radius_m, px)
            masked = mask_below(smooth_chm(arr, kernel=kernel), params.min_height_m)
            valid = ~np.isnan(masked)
            py_rows, _ = locate_seeds(masked, valid, px, params.window)

            out_gpkg = tmpdir / f"{tile['tile_id']}_R.gpkg"
            try:
                run_lidr_segmentation(tile_tif, out_gpkg, params=params)
                r = gpd.read_file(out_gpkg)
            except Exception as exc:  # noqa: BLE001
                logger.warning("lidR failed on %s: %s", tile["tile_id"], exc)
                continue

            pyd = py["crown_diam_m"].to_numpy() if len(py) else np.array([np.nan])
            rd = r["crown_diam_m"].to_numpy() if len(r) else np.array([np.nan])
            rows.append(
                {
                    "tile_id": tile["tile_id"],
                    "site": tile["site"],
                    "stratum": tile["stratum"],
                    "pixel_size_m": px,
                    "py_tops": int(py_rows.size),
                    "r_crowns": int(len(r)),
                    "py_crowns": int(len(py)),
                    "top_ratio": float(py_rows.size / max(len(r), 1)),
                    "crown_ratio": float(len(py) / max(len(r), 1)),
                    "py_diam_mean": float(np.nanmean(pyd)),
                    "r_diam_mean": float(np.nanmean(rd)),
                    "py_diam_p50": float(np.nanpercentile(pyd, 50)),
                    "r_diam_p50": float(np.nanpercentile(rd, 50)),
                    "py_diam_p90": float(np.nanpercentile(pyd, 90)),
                    "r_diam_p90": float(np.nanpercentile(rd, 90)),
                    "py_diam_cv": float(np.nanstd(pyd) / np.nanmean(pyd)),
                    "r_diam_cv": float(np.nanstd(rd) / np.nanmean(rd)),
                }
            )
            logger.info(
                "%s: tops %d vs %d crowns, ratio %.3f",
                tile["tile_id"],
                py_rows.size,
                len(r),
                rows[-1]["top_ratio"],
            )

    frame = pd.DataFrame(rows)
    for key, value in versions.items():
        frame[f"version_{key}"] = value
    write_table(frame, "compare_r.csv")

    print("\nPython against lidR on matched parameters")
    print(
        frame[
            [
                "tile_id",
                "py_tops",
                "r_crowns",
                "top_ratio",
                "py_diam_mean",
                "r_diam_mean",
                "py_diam_cv",
                "r_diam_cv",
            ]
        ].to_string(index=False, float_format="%.3f")
    )
    if len(frame):
        print(
            f"\nTree-top count ratio: mean {frame['top_ratio'].mean():.3f}, "
            f"range {frame['top_ratio'].min():.3f} to {frame['top_ratio'].max():.3f}"
        )
        print(
            f"Mean crown diameter: Python {frame['py_diam_mean'].mean():.2f} m, "
            f"lidR {frame['r_diam_mean'].mean():.2f} m"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
