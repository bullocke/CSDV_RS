"""
12a_download_neon_broader.py — Download NEON CHM, NAIP RGBN, and NAIP CHM for a
broader area around each NEON site, enabling sample-point comparisons across a
spatially representative extent.

The broader bounding boxes cover roughly 8–10 km × 5 km, compared to the ~2 km²
subsets already downloaded. This provides enough spatial heterogeneity for
meaningful sample-point statistics.

Outputs
-------
Data/NEON/CHM/NEON_CHM_{SITE}_Broader_2023.tif      NEON ALS CHM
Data/NAIP/Imagery/NAIP_{SITE}_Broader.tif            NAIP RGBN mosaic (1m, EPSG:5070)
Data/NAIP/National_CHM/NAIPCHM_{SITE}_Broader.tif   NAIP CHM (UInt16)

Usage
-----
  python 12a_download_neon_broader.py --site SCBI
  python 12a_download_neon_broader.py --site HARV

Notes
-----
The NEON ALS CHM uses the specific flight asset path when available to avoid
downloading imagery from multiple years. Adjust the collection path below if
a different NEON AOP flight year is needed.

Bounding boxes (WGS84, west south east north)
---------------------------------------------
SCBI broader: -78.21, 38.87, -78.09, 38.92  (~10 km × 5 km)
HARV broader: -72.29, 42.41, -72.17, 42.46  (~9 km × 5 km)
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import click

_HERE = Path(__file__).resolve()
_CODE_DIR = _HERE.parent.parent
sys.path.insert(0, str(_CODE_DIR))

logger = logging.getLogger(__name__)

_POC = _HERE.parents[2]
_DATA = _POC / "Data"
_DOWNLOADER = _CODE_DIR / "download_gee.py"

# Broader bounding boxes — roughly 10 × 5 km, wider than the existing subsets
BROADER_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "SCBI": (-78.21, 38.87, -78.09, 38.92),
    "HARV": (-72.29, 42.41, -72.17, 42.46),
}

# NEON ALS CHM asset paths — specific flight years to avoid multi-year mosaics
NEON_COLLECTIONS: dict[str, str] = {
    "SCBI": "projects/neon-prod-earthengine/assets/CHM/001/2023_SCBI_6",
    "HARV": "projects/neon-prod-earthengine/assets/CHM/001/2023_HARV_7",
}


def run_download(args: list[str]) -> None:
    """Invoke download_gee.py with the given argument list."""
    cmd = [sys.executable, str(_DOWNLOADER)] + args
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Download failed (exit {result.returncode})")


def verify_output(path: Path, label: str) -> None:
    """Check that a downloaded file exists (handles wxee timestamp suffix)."""
    matches = list(path.parent.glob(f"{path.stem}*.tif"))
    if not matches:
        raise FileNotFoundError(
            f"{label}: no .tif matching {path.stem}* in {path.parent}"
        )
    for f in matches:
        size_mb = f.stat().st_size / 1e6
        logger.info("%s: %s (%.1f MB)", label, f.name, size_mb)
        if size_mb < 0.1:
            logger.warning("%s appears suspiciously small (%.2f MB)", label, size_mb)


@click.command()
@click.option("--site", "-s", required=True,
              type=click.Choice(list(BROADER_BBOXES), case_sensitive=False),
              help="Site code: SCBI or HARV")
@click.option("--project", default="dyce-biomass", show_default=True,
              help="Google Earth Engine project ID.")
def main(site: str, project: str) -> None:
    """Download broader NEON/NAIP data for sample-point comparison."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    s = site.upper()
    bbox = BROADER_BBOXES[s]
    bbox_args = [str(v) for v in bbox]
    neon_collection = NEON_COLLECTIONS[s]

    neon_out_dir = _DATA / "NEON" / "CHM"
    naip_out_dir = _DATA / "NAIP" / "Imagery"
    naipchm_out_dir = _DATA / "NAIP" / "National_CHM"

    logger.info("=== Broader download: %s ===", s)
    logger.info("Bbox (WGS84): W=%.3f S=%.3f E=%.3f N=%.3f", *bbox)

    # 1. NEON ALS CHM
    logger.info("--- NEON ALS CHM ---")
    run_download([
        "--collection", neon_collection,
        "--bbox", *bbox_args,
        "--scale", "1",
        "--crs", "EPSG:5070",
        "--out-dir", str(neon_out_dir),
        "--description", f"NEON_CHM_{s}_Broader_2023",
        "--project", project,
    ])
    verify_output(neon_out_dir / f"NEON_CHM_{s}_Broader_2023.tif", f"NEON CHM {s} broader")

    # 2. NAIP RGBN
    logger.info("--- NAIP RGBN ---")
    run_download([
        "--collection", "USDA/NAIP/DOQQ",
        "--bbox", *bbox_args,
        "--start-date", "2021-01-01",
        "--end-date", "2023-12-31",
        "--bands", "R,G,B,N",
        "--mosaic",
        "--scale", "1",
        "--crs", "EPSG:5070",
        "--out-dir", str(naip_out_dir),
        "--description", f"NAIP_{s}_Broader",
        "--project", project,
    ])
    verify_output(naip_out_dir / f"NAIP_{s}_Broader.tif", f"NAIP RGBN {s} broader")

    # 3. NAIP CHM (Morford et al. 2025)
    logger.info("--- NAIP CHM ---")
    run_download([
        "--collection", "projects/naip-chm/assets/conus-structure-model",
        "--bbox", *bbox_args,
        "--start-date", "2021-01-01",
        "--end-date", "2023-12-31",
        "--mosaic",
        "--scale", "1",
        "--crs", "EPSG:5070",
        "--out-dir", str(naipchm_out_dir),
        "--description", f"NAIPCHM_{s}_Broader",
        "--project", project,
    ])
    verify_output(naipchm_out_dir / f"NAIPCHM_{s}_Broader.tif", f"NAIP CHM {s} broader")

    logger.info("=== Broader downloads complete: %s ===", s)


if __name__ == "__main__":
    main()
