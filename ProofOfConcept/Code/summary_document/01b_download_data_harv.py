"""
01b_download_data_harv.py — Download NAIP, NAIP-CHM, and NEON CHM data for
Harvard Forest (HARV).

Mirrors 01_download_data.py for HARV. Unlike the SCBI script, this also
downloads the NEON CHM, which was pre-fetched separately for SCBI.

The HARV bounding box in download_gee.py is centered on the target area at
lon=-72.234, lat=42.438 and covers roughly 2 km × 1.5 km (EPSG:4326:
-72.244, 42.432, -72.224, 42.444).

Outputs
-------
Data/NAIP/Imagery/NAIP_HARV_2022*.tif           NAIP RGBN mosaic (1m, EPSG:5070)
Data/NAIP/National_CHM/NAIPCHM_HARV_2022*.tif  NAIP-CHM (UInt16, divide by 100 for meters)
Data/NEON/CHM/NEON_CHM_HARV_Subset_2023*.tif   NEON ALS CHM (1m, EPSG:5070)

Note on the NEON CHM download
------------------------------
The script uses the full CHM ImageCollection
(projects/neon-prod-earthengine/assets/CHM/001) filtered to the HARV bbox and
date range, then mosaicked. If you know the specific flight asset path (e.g.,
projects/neon-prod-earthengine/assets/CHM/001/2023_HARV_7), substitute it as
the --collection argument in the run_download call below for a cleaner download.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOWNLOADER = PROJECT_ROOT / "ProofOfConcept" / "Code" / "download_gee.py"

NAIP_OUT = PROJECT_ROOT / "ProofOfConcept" / "Data" / "NAIP" / "Imagery"
NAIP_CHM_OUT = PROJECT_ROOT / "ProofOfConcept" / "Data" / "NAIP" / "National_CHM"
NEON_CHM_OUT = PROJECT_ROOT / "ProofOfConcept" / "Data" / "NEON" / "CHM"


def run_download(args: list[str]) -> None:
    """Invoke download_gee.py with the given argument list."""
    cmd = [sys.executable, str(DOWNLOADER)] + args
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Download failed (exit {result.returncode})")


def verify_output(path: Path, label: str) -> None:
    """Check that a downloaded file exists (wxee may append a timestamp suffix)."""
    matches = list(path.parent.glob(f"{path.stem}*.tif"))
    if not matches:
        raise FileNotFoundError(
            f"{label}: no .tif matching {path.stem}* in {path.parent}"
        )
    for f in matches:
        size_mb = f.stat().st_size / 1e6
        logger.info("%s: %s (%.1f MB)", label, f.name, size_mb)
        if size_mb < 0.01:
            logger.warning("%s appears suspiciously small (%.2f MB)", label, size_mb)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=== Harvard Forest (HARV) Data Download ===")
    logger.info("Project root: %s", PROJECT_ROOT)

    # 1. NAIP RGBN
    logger.info("--- Downloading NAIP RGBN (HARV) ---")
    run_download([
        "--collection", "USDA/NAIP/DOQQ",
        "--site", "harv",
        "--start-date", "2021-01-01",
        "--end-date", "2023-12-31",
        "--bands", "R,G,B,N",
        "--mosaic",
        "--scale", "1",
        "--crs", "EPSG:5070",
        "--out-dir", str(NAIP_OUT),
        "--description", "NAIP_HARV_2022",
    ])
    verify_output(NAIP_OUT / "NAIP_HARV_2022.tif", "NAIP RGBN HARV")

    # 2. NAIP CHM (Morford et al. 2025)
    logger.info("--- Downloading NAIP-CHM (HARV) ---")
    run_download([
        "--collection", "projects/naip-chm/assets/conus-structure-model",
        "--site", "harv",
        "--start-date", "2021-01-01",
        "--end-date", "2023-12-31",
        "--mosaic",
        "--scale", "1",
        "--crs", "EPSG:5070",
        "--out-dir", str(NAIP_CHM_OUT),
        "--description", "NAIPCHM_HARV_2022",
    ])
    verify_output(NAIP_CHM_OUT / "NAIPCHM_HARV_2022.tif", "NAIP-CHM HARV")

    # 3. NEON ALS CHM
    # Uses the full ImageCollection filtered to the HARV bbox and date range.
    # To download a specific flight year, replace the collection argument with
    # the full asset path, e.g.:
    #   projects/neon-prod-earthengine/assets/CHM/001/2023_HARV_7
    logger.info("--- Downloading NEON ALS CHM (HARV) ---")
    run_download([
        "--collection", "projects/neon-prod-earthengine/assets/CHM/001",
        "--site", "harv",
        "--start-date", "2022-01-01",
        "--end-date", "2024-12-31",
        "--mosaic",
        "--scale", "1",
        "--crs", "EPSG:5070",
        "--out-dir", str(NEON_CHM_OUT),
        "--description", "NEON_CHM_HARV_Subset_2023",
    ])
    verify_output(NEON_CHM_OUT / "NEON_CHM_HARV_Subset_2023.tif", "NEON CHM HARV")

    logger.info("=== All downloads complete ===")


if __name__ == "__main__":
    main()
