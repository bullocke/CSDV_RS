"""
01_download_data.py — Download NAIP and NAIP-CHM data for the SCBI summary figures.

Downloads three data products for the SCBI region defined in the proof-of-concept
plan. Calls download_gee.py as a subprocess so the general-purpose downloader
handles all GEE interaction.

The NEON CHM was already downloaded via the test script and is copied to the
canonical data location by the folder setup step (see plan).

Outputs
-------
Data/NAIP/Imagery/NAIP_SCBI_2022.tif         NAIP RGBN mosaic
Data/NAIP/National_CHM/NAIPCHM_SCBI_2022.tif NAIP-CHM mosaic (UInt16, divide by 100 for meters)
Data/NEON/CHM/NEON_CHM_SCBI_Subset_2023.tif  Already present (copied during setup)
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths relative to project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOWNLOADER = PROJECT_ROOT / "ProofOfConcept" / "Code" / "download_gee.py"

NAIP_OUT = PROJECT_ROOT / "ProofOfConcept" / "Data" / "NAIP" / "Imagery"
NAIP_CHM_OUT = PROJECT_ROOT / "ProofOfConcept" / "Data" / "NAIP" / "National_CHM"
NEON_CHM_OUT = PROJECT_ROOT / "ProofOfConcept" / "Data" / "NEON" / "CHM"


def run_download(args: list[str]) -> None:
    """Invoke download_gee.py with the given argument list and log output."""
    cmd = [sys.executable, str(DOWNLOADER)] + args
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Download failed (exit {result.returncode})")


def verify_output(path: Path, label: str) -> None:
    """Check that a downloaded file exists and is non-empty."""
    # wxee appends a timestamp to the description; find the file by prefix
    matches = list(path.parent.glob(f"{path.stem}*.tif"))
    if not matches:
        raise FileNotFoundError(f"{label}: no .tif found matching {path.stem}* in {path.parent}")
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

    logger.info("=== Summary Document Data Download ===")
    logger.info("Project root: %s", PROJECT_ROOT)

    # ------------------------------------------------------------------
    # 1. NAIP RGBN mosaic for SCBI (most recent cycle, 2021-2023)
    # ------------------------------------------------------------------
    logger.info("--- Downloading NAIP RGBN ---")
    run_download([
        "--collection", "USDA/NAIP/DOQQ",
        "--site", "scbi",
        "--start-date", "2021-01-01",
        "--end-date", "2023-12-31",
        "--bands", "R,G,B,N",
        "--mosaic",
        "--scale", "1",
        "--crs", "EPSG:5070",
        "--out-dir", str(NAIP_OUT),
        "--description", "NAIP_SCBI_2022",
    ])
    verify_output(NAIP_OUT / "NAIP_SCBI_2022.tif", "NAIP RGBN")

    # ------------------------------------------------------------------
    # 2. NAIP-CHM (Morford 2025) mosaic for SCBI
    #    Values are UInt16; divide by 100 to get height in meters.
    # ------------------------------------------------------------------
    logger.info("--- Downloading NAIP-CHM ---")
    run_download([
        "--collection", "projects/naip-chm/assets/conus-structure-model",
        "--site", "scbi",
        "--start-date", "2021-01-01",
        "--end-date", "2023-12-31",
        "--mosaic",
        "--scale", "1",
        "--crs", "EPSG:5070",
        "--out-dir", str(NAIP_CHM_OUT),
        "--description", "NAIPCHM_SCBI_2022",
    ])
    verify_output(NAIP_CHM_OUT / "NAIPCHM_SCBI_2022.tif", "NAIP-CHM")

    # ------------------------------------------------------------------
    # 3. NEON CHM — already downloaded; just confirm it's present
    # ------------------------------------------------------------------
    logger.info("--- Verifying NEON CHM ---")
    neon_chm = NEON_CHM_OUT / "NEON_CHM_SCBI_Subset_2023.tif"
    if not neon_chm.exists():
        logger.error(
            "NEON CHM not found at %s. "
            "Run the setup step (copy from Code/testing/wxee_download/).",
            neon_chm,
        )
        sys.exit(1)
    size_mb = neon_chm.stat().st_size / 1e6
    logger.info("NEON CHM: %s (%.1f MB)", neon_chm.name, size_mb)

    logger.info("=== All downloads complete ===")


if __name__ == "__main__":
    main()
