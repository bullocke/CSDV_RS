"""csdv_core.segmentation.lidr_bridge — Bridge to the R lidR pipeline.

Calls the packaged ``crown_segmentation.R`` script (copied from the legacy
PoC) via ``subprocess`` to ``Rscript``. The script signature is:

    Rscript crown_segmentation.R <chm_path> <out_crowns_gpkg> <out_cv_tif> [scale_factor]

If ``rpy2`` is installed, callers can substitute their own R execution
strategy; the subprocess path is the default because it avoids the rpy2
ABI dance and matches the CHPC workflow.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from importlib import resources
from pathlib import Path

logger = logging.getLogger(__name__)


def _rscript_path() -> str:
    """Return ``Rscript`` if on PATH, else raise."""
    path = shutil.which("Rscript")
    if path is None:
        raise FileNotFoundError(
            "Rscript not found on PATH. Install R + the lidR package, "
            "or use csdv_core.segmentation.chm_watershed instead."
        )
    return path


def _packaged_script() -> Path:
    """Return the path to the packaged ``crown_segmentation.R``."""
    ref = resources.files("csdv_core.segmentation").joinpath(
        "scripts/crown_segmentation.R"
    )
    return Path(str(ref))


def run_lidr_segmentation(
    chm_path: Path | str,
    out_crowns: Path | str,
    out_cv_raster: Path | str,
    *,
    scale_factor: float = 1.0,
) -> Path:
    """Run the packaged lidR crown segmentation script.

    Args:
        chm_path: Input CHM GeoTIFF path.
        out_crowns: Output GeoPackage path for crown polygons.
        out_cv_raster: Output GeoTIFF path for the per-50 m crown CV raster.
        scale_factor: Multiplier applied to raw CHM values. Use ``0.01``
            for the uint16 NAIP-CHM product, ``1.0`` for NEON-style float
            metre CHMs.

    Returns:
        ``Path(out_crowns)``.
    """
    chm_path = Path(chm_path).resolve()
    out_crowns = Path(out_crowns).resolve()
    out_cv_raster = Path(out_cv_raster).resolve()
    if not chm_path.exists():
        raise FileNotFoundError(f"CHM not found: {chm_path}")

    out_crowns.parent.mkdir(parents=True, exist_ok=True)
    out_cv_raster.parent.mkdir(parents=True, exist_ok=True)

    script = _packaged_script()
    if not script.exists():
        raise FileNotFoundError(
            f"Packaged R script not found at {script}. "
            "Reinstall the package or verify package_data."
        )

    cmd = [
        _rscript_path(),
        str(script),
        str(chm_path),
        str(out_crowns),
        str(out_cv_raster),
        f"{scale_factor:g}",
    ]
    logger.info("Running lidR segmentation: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        logger.info("lidR stdout:\n%s", result.stdout)
    if result.stderr:
        logger.warning("lidR stderr:\n%s", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"lidR segmentation failed with exit code {result.returncode}"
        )
    return out_crowns


__all__ = ["run_lidr_segmentation"]
