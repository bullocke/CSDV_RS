"""csdv_core.chm_inference.conditioning — NAIP-CHM conditioning rasters.

The upstream NAIP-CHM model (Morford et al., 2025) expects 5 conditioning
rasters covering CONUS:

    elevation.tif      3DEP-derived elevation
    climate_pca.tif    Climate PCA components
    soil_pca.tif       SSURGO PCA components
    nlcd.tif           NLCD land cover
    ecoregion.tif      EPA Level-III ecoregion ID

These are large (multi-GB) and are not downloaded automatically. Run
``csdv download conditioning`` to fetch them from the UMT rangeland server.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from csdv_core.io.paths import ProjectPaths, project_paths

logger = logging.getLogger(__name__)

REQUIRED_RASTERS: tuple[str, ...] = (
    "elevation.tif",
    "climate_pca.tif",
    "soil_pca.tif",
    "nlcd.tif",
    "ecoregion.tif",
)

#: HTTP directory holding the CONUS-wide conditioning rasters on the UMT
#: rangeland server. Files there are named exactly as in ``REQUIRED_RASTERS``.
CONDITIONING_BASE_URL = (
    "http://rangeland.ntsg.umt.edu/data/naip-chm/inference-resources/conditioning-data"
)


@dataclass(frozen=True)
class ConditioningPaths:
    """Resolved paths to the 5 conditioning rasters."""

    root: Path
    elevation: Path
    climate_pca: Path
    soil_pca: Path
    nlcd: Path
    ecoregion: Path


def resolve_conditioning_dir(paths: ProjectPaths | None = None) -> Path:
    """Return the on-disk directory for NAIP-CHM conditioning rasters.

    Honors the ``CSDV_CONDITIONING_DIR`` environment variable when set, so the
    rasters can live outside the default project layout (e.g. a shared,
    read-only location). Otherwise falls back to
    ``$CSDV_DATA_ROOT/chm_model/conditioning``.
    """
    override = os.environ.get("CSDV_CONDITIONING_DIR")
    if override:
        return Path(override)
    p = paths or project_paths()
    return p.chm_model_dir() / "conditioning"


def conditioning_paths(paths: ProjectPaths | None = None) -> ConditioningPaths:
    """Build a :class:`ConditioningPaths` for the current project paths."""
    root = resolve_conditioning_dir(paths)
    return ConditioningPaths(
        root=root,
        elevation=root / "elevation.tif",
        climate_pca=root / "climate_pca.tif",
        soil_pca=root / "soil_pca.tif",
        nlcd=root / "nlcd.tif",
        ecoregion=root / "ecoregion.tif",
    )


def validate(paths: ProjectPaths | None = None) -> ConditioningPaths:
    """Validate that all 5 conditioning rasters exist; return their paths.

    Raises:
        FileNotFoundError: If one or more rasters are missing.
    """
    cp = conditioning_paths(paths)
    missing = [name for name in REQUIRED_RASTERS if not (cp.root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Conditioning rasters missing in {cp.root}: {missing}. "
            "Run 'csdv download conditioning' to fetch them."
        )
    logger.info("Conditioning rasters validated at %s", cp.root)
    return cp


__all__ = [
    "CONDITIONING_BASE_URL",
    "ConditioningPaths",
    "REQUIRED_RASTERS",
    "conditioning_paths",
    "resolve_conditioning_dir",
    "validate",
]
