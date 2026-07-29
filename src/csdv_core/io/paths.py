"""csdv_core.io.paths — project path resolution.

Three roots drive on-disk layout and are configured via environment
variables so the same code runs locally, on CHPC, and on Colab. See
``docs/data_layout.md`` for the full directory tree.

Environment variables (each falls back to a repo-relative default):

    CSDV_DATA_ROOT      inputs (NAIP, NEON, topo, soils, CHM weights)
    CSDV_RESULTS_ROOT   outputs (metrics, stages, trajectories, figures)
    CSDV_CACHE_ROOT     intermediate caches
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    # src/csdv_core/io/paths.py -> repo root is parents[3]
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved on-disk roots for the CSDV pipeline."""

    data_root: Path
    results_root: Path
    cache_root: Path

    # ---- input subpaths -------------------------------------------------

    def naip_dir(self, site: str, year: int) -> Path:
        return self.data_root / "naip" / site / str(year)

    def naip_chm_dir(self, site: str, year: int) -> Path:
        return self.data_root / "naip_chm" / site / str(year)

    def neon_chm_dir(self, site: str, year: int) -> Path:
        return self.data_root / "neon" / site / "als_chm" / str(year)

    def neon_lidar_dir(self, site: str, year: int) -> Path:
        return self.data_root / "neon" / site / "lidar" / str(year)

    def neon_field_dir(self, site: str, year: int) -> Path:
        return self.data_root / "neon" / site / "field" / str(year)

    def topo_dir(self, site: str) -> Path:
        return self.data_root / "topo" / site

    def soils_dir(self, site: str) -> Path:
        return self.data_root / "soils" / site

    def chm_model_dir(self) -> Path:
        return self.data_root / "chm_model"

    def satellite_dir(self, site: str) -> Path:
        """Per-stand satellite observation cache under ``data_root/satellite/<site>/``.

        This sits under ``data_root`` rather than ``results_root`` because it is
        a cache of an external archive, not something the pipeline derived. The
        per-year metrics computed from it are a result and live under
        :meth:`stands_dir`.
        """
        return self.data_root / "satellite" / site

    # ---- output subpaths ------------------------------------------------

    def metrics_dir(self, site: str, year: int, window_m: int | float) -> Path:
        return self.results_root / "metrics" / site / str(year) / f"{int(window_m)}m"

    def crowns_dir(self, site: str, year: int) -> Path:
        """Per-year crown vector outputs under ``results_root/crowns/<site>/<year>/``."""
        return self.results_root / "crowns" / site / str(year)

    def stratification_dir(self, site: str, window_m: int | float) -> Path:
        """Per-window stratification outputs (site_type, match_score)."""
        return self.results_root / "stratification" / site / f"{int(window_m)}m"

    def stages_dir(
        self, site: str, year: int, window_m: int | float | None = None
    ) -> Path:
        """Per-window stage rasters under ``results_root/stages/<site>/<year>/``.

        If ``window_m`` is given, returns the ``<window_m>m`` subdirectory.
        Without it, returns the year-level directory (for backward
        compatibility with code that builds its own subpaths).
        """
        base = self.results_root / "stages" / site / str(year)
        if window_m is None:
            return base
        return base / f"{int(window_m)}m"

    def trajectories_dir(self, site: str, window_m: int | float | None = None) -> Path:
        """Per-window trajectory rasters under ``results_root/trajectories/<site>/``.

        If ``window_m`` is given, returns the ``<window_m>m`` subdirectory.
        Without it, returns the site-level directory (back-compat with code
        that builds its own subpaths).
        """
        base = self.results_root / "trajectories" / site
        if window_m is None:
            return base
        return base / f"{int(window_m)}m"

    def stands_dir(self, site: str) -> Path:
        """Per-stand derived products under ``results_root/stands/<site>/``.

        Holds the stand metric table, the per-year satellite metrics, the stage
        and trajectory tables, and the cached crown segmentations. This is the
        polygon-based counterpart to :meth:`metrics_dir`, which is windowed.
        """
        return self.results_root / "stands" / site

    def figures_dir(self) -> Path:
        return self.results_root / "figures"


def project_paths() -> ProjectPaths:
    """Resolve project paths from environment, with repo-relative defaults."""
    repo = _repo_root()
    data = Path(os.environ.get("CSDV_DATA_ROOT", repo / "data"))
    results = Path(os.environ.get("CSDV_RESULTS_ROOT", repo / "results"))
    cache = Path(os.environ.get("CSDV_CACHE_ROOT", repo / ".cache"))
    logger.debug("project_paths: data=%s results=%s cache=%s", data, results, cache)
    return ProjectPaths(data_root=data, results_root=results, cache_root=cache)
