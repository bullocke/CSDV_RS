"""csdv_core.segmentation.lidr_bridge — bridge to the R lidR pipeline.

Runs the packaged ``crown_segmentation.R`` through ``Rscript``. This is the
independent reference implementation, not the production path. Production
segmentation runs in :mod:`csdv_core.segmentation.chm_watershed`, which keeps
the pipeline free of an R dependency.

The script takes named ``--key=value`` arguments and every length is in metres.
:func:`run_lidr_segmentation` accepts a
:class:`~csdv_core.segmentation.params.SegmentationParams` and translates it,
so a comparison between the two engines varies the algorithm and nothing else.
That matters more than it sounds: lidR's ``max_cr`` is in pixels, so the same
nominal setting is a 10 m crown radius on a 1 m CHM and 6 m on a 0.6 m CHM. The
R script now converts from metres itself.

``rpy2`` is deliberately unused. The subprocess path avoids the ABI coupling
and matches how the CHPC jobs invoke R.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from importlib import resources
from pathlib import Path

from csdv_core.segmentation.params import DEFAULT_PARAMS, SegmentationParams

logger = logging.getLogger(__name__)

__all__ = ["run_lidr_segmentation", "lidr_available", "parse_r_timings"]


def _rscript_path() -> str:
    """Return ``Rscript`` if on PATH, else raise."""
    path = shutil.which("Rscript")
    if path is None:
        raise FileNotFoundError(
            "Rscript not found on PATH. Install R and the lidR package, "
            "or use csdv_core.segmentation.chm_watershed instead."
        )
    return path


def lidr_available() -> bool:
    """True when Rscript and lidR can both be reached.

    Tests skip on this rather than failing, because R is not declared in
    ``environment.yml`` and is not needed for the production path.
    """
    if shutil.which("Rscript") is None:
        return False
    probe = subprocess.run(
        [_rscript_path(), "-e", "library(lidR)"],
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def _packaged_script() -> Path:
    """Return the path to the packaged ``crown_segmentation.R``."""
    ref = resources.files("csdv_core.segmentation").joinpath(
        "scripts/crown_segmentation.R"
    )
    return Path(str(ref))


def run_lidr_segmentation(
    chm_path: Path | str,
    out_crowns: Path | str,
    *,
    params: SegmentationParams = DEFAULT_PARAMS,
    scale_factor: float = 1.0,
    th_seed: float = 0.45,
    out_cv_raster: Path | str | None = None,
    cv_grid_m: float = 50.0,
    cv_min_crowns: int = 3,
    timings: dict[str, float] | None = None,
) -> Path:
    """Run the packaged lidR crown segmentation.

    Args:
        chm_path: Input CHM GeoTIFF.
        out_crowns: Output GeoPackage for crown polygons.
        params: The same parameter object the Python engine takes. Window
            coefficients, smoothing radius, canopy floor, ``th_cr`` and the
            crown radius ceiling are all forwarded, so the two engines differ
            only in how they grow a crown from a tree top.
        scale_factor: Multiplier applied to raw CHM values. Use ``0.01`` for
            the uint16 NAIP-CHM product and ``1.0`` for float metre CHMs.
        th_seed: ``dalponte2016`` growth rule. A pixel joins a crown only while
            it is above ``th_seed`` times the crown's seed height. The Python
            engine has no equivalent. The default is lidR's own.
        out_cv_raster: Optional crown-diameter CV raster. Skipped when None.
        cv_grid_m: Grid size for that raster.
        cv_min_crowns: Minimum crowns per cell before a CV is written.
        timings: Optional dict that receives wall seconds per R step (``load``,
            ``smooth``, ``mask``, ``seeds``, ``grow``, ``vectorize``,
            ``write``) and ``peak_rss_mb``, the R process's peak resident
            memory.

    Returns:
        ``Path(out_crowns)``.
    """
    chm_path = Path(chm_path).resolve()
    out_crowns = Path(out_crowns).resolve()
    if not chm_path.exists():
        raise FileNotFoundError(f"CHM not found: {chm_path}")
    out_crowns.parent.mkdir(parents=True, exist_ok=True)

    script = _packaged_script()
    if not script.exists():
        raise FileNotFoundError(
            f"Packaged R script not found at {script}. "
            "Reinstall the package or verify package_data."
        )

    window = params.window
    # lidR has no "no ceiling" option, so an unbounded Python run maps to a
    # ceiling far beyond any real crown rather than to a missing argument.
    max_radius_m = (
        1000.0 if params.max_crown_radius_m is None else params.max_crown_radius_m
    )
    cmd = [
        _rscript_path(),
        str(script),
        f"--chm={chm_path}",
        f"--out-crowns={out_crowns}",
        f"--ws-a={window.a:g}",
        f"--ws-b={window.b:g}",
        f"--ws-c={window.c:g}",
        f"--ws-lo={window.lo:g}",
        f"--ws-hi={window.hi:g}",
        f"--smooth-radius-m={params.smooth_radius_m:g}",
        f"--min-height-m={params.min_height_m:g}",
        f"--th-seed={th_seed:g}",
        f"--th-cr={params.th_cr:g}",
        f"--max-crown-radius-m={max_radius_m:g}",
        f"--min-crown-area-m2={params.min_crown_area_m2:g}",
        f"--scale-factor={scale_factor:g}",
    ]
    if out_cv_raster is not None:
        out_cv = Path(out_cv_raster).resolve()
        out_cv.parent.mkdir(parents=True, exist_ok=True)
        cmd += [
            f"--out-cv-raster={out_cv}",
            f"--cv-grid-m={cv_grid_m:g}",
            f"--cv-min-crowns={int(cv_min_crowns)}",
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
    if timings is not None:
        timings.update(parse_r_timings(result.stdout))
    return out_crowns


def parse_r_timings(stdout: str) -> dict[str, float]:
    """Read the ``TIMING`` and ``PEAK_RSS_KB`` lines the R script prints.

    Returns:
        Seconds per step, plus ``peak_rss_mb`` when R could read it.
    """
    out: dict[str, float] = {}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "TIMING":
            out[parts[1]] = out.get(parts[1], 0.0) + float(parts[2])
        elif len(parts) == 2 and parts[0] == "PEAK_RSS_KB" and parts[1] != "NA":
            out["peak_rss_mb"] = float(parts[1]) / 1024.0
    return out
