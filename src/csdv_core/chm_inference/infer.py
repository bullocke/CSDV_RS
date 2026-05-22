"""csdv_core.chm_inference.infer — NAIP-CHM inference wrapper.

Wraps the upstream ``smorf-ntsg/naip-chm`` repository via subprocess.
The Phase 3 plan called for a vendored in-process import, but the
upstream project does not expose a stable Python API. We invoke its
``scripts/inference.py`` instead. The ``[chm_inference]`` extra installs
``torch`` only. Set ``NAIPCHM_REPO_DIR`` (or pass ``--repo-dir``) to a
local clone of https://github.com/smorf-ntsg/naip-chm.

The function validates that the NAIP filename contains a ``YYYYMMDD``
token (the upstream model extracts DOY from the filename) and that the
five conditioning rasters are present.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import click

from csdv_core.chm_inference.conditioning import validate as validate_conditioning
from csdv_core.config import load_metrics

logger = logging.getLogger(__name__)

_FILENAME_DATE_RE = re.compile(r"\d{8}")


def _validate_filename(path: Path) -> None:
    if not _FILENAME_DATE_RE.search(path.stem):
        raise ValueError(
            f"NAIP filename {path.name!r} must contain a YYYYMMDD token; "
            "the upstream DOY extractor requires it."
        )


def _resolve_repo_dir(repo_dir: Path | None) -> Path:
    if repo_dir is not None:
        return Path(repo_dir).resolve()
    env = os.environ.get("NAIPCHM_REPO_DIR")
    if env:
        return Path(env).resolve()
    return (Path.home() / "code" / "naip-chm").resolve()


def predict_chm(
    naip_quad: Path | str,
    output_dir: Path | str,
    *,
    repo_dir: Path | None = None,
    model_checkpoint: Path | str = "model/model_20251016.pt",
    config_path: Path | str = "configs/config.yaml",
    conditioning_dir: Path | None = None,
    chip_size: int | None = None,
    chip_overlap: float | None = None,
    dry_run: bool = False,
) -> Path:
    """Run NAIP-CHM inference on a single NAIP AOI GeoTIFF.

    Args:
        naip_quad: Input 4-band NAIP GeoTIFF. Filename must contain a
            ``YYYYMMDD`` token.
        output_dir: Directory to receive the predicted CHM GeoTIFF.
        repo_dir: Path to the cloned ``smorf-ntsg/naip-chm`` repository.
            Defaults to ``$NAIPCHM_REPO_DIR`` then ``~/code/naip-chm``.
        model_checkpoint: Checkpoint path, relative to ``repo_dir``.
        config_path: Upstream YAML config path, relative to ``repo_dir``.
        conditioning_dir: Directory with the 5 conditioning rasters. If
            None, resolves via :mod:`csdv_core.chm_inference.conditioning`.
        chip_size: Tile size in pixels. Default from ``metrics.yaml``.
        chip_overlap: Tile overlap fraction. Default from ``metrics.yaml``.
        dry_run: If True, pass ``--dry-run`` to the upstream CLI.

    Returns:
        Output directory path.
    """
    naip_quad = Path(naip_quad).resolve()
    if not naip_quad.exists():
        raise FileNotFoundError(naip_quad)
    _validate_filename(naip_quad)

    defaults = load_metrics().chm_inference
    chip_size = chip_size if chip_size is not None else defaults.chip_size
    chip_overlap = chip_overlap if chip_overlap is not None else defaults.chip_overlap

    repo = _resolve_repo_dir(repo_dir)
    inference_script = repo / "scripts" / "inference.py"
    if not inference_script.exists():
        raise FileNotFoundError(
            f"Upstream naip-chm repo not found at {repo}. "
            "Set NAIPCHM_REPO_DIR or pass --repo-dir."
        )

    if conditioning_dir is None:
        cond = validate_conditioning().root
    else:
        cond = Path(conditioning_dir).resolve()

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(inference_script),
        "--naip-quad",
        str(naip_quad),
        "--output-dir",
        str(output_dir),
        "--model-checkpoint",
        str(model_checkpoint),
        "--config",
        str(config_path),
        "--static-rasters-dir",
        str(cond),
        "--chip-size",
        str(chip_size),
        "--chip-overlap",
        f"{chip_overlap:g}",
    ]
    if dry_run:
        cmd.append("--dry-run")

    logger.info("Running NAIP-CHM inference: %s", " ".join(cmd))
    logger.info("CWD: %s", repo)
    result = subprocess.run(cmd, cwd=repo)
    if result.returncode != 0:
        raise RuntimeError(f"NAIP-CHM inference exited with code {result.returncode}")
    logger.info("Inference complete; outputs in %s", output_dir)
    return output_dir


@click.command("chm-inference")
@click.option(
    "--naip-quad",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option(
    "--repo-dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Path to the cloned smorf-ntsg/naip-chm repository.",
)
@click.option(
    "--conditioning-dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory of conditioning rasters; defaults to project paths.",
)
@click.option("--chip-size", default=None, type=int)
@click.option("--chip-overlap", default=None, type=float)
@click.option("--dry-run", is_flag=True, default=False)
def cli(
    naip_quad: Path,
    output_dir: Path,
    repo_dir: Path | None,
    conditioning_dir: Path | None,
    chip_size: int | None,
    chip_overlap: float | None,
    dry_run: bool,
) -> None:
    """Run NAIP-CHM inference on a single NAIP AOI GeoTIFF."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s"
    )
    predict_chm(
        naip_quad=naip_quad,
        output_dir=output_dir,
        repo_dir=repo_dir,
        conditioning_dir=conditioning_dir,
        chip_size=chip_size,
        chip_overlap=chip_overlap,
        dry_run=dry_run,
    )


__all__ = ["predict_chm", "cli"]
