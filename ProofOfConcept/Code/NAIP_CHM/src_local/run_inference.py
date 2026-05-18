"""Thin wrapper around the upstream ``scripts/inference.py`` CLI.

Validates inputs (filename must contain ``YYYYMMDD`` for DOY extraction),
locates the upstream repo, the model checkpoint, the conditioning data dir,
and shells out to inference. Used by both interactive runs and the SLURM
template.

Environment variables (override CLI options):
    NAIPCHM_REPO_DIR    Path to the cloned smorf-ntsg/naip-chm repository.
    NAIPCHM_COND_DIR    Path to the directory with the 5 conditioning rasters.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import click

logger = logging.getLogger(__name__)


def _validate_filename(path: Path) -> None:
    if not re.search(r"\d{8}", path.stem):
        raise click.BadParameter(
            f"NAIP filename '{path.name}' must contain a YYYYMMDD token; "
            "the upstream DOY extractor requires it.",
            param_hint="--naip-quad",
        )


@click.command()
@click.option("--naip-quad", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--repo-dir", default=lambda: os.environ.get("NAIPCHM_REPO_DIR", str(Path.home() / "code" / "naip-chm")), show_default=True, type=click.Path(path_type=Path))
@click.option("--cond-dir", default=lambda: os.environ.get("NAIPCHM_COND_DIR", ""), type=click.Path(path_type=Path), help="Directory with the 5 conditioning rasters. Defaults to <repo>/data/conditioning_data.")
@click.option("--chip-size", default=432, show_default=True, type=int)
@click.option("--chip-overlap", default=0.2, show_default=True, type=float)
@click.option("--dry-run", is_flag=True, default=False)
def main(naip_quad: Path, output_dir: Path, repo_dir: Path, cond_dir: Path, chip_size: int, chip_overlap: float, dry_run: bool) -> None:
    """Run the upstream NAIP-CHM inference CLI on a single AOI GeoTIFF."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")

    _validate_filename(naip_quad)

    repo_dir = Path(repo_dir).resolve()
    if not (repo_dir / "scripts" / "inference.py").exists():
        raise click.BadParameter(f"Upstream repo not found at {repo_dir}. Run setup/clone_and_install.sh first.", param_hint="--repo-dir")

    cond = Path(cond_dir).resolve() if str(cond_dir) else (repo_dir / "data" / "conditioning_data")
    required = ["elevation.tif", "climate_pca.tif", "soil_pca.tif", "nlcd.tif", "ecoregion.tif"]
    missing = [f for f in required if not (cond / f).exists()]
    if missing:
        raise click.BadParameter(f"Conditioning rasters missing in {cond}: {missing}", param_hint="--cond-dir")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "scripts/inference.py",
        "--naip-quad", str(naip_quad.resolve()),
        "--output-dir", str(output_dir),
        "--model-checkpoint", "model/model_20251016.pt",
        "--config", "configs/config.yaml",
        "--static-rasters-dir", str(cond),
        "--chip-size", str(chip_size),
        "--chip-overlap", str(chip_overlap),
    ]
    if dry_run:
        cmd.append("--dry-run")

    logger.info("Running: %s", " ".join(cmd))
    logger.info("CWD: %s", repo_dir)
    result = subprocess.run(cmd, cwd=repo_dir)
    if result.returncode != 0:
        logger.error("Inference exited with code %d", result.returncode)
        sys.exit(result.returncode)
    logger.info("Inference complete; outputs in %s", output_dir)


if __name__ == "__main__":
    main()
