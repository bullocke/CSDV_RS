"""csdv_core.chm_inference.infer — NAIP-CHM inference wrapper.

Wraps the upstream ``smorf-ntsg/naip-chm`` repository via subprocess.
The Phase 3 plan called for a vendored in-process import, but the
upstream project does not expose a stable Python API. We invoke its
``scripts/inference.py`` instead. The ``[chm_inference]`` extra installs
``torch`` only. Set ``NAIPCHM_REPO_DIR`` (or pass ``--repo-dir``) to a
local clone of https://github.com/smorf-ntsg/naip-chm.

The function validates that the NAIP filename contains a ``YYYYMMDD``
token (the upstream model extracts DOY from the filename) and that the
five conditioning rasters are present. After inference, CHM pixels where the
NAIP has no imagery, and a band around them, are set to nodata
(:mod:`csdv_core.chm_inference.naip_gaps`), because the upstream model
predicts heights over black fill.

With ``chip_lattice`` on (the default), the upstream script runs through
:mod:`csdv_core.chm_inference.tiling`, which places the chips on a fixed map
lattice so that a location gets the same CHM in every processing unit. Each
CHM gets a ``<naip stem>_tiling.json`` sidecar recording the tiling.
"""

from __future__ import annotations

import json
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


def _mask_chm_outputs(naip_quad: Path, output_dir: Path, edge_m: float) -> None:
    """Mask NAIP gaps in the CHM that inference wrote for ``naip_quad``.

    The CHM is found with the pattern ``<naip stem>*chm*.tif``, the same one
    ``scripts/slurm/run_chm_inference.sbatch`` uses to detect finished work.
    """
    from csdv_core.chm_inference.naip_gaps import mask_naip_gaps

    outputs = sorted(output_dir.glob(f"{naip_quad.stem}*chm*.tif"))
    if len(outputs) != 1:
        logger.warning(
            "Expected one CHM for %s in %s, found %d; NAIP gaps not masked",
            naip_quad.name,
            output_dir,
            len(outputs),
        )
        return
    mask_naip_gaps(naip_quad, outputs[0], edge_m=edge_m)


def _upstream_commit(repo: Path) -> str:
    """Short commit of the upstream clone, read from its ``.git`` files."""
    git = repo / ".git"
    try:
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head[:7]
        ref = head[len("ref: ") :]
        if (git / ref).is_file():
            return (git / ref).read_text(encoding="utf-8").strip()[:7]
        for line in (git / "packed-refs").read_text(encoding="utf-8").splitlines():
            if line.endswith(" " + ref):
                return line.split()[0][:7]
    except OSError:
        pass
    return ""


def _write_tiling_sidecar(
    naip_quad: Path,
    output_dir: Path,
    *,
    repo: Path,
    chip_size: int,
    chip_overlap: float,
    chip_lattice: bool,
) -> Path:
    """Record how the CHM of ``naip_quad`` was tiled, next to the CHM."""
    import rasterio

    from csdv_core.chm_inference.tiling import TARGET_RES_M, global_pixel_origin

    record: dict[str, object] = {
        "naip": naip_quad.name,
        "chip_size": chip_size,
        "chip_overlap": chip_overlap,
        "chip_lattice": chip_lattice,
        "upstream_commit": _upstream_commit(repo),
    }
    if chip_lattice:
        with rasterio.open(naip_quad) as src:
            col0, row0 = global_pixel_origin(src.transform)
        record |= {
            "lattice_res_m": TARGET_RES_M,
            "origin_col": col0,
            "origin_row": row0,
        }
    path = output_dir / f"{naip_quad.stem}_tiling.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


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
    chip_lattice: bool | None = None,
    dry_run: bool = False,
    download_conditioning: bool = False,
    mask_gaps: bool = True,
    gap_edge_m: float | None = None,
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
        chip_lattice: Place the chips on a fixed map lattice
            (:mod:`csdv_core.chm_inference.tiling`) instead of from the
            mosaic's top-left pixel. Default from ``metrics.yaml``.
        dry_run: If True, pass ``--dry-run`` to the upstream CLI.
        download_conditioning: If True, fetch any missing conditioning
            rasters (~1.65 GB) before validating. Off by default so inference
            jobs do not trigger a surprise download.
        mask_gaps: Set CHM pixels without NAIP imagery to nodata after
            inference (skipped for a dry run).
        gap_edge_m: Width (m) of the band also masked around each gap.
            Default from ``metrics.yaml``.

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
    gap_edge_m = gap_edge_m if gap_edge_m is not None else defaults.gap_edge_m
    chip_lattice = chip_lattice if chip_lattice is not None else defaults.chip_lattice

    repo = _resolve_repo_dir(repo_dir)
    inference_script = repo / "scripts" / "inference.py"
    if not inference_script.exists():
        raise FileNotFoundError(
            f"Upstream naip-chm repo not found at {repo}. "
            "Set NAIPCHM_REPO_DIR or pass --repo-dir."
        )

    if conditioning_dir is None:
        if download_conditioning:
            from csdv_core.download.chm_model import download_conditioning_rasters

            download_conditioning_rasters()
        cond = validate_conditioning().root
    else:
        cond = Path(conditioning_dir).resolve()

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = [sys.executable]
    if chip_lattice:
        runner += ["-m", "csdv_core.chm_inference.tiling"]
    cmd = [
        *runner,
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
    if mask_gaps and not dry_run:
        _mask_chm_outputs(naip_quad, output_dir, gap_edge_m)
    if not dry_run:
        _write_tiling_sidecar(
            naip_quad,
            output_dir,
            repo=repo,
            chip_size=chip_size,
            chip_overlap=chip_overlap,
            chip_lattice=chip_lattice,
        )
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
@click.option(
    "--chip-lattice/--no-chip-lattice",
    default=None,
    help="Place chips on a fixed map lattice. Default: metrics.yaml (on).",
)
@click.option("--dry-run", is_flag=True, default=False)
@click.option(
    "--download-conditioning",
    is_flag=True,
    default=False,
    help="Fetch missing conditioning rasters (~1.65 GB) before inference.",
)
@click.option(
    "--mask-gaps/--no-mask-gaps",
    default=True,
    show_default=True,
    help="Set CHM pixels without NAIP imagery to nodata after inference.",
)
@click.option(
    "--gap-edge-m",
    default=None,
    type=float,
    help="Band (m) also masked around each NAIP gap. Default: metrics.yaml.",
)
def cli(
    naip_quad: Path,
    output_dir: Path,
    repo_dir: Path | None,
    conditioning_dir: Path | None,
    chip_size: int | None,
    chip_overlap: float | None,
    chip_lattice: bool | None,
    dry_run: bool,
    download_conditioning: bool,
    mask_gaps: bool,
    gap_edge_m: float | None,
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
        chip_lattice=chip_lattice,
        dry_run=dry_run,
        download_conditioning=download_conditioning,
        mask_gaps=mask_gaps,
        gap_edge_m=gap_edge_m,
    )


__all__ = ["predict_chm", "cli"]
