"""csdv_core.grid.export — the small download set of a run.

A detection run folder holds float32 layers, z stacks and logs. The 0.6 m NAIP
and CHM mosaics behind it are a few GB per year. Neither should leave CHPC
just to look at results. This module writes what is needed to review a run
locally into one folder tree under ``<export dir>``:

``runs/<run_tag>/<unit>/``
    ``onset_year_u8.tif`` (onset year minus 2000, 0 for monitored forest with
    no detection, 255 for never monitored, with a colour table),
    ``max_probability_u8.tif`` (highest decision probability over all
    intervals in percent, 255 nodata), the run's ``disturbances.gpkg``,
    ``objects.csv``, ``summary.json``, ``report.md``, ``run_params.json``,
    log, ``figures/``, ``inspection/`` and a copy of ``unit.json``.

``quicklooks/<unit>/``
    One 3 m CHM per year (uint8, 0.25 m steps, 255 nodata) and one 3 m NAIP
    RGB per year (JPEG with an internal mask). These do not depend on the run,
    so they are written once per unit.

Every raster is a Cloud Optimized GeoTIFF with overviews.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from csdv_core.grid.units import GridPaths

logger = logging.getLogger(__name__)

#: Offset subtracted from onset years to fit them in a byte.
ONSET_YEAR_BASE = 2000

#: Byte value for nodata in every exported raster.
NODATA_U8 = 255

#: Height step of the CHM quicklooks in metres.
CHM_STEP_M = 0.25

#: Run files copied as they are (relative to the run folder -> export folder).
RUN_FILES = {
    "run_params.json": "run_params.json",
    "summary.json": "summary.json",
    "report.md": "report.md",
    "detect_disturbance.log": "detect_disturbance.log",
    "detection/disturbances.gpkg": "disturbances.gpkg",
    "detection/objects.csv": "objects.csv",
}

#: Run folders copied as they are.
RUN_DIRS = ("figures", "inspection")

__all__ = [
    "CHM_STEP_M",
    "NODATA_U8",
    "ONSET_YEAR_BASE",
    "ExportResult",
    "encode_chm",
    "encode_max_probability",
    "encode_onset",
    "export_quicklooks",
    "export_run",
    "onset_colormap",
    "write_cog",
]


@dataclass
class ExportResult:
    """What one export wrote."""

    run_dir: Path
    export_dir: Path
    files: list[Path] = field(default_factory=list)

    @property
    def size_mb(self) -> float:
        return sum(f.stat().st_size for f in self.files if f.is_file()) / 1e6


def encode_onset(onset: np.ndarray) -> np.ndarray:
    """Onset years (int16: year, 0 none, -1 never monitored) as bytes.

    Years become ``year - 2000`` (2018 -> 18), 0 stays 0 and -1 becomes 255.
    """
    out = np.zeros(onset.shape, dtype=np.uint8)
    years = onset > 0
    out[years] = np.clip(onset[years] - ONSET_YEAR_BASE, 1, NODATA_U8 - 1)
    out[onset < 0] = NODATA_U8
    return out


def onset_colormap(codes: Sequence[int]) -> dict[int, tuple[int, int, int, int]]:
    """Colour table for :func:`encode_onset` codes: early yellow to late purple.

    0 (no detection) is light grey and 255 (never monitored) transparent.
    """
    from matplotlib import colormaps

    cmap = colormaps["plasma_r"]
    years = sorted(c for c in set(codes) if 0 < c < NODATA_U8)
    table: dict[int, tuple[int, int, int, int]] = {
        0: (220, 220, 220, 255),
        NODATA_U8: (0, 0, 0, 0),
    }
    for i, code in enumerate(years):
        r, g, b, _ = cmap(0.15 + 0.85 * i / max(len(years) - 1, 1))
        table[code] = (int(255 * r), int(255 * g), int(255 * b), 255)
    return table


def encode_max_probability(probability: np.ndarray) -> np.ndarray:
    """Highest probability over the bands (intervals) in percent, as bytes.

    Cells that are NaN in every band become 255.
    """
    valid = np.isfinite(probability).any(axis=0)
    peak = np.nanmax(np.where(np.isfinite(probability), probability, -np.inf), axis=0)
    out = np.full(peak.shape, NODATA_U8, dtype=np.uint8)
    out[valid] = np.clip(np.rint(100 * peak[valid]), 0, 100).astype(np.uint8)
    return out


def encode_chm(height_m: np.ma.MaskedArray) -> np.ndarray:
    """Heights in metres as bytes in :data:`CHM_STEP_M` steps (0 to 63.5 m)."""
    filled = np.ma.filled(np.ma.asarray(height_m, dtype="float64"), np.nan)
    out = np.full(filled.shape, NODATA_U8, dtype=np.uint8)
    valid = np.isfinite(filled)
    out[valid] = np.clip(np.rint(filled[valid] / CHM_STEP_M), 0, NODATA_U8 - 1)
    return out


def write_cog(
    path: Path,
    data: np.ndarray,
    *,
    crs: Any,
    transform: Any,
    nodata: float | None = NODATA_U8,
    compress: str = "DEFLATE",
    resampling: str = "NEAREST",
    colormap: dict[int, tuple[int, int, int, int]] | None = None,
    mask: np.ndarray | None = None,
    scale: float | None = None,
    descriptions: Sequence[str] | None = None,
) -> Path:
    """Write a ``(bands, rows, cols)`` or 2-D array as a Cloud Optimized GeoTIFF.

    Args:
        compress: ``DEFLATE`` for codes and heights, ``JPEG`` for RGB imagery
            (then pass ``mask`` and ``nodata=None``).
        resampling: Overview resampling, ``NEAREST`` for codes.
        colormap: Colour table of band 1.
        mask: 2-D validity mask (255 valid, 0 not), stored as an internal mask.
        scale: Value scale written to the band metadata (e.g. 0.25 m per step).
        descriptions: Band descriptions.
    """
    import rasterio
    import rasterio.shutil
    from rasterio.io import MemoryFile

    if data.ndim == 2:
        data = data[np.newaxis]
    count, height, width = data.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": data.dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    if compress == "JPEG":
        profile["photometric"] = "RGB"
    path.parent.mkdir(parents=True, exist_ok=True)
    options: dict[str, Any] = {
        "COMPRESS": compress,
        "OVERVIEW_RESAMPLING": resampling,
        "BLOCKSIZE": 256,
    }
    if compress == "JPEG":
        options["QUALITY"] = 85
    else:
        options["PREDICTOR"] = "YES"
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), MemoryFile() as mem:
        with mem.open(**profile) as tmp:
            tmp.write(data)
            if colormap is not None:
                tmp.write_colormap(1, colormap)
            if mask is not None:
                tmp.write_mask(mask)
            if scale is not None:
                tmp.scales = (scale,) * count
            for i, text in enumerate(descriptions or (), start=1):
                tmp.set_band_description(i, text)
        with mem.open() as tmp:
            rasterio.shutil.copy(tmp, path, driver="COG", **options)
    return path


def _decimated_read(src: Any, indexes: list[int], res_m: float) -> tuple[Any, Any]:
    """Read bands averaged to ``res_m`` pixels; memory stays at the output size."""
    from affine import Affine
    from rasterio.enums import Resampling

    factor = res_m / src.res[0]
    width = max(1, int(round(src.width / factor)))
    height = max(1, int(round(src.height / factor)))
    data = src.read(
        indexes,
        out_shape=(len(indexes), height, width),
        resampling=Resampling.average,
        masked=True,
    )
    transform = src.transform @ Affine.scale(src.width / width, src.height / height)
    return data, transform


def _year_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir() and d.name.isdigit())


def export_quicklooks(
    paths: GridPaths,
    unit_id: str,
    *,
    res_m: float = 3.0,
    overwrite: bool = False,
) -> list[Path]:
    """Write the unit's 3 m CHM and NAIP RGB quicklooks, one per year.

    Existing quicklooks are kept unless ``overwrite`` is set.
    """
    import rasterio

    out_dir = paths.quicklook_dir(unit_id)
    tag = f"{res_m:g}m".replace(".", "p")
    written: list[Path] = []

    for year_dir in _year_dirs(paths.data_root / "naip_chm" / unit_id):
        tifs = sorted(year_dir.glob("*chm*.tif"))
        out = out_dir / f"chm_{year_dir.name}_{tag}.tif"
        if not tifs or (out.exists() and not overwrite):
            continue
        with rasterio.open(tifs[0]) as src:
            data, transform = _decimated_read(src, [1], res_m)
            unit_scale = 0.01 if np.dtype(src.dtypes[0]).kind in "iu" else 1.0
            crs = src.crs
        codes = encode_chm(data[0].astype("float64") * unit_scale)
        written.append(
            write_cog(
                out,
                codes,
                crs=crs,
                transform=transform,
                resampling="AVERAGE",
                scale=CHM_STEP_M,
                descriptions=[f"CHM {year_dir.name}, metres = value x {CHM_STEP_M}"],
            )
        )
        logger.info("Wrote %s", out)

    for year_dir in _year_dirs(paths.data_root / "naip" / unit_id):
        tifs = sorted(year_dir.glob("*.tif"))
        out = out_dir / f"naip_{year_dir.name}_{tag}_rgb.tif"
        if not tifs or (out.exists() and not overwrite):
            continue
        with rasterio.open(tifs[0]) as src:
            data, transform = _decimated_read(src, [1, 2, 3], res_m)
            crs = src.crs
        valid = ~np.ma.getmaskarray(data).any(axis=0)
        rgb = np.ma.filled(data, 0).astype(np.uint8)
        written.append(
            write_cog(
                out,
                rgb,
                crs=crs,
                transform=transform,
                nodata=None,
                compress="JPEG",
                resampling="AVERAGE",
                mask=np.where(valid, 255, 0).astype(np.uint8),
            )
        )
        logger.info("Wrote %s", out)
    return written


def export_run(
    paths: GridPaths,
    unit_id: str,
    run_tag: str,
    *,
    quicklooks: bool = True,
    quicklook_res_m: float = 3.0,
    overwrite_quicklooks: bool = False,
) -> ExportResult:
    """Write the download set of one run (and the unit's quicklooks).

    Raises:
        FileNotFoundError: If the run folder has no ``summary.json``.
    """
    import rasterio

    run_dir = paths.run_dir(unit_id, run_tag)
    if not (run_dir / "summary.json").exists():
        raise FileNotFoundError(f"No finished detection run at {run_dir}")
    out_dir = paths.export_run_dir(unit_id, run_tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = ExportResult(run_dir, out_dir)

    detection = run_dir / "detection"
    with rasterio.open(detection / "onset_year.tif") as src:
        onset = src.read(1)
        crs, transform = src.crs, src.transform
    codes = encode_onset(onset)
    result.files.append(
        write_cog(
            out_dir / "onset_year_u8.tif",
            codes,
            crs=crs,
            transform=transform,
            colormap=onset_colormap(np.unique(codes).tolist()),
            descriptions=[
                f"Onset year minus {ONSET_YEAR_BASE}; 0 no detection; "
                f"{NODATA_U8} never monitored"
            ],
        )
    )
    if (detection / "probability.tif").exists():
        with rasterio.open(detection / "probability.tif") as src:
            probability = src.read().astype("float32")
            if src.nodata is not None and not np.isnan(src.nodata):
                probability[probability == src.nodata] = np.nan
        result.files.append(
            write_cog(
                out_dir / "max_probability_u8.tif",
                encode_max_probability(probability),
                crs=crs,
                transform=transform,
                resampling="AVERAGE",
                descriptions=["Highest disturbance probability over intervals, %"],
            )
        )

    for src_rel, dst_rel in RUN_FILES.items():
        src_path = run_dir / src_rel
        if src_path.exists():
            result.files.append(Path(shutil.copy2(src_path, out_dir / dst_rel)))
    for name in RUN_DIRS:
        if (run_dir / name).is_dir():
            shutil.copytree(run_dir / name, out_dir / name, dirs_exist_ok=True)
            result.files.extend(p for p in (out_dir / name).rglob("*") if p.is_file())
    unit_json = paths.unit_json(unit_id)
    if unit_json.exists():
        result.files.append(Path(shutil.copy2(unit_json, out_dir / "unit.json")))
    (out_dir / "export.json").write_text(
        json.dumps(
            {"unit": unit_id, "run_tag": run_tag, "run_dir": str(run_dir)}, indent=2
        ),
        encoding="utf-8",
    )

    if quicklooks:
        result.files.extend(
            export_quicklooks(
                paths, unit_id, res_m=quicklook_res_m, overwrite=overwrite_quicklooks
            )
        )
    logger.info(
        "Exported %s %s: %.1f MB in %s", unit_id, run_tag, result.size_mb, out_dir
    )
    return result
