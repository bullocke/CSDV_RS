"""csdv_core.pairmodel.crops: 0.6 m NAIP crops of chosen boxes, for close-ups.

Inputs: a table of boxes (``unit``, ``box``, ``epsg``, ``minx``, ``miny``,
``maxx``, ``maxy``, ``years`` as space-separated years) and the unit's local
NAIP mosaics under ``<data_root>/naip/<unit>/<year>/``. Output: one
four-band uint8 GeoTIFF per box and year, ``<out_dir>/<unit>/<box>_<year>.tif``,
on the 0.6 m lattice.

The mosaics are on CHPC only. A box of 1 km and four dates is about 11 MB,
so a few dozen boxes can be pulled to the laptop for figures.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from csdv_core.label.lattice import GSD, grid_transform
from csdv_core.label.naip import local_mosaics, warp_sources

logger = logging.getLogger(__name__)

BOX_COLUMNS = ("unit", "box", "epsg", "minx", "miny", "maxx", "maxy", "years")
#: Largest box side, metres. A crop is a view for a figure, not a map.
MAX_SIDE_M = 3000.0

__all__ = ["BOX_COLUMNS", "CropReport", "read_boxes", "snap_bounds", "write_crops"]


@dataclass
class CropReport:
    """What :func:`write_crops` wrote."""

    written: list[Path] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def snap_bounds(
    bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Bounds widened to whole 0.6 m pixels of the lattice."""
    minx, miny, maxx, maxy = bounds
    return (
        math.floor(round(minx / GSD, 6)) * GSD,
        math.floor(round(miny / GSD, 6)) * GSD,
        math.ceil(round(maxx / GSD, 6)) * GSD,
        math.ceil(round(maxy / GSD, 6)) * GSD,
    )


def read_boxes(path: Path) -> pd.DataFrame:
    """Read and check a table of boxes.

    Raises:
        ValueError: If a column is missing or a box is empty or too large.
    """
    boxes = pd.read_csv(path, dtype={"years": str, "box": str, "unit": str})
    missing = [c for c in BOX_COLUMNS if c not in boxes]
    if missing:
        raise ValueError(f"{path} lacks the columns {missing}")
    width = boxes["maxx"] - boxes["minx"]
    height = boxes["maxy"] - boxes["miny"]
    bad = (width <= 0) | (height <= 0) | (width > MAX_SIDE_M) | (height > MAX_SIDE_M)
    if bad.any():
        raise ValueError(
            f"Boxes empty or wider than {MAX_SIDE_M} m: {list(boxes.loc[bad, 'box'])}"
        )
    return boxes


def write_crops(boxes: pd.DataFrame, data_root: Path, out_dir: Path) -> CropReport:
    """Write the crop of every box and year that has a mosaic.

    Existing crops are kept. A year without a mosaic is reported, not fatal.
    """
    import rasterio

    report = CropReport()
    for row in boxes.itertuples():
        bounds = snap_bounds((row.minx, row.miny, row.maxx, row.maxy))
        folder = Path(out_dir) / row.unit
        folder.mkdir(parents=True, exist_ok=True)
        for year in str(row.years).split():
            path = folder / f"{row.box}_{year}.tif"
            if path.exists():
                report.written.append(path)
                continue
            mosaics = local_mosaics(data_root, row.unit, int(year))
            if not mosaics:
                report.missing.append(f"{row.unit} {year}")
                logger.warning("%s %s: no mosaic", row.unit, year)
                continue
            data, _ = warp_sources([str(p) for p in mosaics], int(row.epsg), bounds)
            profile = {
                "driver": "GTiff",
                "height": data.shape[1],
                "width": data.shape[2],
                "count": data.shape[0],
                "dtype": "uint8",
                "crs": f"EPSG:{int(row.epsg)}",
                "transform": grid_transform(bounds),
                "nodata": 0,
                "compress": "deflate",
                "tiled": True,
            }
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(data)
            report.written.append(path)
            logger.info(
                "%s: %s, %.0f%% filled",
                path.name,
                row.unit,
                100 * float(np.mean(data.any(axis=0))),
            )
    return report
