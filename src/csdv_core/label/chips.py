"""csdv_core.label.chips: write the training chips of a window.

Inputs: a window, its imagery stack and its label rasters.
Outputs: per chip, ``<chip_id>_naip.tif`` (dates by four bands),
``_change.tif``, ``_agent.tif``, ``_reason.tif`` (one band per interval) and
``_aux.tif`` (disturbance probability per interval and the forest mask per
date, which are not model inputs), plus one table row per chip.

The vector labels are the source of truth. Chips are derived and can be
rebuilt, and the change mask itself can be rebuilt from the reason raster
under another policy.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from csdv_core.label.labels import WindowLabels
from csdv_core.label.lattice import Chip, Window, pixel_slices
from csdv_core.label.rules import IGNORE
from csdv_core.label.stacks import BAND_NAMES, WindowStack

logger = logging.getLogger(__name__)

__all__ = [
    "read_chip",
    "write_label_chips",
    "write_naip_chips",
    "write_window_chips",
]


def _write(
    path: Path,
    data: np.ndarray,
    chip: Chip,
    names: list[str],
    *,
    nodata: int | None = None,
    predictor: int = 1,
) -> None:
    import rasterio

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype="uint8",
        crs=f"EPSG:{chip.epsg}",
        transform=chip.transform,
        compress="DEFLATE",
        predictor=predictor,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        photometric="MINISBLACK",
        nodata=nodata,
    ) as dst:
        dst.write(data)
        dst.descriptions = tuple(names)


def write_naip_chips(
    window: Window,
    stack: WindowStack,
    out_dir: Path,
    *,
    only: set[str] | None = None,
) -> int:
    """Write ``<chip_id>_naip.tif`` for the chips of a window.

    Needs the imagery only, so it can run where the mosaics are.

    Args:
        only: Chip IDs to write. A chip that lies in two windows belongs to
            the first, which also gives it its label rasters and its dates.
            Windows are dated from their own quad, so the same chip written
            from the second window could carry other band names than its
            labels. Default: every chip of the window.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [f"{d}_{b}" for d in stack.dates for b in BAND_NAMES]
    chips = [c for c in window.chips() if only is None or c.chip_id in only]
    for chip in chips:
        naip = stack.crop(chip.bounds)
        naip = naip.reshape(-1, naip.shape[-2], naip.shape[-1])
        _write(out_dir / f"{chip.chip_id}_naip.tif", naip, chip, names, predictor=2)
    return len(chips)


def write_label_chips(
    window: Window,
    labels: WindowLabels,
    out_dir: Path,
    *,
    dates: list[str],
    native_m: list[float | None] | None = None,
    skip: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Write the label rasters of every chip and return one table row per chip.

    Needs no imagery, so it can run where the verdicts are.

    Args:
        skip: Chip IDs that another window has already written. They are
            left as they are and get no row.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    aux_names = [f"prob_{i}" for i in labels.intervals] + [f"forest_{d}" for d in dates]
    for chip in window.chips():
        if skip and chip.chip_id in skip:
            continue
        r, c = pixel_slices(chip.bounds, window.bounds)
        change = labels.change[:, r, c]
        stem = out_dir / chip.chip_id
        _write(
            Path(f"{stem}_change.tif"), change, chip, labels.intervals, nodata=IGNORE
        )
        _write(
            Path(f"{stem}_agent.tif"),
            labels.agent[:, r, c],
            chip,
            labels.intervals,
            nodata=IGNORE,
        )
        _write(
            Path(f"{stem}_reason.tif"), labels.reason[:, r, c], chip, labels.intervals
        )
        aux = np.concatenate([labels.prob[:, r, c], labels.forest[:, r, c]])
        _write(Path(f"{stem}_aux.tif"), aux, chip, aux_names)
        rows.append(
            {
                "chip_id": chip.chip_id,
                "window_id": window.window_id,
                "unit": window.unit,
                "epsg": chip.epsg,
                "bounds": json.dumps(list(chip.bounds)),
                "dates": json.dumps(dates),
                "native_m": json.dumps(native_m or []),
                "background": json.dumps((change == 0).sum(axis=(1, 2)).tolist()),
                "partial": json.dumps((change == 1).sum(axis=(1, 2)).tolist()),
                "full": json.dumps((change == 2).sum(axis=(1, 2)).tolist()),
                "ignore_share": float((change == IGNORE).mean()),
                "pixels": int(change[0].size),
            }
        )
    return rows


def write_window_chips(
    window: Window,
    stack: WindowStack,
    labels: WindowLabels,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Write imagery and labels of every chip of a window."""
    write_naip_chips(window, stack, out_dir)
    return write_label_chips(
        window, labels, out_dir, dates=stack.dates, native_m=stack.native_m
    )


def read_chip(stem: Path) -> dict[str, Any]:
    """Read a chip back: arrays and band names of each of its five files."""
    import rasterio

    out: dict[str, Any] = {}
    for part in ("naip", "change", "agent", "reason", "aux"):
        with rasterio.open(f"{stem}_{part}.tif") as src:
            out[part] = src.read()
            out[f"{part}_names"] = list(src.descriptions)
            out["transform"] = src.transform
            out["crs"] = src.crs
    return out
