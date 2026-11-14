"""csdv_core.audit.windows: images of a window item (background check).

A window item shows every date of a complete window with its label drawn
over it, so the interpreter can look for loss on ground labelled background.
It needs the window's stack (one four-band GeoTIFF per year), which a label
run keeps on CHPC until it is pulled.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import rasterio
from PIL import Image

from csdv_core.audit.build import label_overlay_image, save_overlay
from csdv_core.label.render import BANDS, stretch_limits, to_rgb

log = logging.getLogger(__name__)

JPEG_QUALITY = 90

__all__ = ["stack_years", "window_item"]


def stack_years(label_dir: Path, window_id: str) -> dict[str, Path] | None:
    """Year and file of every date of a window's stack, or None when a
    year's imagery is not on this machine."""
    folder = label_dir / "stacks" / window_id
    meta_path = folder / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    years = {year: folder / f"{year}.tif" for year in sorted(meta["years"])}
    if not years or not all(path.exists() for path in years.values()):
        return None
    return years


def window_item(
    row: pd.Series,
    label_dir: Path,
    site: Path,
    chips: pd.DataFrame,
    refresh: bool = False,
) -> dict | None:
    """Render a window item and return its page entry.

    Returns:
        The entry, or None when the window's stack is not on this machine.
    """
    window = row["window_id"]
    years = stack_years(label_dir, window)
    if years is None:
        return None
    meta = json.loads((label_dir / "stacks" / window / "meta.json").read_text())
    folder = site / "img" / row["item"]
    folder.mkdir(parents=True, exist_ok=True)
    dates, images, view = [], [], None
    for i, (year, path) in enumerate(years.items(), start=1):
        dates.append(meta["years"][year]["date"])
        images.append(f"img/{row['item']}/{i}.jpg")
        with rasterio.open(path) as src:
            b = src.bounds
            view = [b.left, b.bottom, b.right, b.top, src.width]
            if (folder / f"{i}.jpg").exists():
                continue
            block = src.read()
        limits = stretch_limits(block, BANDS["rgb"])
        Image.fromarray(to_rgb(block, limits)).save(
            folder / f"{i}.jpg", quality=JPEG_QUALITY
        )
    unit_chips = chips[chips["unit"] == row["unit"]]
    labels = []
    for number in range(2, len(dates) + 1):
        name = f"label_{number}.png"
        if refresh or not (folder / name).exists():
            overlay = label_overlay_image(
                unit_chips,
                label_dir / "chips",
                view,
                dates[number - 2],
                dates[number - 1],
            )
            if overlay is None:
                continue
            save_overlay(overlay, folder / name)
        labels.append({"image": number, "png": f"img/{row['item']}/{name}"})
    return {
        "id": row["item"],
        "kind": "window",
        "block": int(row["block"]),
        "reserve": bool(row["reserve"]),
        "dates": dates,
        "imgs": images,
        "metres": round(float(view[2] - view[0])),
        "px": int(view[4]),
        "place": "",
        "labels": labels,
    }
