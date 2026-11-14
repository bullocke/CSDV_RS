"""csdv_core.audit.build: the static page of an audit, with every image.

Inputs: the audit items and the label run they were drawn from. Output: a
``site`` folder holding the page, one folder of images per item and
``items.js``, the item list the page reads.

Candidate items reuse the request images the vision models saw (one per
date, with the candidate outlined), so they need no imagery beyond what a
label run already holds. The label of an accepted event is drawn as a
transparent overlay from the chips' change rasters. Nothing that names a
candidate, a status or a model enters the folder: items carry opaque names
and the key stays in ``items.csv``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from scipy import ndimage

log = logging.getLogger(__name__)

PAGE_FILES = ("index.html", "app.js", "style.css")
LOSS_RGB = (255, 60, 60)
IGNORE_RGB = (235, 235, 235)
NO_LABEL = 254
_PLACE = re.compile(r"area in (.+?)\. North is up")

__all__ = [
    "BuildReport",
    "build_site",
    "bundle_site",
    "label_overlay_image",
    "read_entries",
    "save_overlay",
    "write_page",
]


@dataclass
class BuildReport:
    """What a build wrote.

    Attributes:
        built: Items whose images are in the site folder.
        missing_stacks: Windows of target window items without imagery.
        missing_reserve: The same for the reserve items.
        skipped: Items left out, with the reason.
    """

    built: list[str] = field(default_factory=list)
    missing_stacks: list[str] = field(default_factory=list)
    missing_reserve: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)


def write_page(site: Path) -> None:
    """Copy the page files of the package into ``site``."""
    site.mkdir(parents=True, exist_ok=True)
    page = resources.files("csdv_core.audit") / "page"
    for name in PAGE_FILES:
        _replace(site / name, (page / name).read_bytes())


def _replace(path: Path, content: bytes) -> None:
    """Write a file in one step, so a page in use never reads half of it."""
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(content)
    partial.replace(path)


def _place(src: Path, dst: Path, link: bool) -> None:
    if dst.exists():
        return
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def save_overlay(overlay: Image.Image, path: Path) -> None:
    """Write an overlay in one step, so a page in use never reads half of it."""
    partial = path.with_suffix(".tmp")
    overlay.save(partial, format="PNG")
    partial.replace(path)


def _requests(label_dir: Path, window_id: str) -> dict[str, dict]:
    """Verdict requests of a window by candidate."""
    out = {}
    path = label_dir / "requests" / window_id / "requests.jsonl"
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("kind") == "verdict":
                out[record["cand_id"]] = record
    return out


def _interval_of(dates: list[str], date_a: str, date_b: str) -> int | None:
    """Index of the interval ``date_a`` to ``date_b`` among a chip's dates.

    A chip that lies in two windows carries the dates of the window that
    owns it. The other window may be dated from a neighbouring flight line
    of the same years, so when the exact dates are not found the interval
    with the same pair of years is taken.
    """
    pairs = list(zip(dates[:-1], dates[1:], strict=True))
    if (date_a, date_b) in pairs:
        return pairs.index((date_a, date_b))
    years = [(a[:4], b[:4]) for a, b in pairs]
    if (date_a[:4], date_b[:4]) in years:
        return years.index((date_a[:4], date_b[:4]))
    return None


def label_overlay_image(
    chips: pd.DataFrame,
    chips_dir: Path,
    view: list[float],
    date_a: str,
    date_b: str,
    veil_unlabelled: bool = True,
) -> Image.Image | None:
    """The label of one interval over a view, as a transparent image.

    Loss is a light red fill, with a solid edge where it meets labelled
    ground of another kind. Ignored ground is hatched and background stays
    clear. Ground without a label (outside every chip) is darkened, unless
    ``veil_unlabelled`` is off.

    Args:
        chips: Rows of ``chips.parquet`` (``chip_id``, ``epsg``, ``bounds``
            and ``dates`` as JSON text).
        chips_dir: Folder of the chip rasters.
        view: ``[minx, miny, maxx, maxy, px]`` of the image, in the CRS of
            the chips.
        date_a: Earlier date of the interval.
        date_b: Later date of the interval.

    Returns:
        An RGBA image of ``px`` by ``px``, or None when no chip of the view
        holds that interval.
    """
    minx, miny, maxx, maxy, px = view
    px = int(px)
    res = (maxx - minx) / px
    xs = minx + (np.arange(px) + 0.5) * res
    ys = maxy - (np.arange(px) + 0.5) * res
    label = np.full((px, px), NO_LABEL, dtype=np.uint8)
    found = False
    for _, chip in chips.iterrows():
        cminx, cminy, cmaxx, cmaxy = json.loads(chip["bounds"])
        if cmaxx <= minx or cminx >= maxx or cmaxy <= miny or cminy >= maxy:
            continue
        k = _interval_of(json.loads(chip["dates"]), date_a, date_b)
        if k is None:
            continue
        path = chips_dir / f"{chip['chip_id']}_change.tif"
        if not path.exists():
            continue
        with rasterio.open(path) as src:
            band = src.read(k + 1)
        size = band.shape[0]
        step = (cmaxx - cminx) / size
        cols = np.floor((xs - cminx) / step).astype(int)
        rows = np.floor((cmaxy - ys) / step).astype(int)
        col_ok = (cols >= 0) & (cols < size)
        row_ok = (rows >= 0) & (rows < size)
        if not col_ok.any() or not row_ok.any():
            continue
        label[np.ix_(row_ok, col_ok)] = band[np.ix_(rows[row_ok], cols[col_ok])]
        found = True
    if not found:
        return None
    loss = (label == 1) | (label == 2)
    ignore = label == 255
    rgba = np.zeros((px, px, 4), dtype=np.uint8)
    rgba[loss] = (*LOSS_RGB, 70)
    # No edge where loss simply runs out of the labelled ground.
    other = ~loss & (label != NO_LABEL)
    edge = loss & ndimage.binary_dilation(other, iterations=2)
    rgba[edge] = (*LOSS_RGB, 255)
    rr, cc = np.indices(label.shape)
    # A light line beside a dark one, so the hatch shows on any ground.
    rgba[ignore & ((rr + cc) % 8 == 0)] = (*IGNORE_RGB, 210)
    rgba[ignore & ((rr + cc) % 8 == 1)] = (0, 0, 0, 170)
    if veil_unlabelled:
        rgba[label == NO_LABEL] = (0, 0, 0, 150)
    return Image.fromarray(rgba, "RGBA")


def _candidate_item(
    row: pd.Series,
    label_dir: Path,
    site: Path,
    chips: pd.DataFrame,
    requests: dict[str, dict[str, dict]],
    link: bool,
    refresh: bool = False,
) -> dict:
    """Place the images of a candidate item and return its page entry."""
    window = row["window_id"]
    if window not in requests:
        requests[window] = _requests(label_dir, window)
    request = requests[window][row["cand_id"]]
    folder = site / "img" / row["item"]
    folder.mkdir(parents=True, exist_ok=True)
    images = []
    for i, name in enumerate(request["images"], start=1):
        _place(label_dir / "requests" / window / name, folder / f"{i}.jpg", link)
        images.append(f"img/{row['item']}/{i}.jpg")
    dates = list(request["dates"])
    view = request["view"]
    labels = []
    numbers = [int(n) for n in str(row["label_images"]).split(",") if n]
    unit_chips = chips[chips["unit"] == row["unit"]]
    for number in numbers:
        if number > len(dates):
            continue
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
    match = _PLACE.search(request.get("prompt", ""))
    return {
        "id": row["item"],
        "kind": "cand",
        "block": int(row["block"]),
        "reserve": bool(row["reserve"]),
        "dates": dates,
        "imgs": images,
        "metres": round(float(view[2] - view[0])),
        "place": match.group(1) if match else "",
        "labels": labels,
    }


def build_site(
    label_dir: Path,
    audit_dir: Path,
    items: pd.DataFrame,
    *,
    name: str = "audit",
    link: bool = True,
    refresh_labels: bool = False,
) -> BuildReport:
    """Write the page, the images of every item that can be built, and
    ``items.js``.

    Args:
        label_dir: Folder of the label run.
        audit_dir: Folder of the audit. The page goes to ``site`` inside it.
        items: The audit items (see :mod:`csdv_core.audit.draw`).
        name: Name of the audit, kept in the page for its stored answers.
        link: Hard-link request images instead of copying them where the
            file system allows it.
        refresh_labels: Draw the label overlays again. Images are left as
            they are.

    Returns:
        What was built and which windows still lack imagery.
    """
    from csdv_core.audit.windows import window_item

    label_dir, audit_dir = Path(label_dir), Path(audit_dir)
    site = audit_dir / "site"
    write_page(site)
    chips = pd.read_parquet(label_dir / "chips" / "chips.parquet")
    report = BuildReport()
    requests: dict[str, dict[str, dict]] = {}
    entries = []
    for _, row in items.sort_values(["block", "order"]).iterrows():
        try:
            if row["kind"] == "cand":
                entry = _candidate_item(
                    row, label_dir, site, chips, requests, link, refresh_labels
                )
            else:
                entry = window_item(row, label_dir, site, chips, refresh_labels)
                if entry is None:
                    missing = (
                        report.missing_reserve
                        if row["reserve"]
                        else report.missing_stacks
                    )
                    missing.append(row["window_id"])
                    continue
        except (KeyError, FileNotFoundError) as err:
            report.skipped[row["item"]] = repr(err)
            log.warning("Item %s left out: %r", row["item"], err)
            continue
        entries.append(entry)
        report.built.append(row["item"])
    _write_entries(site, {"name": name, "n_items": len(entries)}, entries)
    log.info(
        "Built %d items, %d windows without imagery, %d left out",
        len(report.built),
        len(report.missing_stacks) + len(report.missing_reserve),
        len(report.skipped),
    )
    return report


_META = "window.AUDIT_META = "
_ITEMS = ";\nwindow.AUDIT_ITEMS = "


def _write_entries(site: Path, meta: dict, entries: list[dict]) -> None:
    """Write ``items.js``. A script, not JSON, so the page opens from disk."""
    text = _META + json.dumps(meta) + _ITEMS + json.dumps(entries, indent=0) + ";\n"
    _replace(site / "items.js", text.encode())


def read_entries(site: Path) -> tuple[dict, list[dict]]:
    """The audit's name and item entries of a built page."""
    text = (Path(site) / "items.js").read_text()
    meta, entries = text[len(_META) :].split(_ITEMS, 1)
    return json.loads(meta), json.loads(entries.rstrip().rstrip(";"))


def bundle_site(
    audit_dir: Path, out_dir: Path, items: pd.DataFrame, *, shared_only: bool = True
) -> list[str]:
    """Copy the page with a subset of the built items into a folder to share.

    The folder is self-contained: an interpreter opens ``index.html`` from
    disk, answers stay in the browser, and "Download answers" saves them as
    one file to send back.

    Args:
        audit_dir: Folder of the audit, with a built ``site``.
        out_dir: Folder to write. It must not hold another bundle.
        items: The audit items.
        shared_only: Keep only the items every interpreter judges.

    Returns:
        Wanted items that are not built yet (for example windows whose
        imagery has not been pulled), which the bundle lacks.

    Raises:
        FileExistsError: If ``out_dir`` already holds a page.
    """
    site, out_dir = Path(audit_dir) / "site", Path(out_dir)
    if (out_dir / "items.js").exists():
        raise FileExistsError(f"{out_dir} already holds a bundle")
    meta, entries = read_entries(site)
    wanted = items[items["shared"]] if shared_only else items
    wanted = wanted.sort_values(["block", "order"])["item"].tolist()
    built = {entry["id"]: entry for entry in entries}
    kept = [built[item] for item in wanted if item in built]
    write_page(out_dir)
    for entry in kept:
        shutil.copytree(site / "img" / entry["id"], out_dir / "img" / entry["id"])
    suffix = "shared" if shared_only else "all"
    _write_entries(
        out_dir, {"name": f"{meta['name']}_{suffix}", "n_items": len(kept)}, kept
    )
    missing = [item for item in wanted if item not in built]
    log.info("Bundle %s: %d items, %d not built yet", out_dir, len(kept), len(missing))
    return missing
