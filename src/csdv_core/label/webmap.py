"""csdv_core.label.webmap: a static map page of a label run.

The page shows every place of a label run on one map: the images of each
year, the labels of each interval by loss type, the candidates that were
rejected, and the map of the pair model that proposed candidates. It is a
folder of plain files (one HTML page, images, small scripts with the data),
so any web server can serve it and it also opens from disk.

Layout of the output folder::

    index.html                 the page
    page.js                    places, years, intervals, counts, examples
    p/<unit>/naip_<year>.webp  image of a year, Web Mercator, with alpha
    p/<unit>/model_<a>_<b>.png probability of loss of the pair model
    p/<unit>/labels.js         events and rejected candidates of the place
    ex/<n>_a.webp, <n>_b.webp  before and after of an example event

Images come from the NAIP mosaic of a unit and year
(``<data_root>/naip/<unit>/<year>/*.tif``) where it exists, and from the
3 m quicklook of the exports otherwise (``<export_dir>/quicklooks``). A
place is drawn at the requested resolution with mosaics and at the
quicklooks' resolution without. All images of a place share one pixel grid.

An event is stored once per window that covers it, and windows overlap, so
events are counted and drawn once per candidate.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["WebmapConfig", "WebmapResult", "build_webmap", "target_group"]

#: Loss types of the labels, in legend order.
TYPES = ("harvest", "fire", "wind", "biotic_mortality", "conversion", "other", "none")
#: Statuses of judged candidates that are drawn besides the events.
REJECTED = {
    "rejected": "background",
    "rejected_weak": "unlabelled",
    "disagree": "unlabelled",
}
#: Cell size of the pair model's map in metres.
MODEL_CELL_M = 9.6
#: Probability (percent) below which the model's map is transparent.
MODEL_FLOOR = 20
_MODEL_RGB = (255, 43, 214)
_TEMPLATE = Path(__file__).with_name("webmap.html")


@dataclass
class WebmapConfig:
    """Settings of one page.

    Attributes:
        label_dir: Label run folder (``labels/labels.gpkg``, ``sites.csv``).
        out_dir: Folder of the page.
        export_dir: Exports folder with ``quicklooks`` and ``run_layers``.
        data_root: Data folder with ``naip/<unit>/<year>`` mosaics, or None.
        sites_csv: Site table. Default ``sites.csv`` of the label run.
        title: Heading of the page.
        resolution: Ground resolution of the images in metres.
        quality: WebP quality of the images.
        units: Only these units, or None for all.
        examples: Example events per loss type.
        run_tag: Tag of the disturbance run that holds the model's map.
        workers: Processes, one place each.
        overwrite: Draw images again that exist.
        budget_gb: Largest size of the page. The first place is drawn, the
            whole page is projected from its size per year, and the build
            stops there when the projection is larger. None for no limit.
    """

    label_dir: Path
    out_dir: Path
    export_dir: Path
    data_root: Path | None = None
    sites_csv: Path | None = None
    title: str = "Training labels"
    resolution: float = 1.2
    quality: int = 80
    units: tuple[str, ...] | None = None
    examples: int = 12
    run_tag: str = "default"
    workers: int = 1
    overwrite: bool = False
    budget_gb: float | None = None


@dataclass
class WebmapResult:
    """What a build wrote."""

    places: int
    images: int
    events: int
    megabytes: float
    skipped: list[str] = field(default_factory=list)


def target_group(target: str, kind: str) -> str:
    """Loss type a place was sited for, or ``background``."""
    if kind != "loss":
        return "background"
    for prefix, group in (
        ("harvest", "harvest"),
        ("fire", "fire"),
        ("wind", "wind"),
        ("conversion", "conversion"),
        ("beetle", "biotic_mortality"),
        ("drought", "biotic_mortality"),
        ("mortality", "biotic_mortality"),
        ("southern_pine_beetle", "biotic_mortality"),
    ):
        if target.startswith(prefix):
            return group
    return "other"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _years_of(interval: str) -> tuple[int, int]:
    """Years of ``2012-06-29_2014-07-24`` or ``2012-2014``."""
    parts = interval.split("_" if "_" in interval else "-")
    return int(parts[0][:4]), int(parts[-1][:4])


def _text(value: Any) -> str:
    return (
        ""
        if value is None or (isinstance(value, float) and math.isnan(value))
        else str(value)
    )


def _polygonal(geometry: Any) -> Any:
    """The parts of a geometry that have an area (a mixed overlay result)."""
    if geometry.geom_type != "GeometryCollection":
        return geometry
    from shapely.ops import unary_union

    return unary_union([part for part in geometry.geoms if part.area > 0])


def read_labels(label_dir: Path) -> tuple[Any, Any, pd.DataFrame]:
    """Events, drawn candidates and acquisitions of a label run.

    Returns:
        Events (one row per candidate, EPSG:5070, with the readings of both
        models joined on), the rejected and split candidates (one row per
        candidate), and the acquisition table.
    """
    import geopandas as gpd

    gpkg = Path(label_dir) / "labels" / "labels.gpkg"
    events = gpd.read_file(gpkg, layer="events")
    events = events.drop_duplicates("candidate_ids").reset_index(drop=True)
    cands = gpd.read_file(gpkg, layer="candidates")
    cands = cands[cands["judged"].astype(bool)].drop_duplicates("cand_id")
    told = cands.set_index("cand_id")[
        ["status", "types", "reason_gem37", "reason_terra", "area_ha"]
    ]
    events = events.join(told, on="candidate_ids")
    events["geometry"] = events.geometry.map(_polygonal)
    events["ha"] = events.geometry.area / 1e4
    years = events["loss_intervals"].map(_years_of)
    events["a"] = [y[0] for y in years]
    events["b"] = [y[1] for y in years]
    events["agent"] = events["agent"].where(events["agent"].notna(), "none")
    rejected = cands[cands["status"].isin(REJECTED)].reset_index(drop=True)
    rejected["geometry"] = rejected.geometry.map(_polygonal)
    years = rejected["interval"].map(_years_of)
    rejected["a"] = [y[0] for y in years]
    rejected["b"] = [y[1] for y in years]
    acq = gpd.read_file(gpkg, layer="acquisitions", ignore_geometry=True)
    return events, rejected, pd.DataFrame(acq)


def choose_examples(events: Any, per_type: int) -> dict[str, list[int]]:
    """Index labels of example events, spread over places, per unit.

    Events of 0.3 to 40 ha that both models accepted with the interval
    agreed are ranked by confidence, outline quality and size. Places take
    turns until a type has ``per_type`` examples.
    """
    pool = events[
        events["ha"].between(0.3, 40)
        & (events["status"] == "accepted")
        & (events["confidence"] >= 2)
    ].copy()
    pool["precise"] = pool["geometry_quality"] == "precise"
    pool = pool.sort_values(
        ["confidence", "precise", "ha", "candidate_ids"],
        ascending=[False, False, False, True],
    )
    chosen: dict[str, list[int]] = {}
    for agent in TYPES[:5]:
        of_type = pool[pool["agent"] == agent]
        queues = [list(group.index) for _, group in of_type.groupby("unit", sort=True)]
        queues.sort(key=len, reverse=True)
        picked: list[int] = []
        while queues and len(picked) < per_type:
            for queue in list(queues):
                if len(picked) >= per_type:
                    break
                picked.append(queue.pop(0))
                if not queue:
                    queues.remove(queue)
        for index in picked:
            chosen.setdefault(events.at[index, "unit"], []).append(index)
    return chosen


# ---------------------------------------------------------------------------
# Rasters
# ---------------------------------------------------------------------------


def _naip_source(cfg: WebmapConfig, unit: str, year: int) -> tuple[Path | None, bool]:
    """Image file of a unit and year, and whether it is the full mosaic."""
    if cfg.data_root is not None:
        tifs = sorted((Path(cfg.data_root) / "naip" / unit / str(year)).glob("*.tif"))
        if tifs:
            return tifs[0], True
    looks = sorted(
        (Path(cfg.export_dir) / "quicklooks" / unit).glob(f"naip_{year}_*_rgb.tif")
    )
    return (looks[0], False) if looks else (None, False)


def _place_grid(path: Path, res_m: float) -> tuple[Any, int, int, list[list[float]]]:
    """Web Mercator grid over an image: transform, width, height, bounds.

    The bounds are ``[[south, west], [north, east]]`` in degrees, as Leaflet
    takes them. A pixel is ``res_m`` on the ground at the place's latitude.
    """
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds

    with rasterio.open(path) as src:
        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        left, bottom, right, top = transform_bounds(
            src.crs, "EPSG:3857", *src.bounds, densify_pts=21
        )
    step = res_m / math.cos(math.radians((south + north) / 2))
    width = math.ceil((right - left) / step)
    height = math.ceil((top - bottom) / step)
    transform = from_origin(left, top, step, step)
    right, bottom = left + width * step, top - height * step
    lon0, lat0, lon1, lat1 = transform_bounds(
        "EPSG:3857", "EPSG:4326", left, bottom, right, top
    )
    return transform, width, height, [[lat0, lon0], [lat1, lon1]]


def _warp_rgba(
    path: Path, transform: Any, width: int, height: int, full: bool
) -> np.ndarray:
    """An image on the place grid as ``(height, width, 4)`` bytes."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    out = np.zeros((4, height, width), dtype=np.uint8)
    with rasterio.open(path) as src:
        reproject(
            rasterio.band(src, [1, 2, 3]),
            out[:3],
            dst_transform=transform,
            dst_crs="EPSG:3857",
            resampling=Resampling.average if full else Resampling.bilinear,
        )
        mask = src.dataset_mask()
        reproject(
            mask,
            out[3],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs="EPSG:3857",
            resampling=Resampling.nearest,
        )
    return np.moveaxis(out, 0, -1)


def _model_images(
    path: Path, transform: Any, width: int, height: int, res_m: float, out_dir: Path
) -> dict[tuple[int, int], str]:
    """Write the model's map per interval as PNG. Returns file names."""
    import rasterio
    from affine import Affine
    from PIL import Image
    from rasterio.warp import Resampling, reproject

    factor = max(1, round(MODEL_CELL_M / res_m))
    w, h = math.ceil(width / factor), math.ceil(height / factor)
    coarse = transform @ Affine.scale(factor)
    written: dict[tuple[int, int], str] = {}
    with rasterio.open(path) as src:
        for band, name in enumerate(src.descriptions, start=1):
            if not name:
                continue
            years = _years_of(name)
            prob = np.full((h, w), 255, dtype=np.uint8)
            reproject(
                rasterio.band(src, band),
                prob,
                dst_transform=coarse,
                dst_crs="EPSG:3857",
                src_nodata=255,
                dst_nodata=255,
                resampling=Resampling.nearest,
            )
            share = np.where(prob == 255, 0, prob).astype(np.float32)
            alpha = np.clip((share - MODEL_FLOOR) / (100 - MODEL_FLOOR), 0, 1)
            alpha = np.where(share >= MODEL_FLOOR, 60 + alpha * 160, 0)
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[..., :3] = _MODEL_RGB
            rgba[..., 3] = alpha.astype(np.uint8)
            file = f"model_{years[0]}_{years[1]}.png"
            Image.fromarray(rgba, "RGBA").save(out_dir / file, optimize=True)
            written[years] = file
    return written


def _thumb(
    image: np.ndarray, box: tuple[int, int, int, int], rings: list, size: int = 240
) -> Any:
    """A square crop of a place image, with an outline when rings are given."""
    from PIL import Image, ImageDraw

    x0, y0, x1, y1 = box
    crop = Image.fromarray(image[y0:y1, x0:x1, :3], "RGB").resize(
        (size, size), Image.Resampling.LANCZOS
    )
    scale = size / (x1 - x0)
    draw = ImageDraw.Draw(crop)
    for ring in rings:
        points = [((x - x0) * scale, (y - y0) * scale) for x, y in ring]
        draw.line(points + points[:1], fill=(255, 255, 255), width=2)
    return crop


# ---------------------------------------------------------------------------
# One place
# ---------------------------------------------------------------------------


def _rounded(geometry: Any) -> dict[str, Any]:
    """GeoJSON geometry with coordinates to six decimals (about 0.1 m)."""
    from shapely.geometry import mapping

    def walk(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            if value and isinstance(value[0], (int, float)):
                return [round(value[0], 6), round(value[1], 6)]
            return [walk(v) for v in value]
        return value

    out = mapping(geometry)
    return {"type": out["type"], "coordinates": walk(out["coordinates"])}


def _features(events: Any, rejected: Any) -> list[dict[str, Any]]:
    """Events and rejected candidates of a place as GeoJSON features."""
    features = []
    ev = events.to_crs("EPSG:4326")
    for row in ev.itertuples():
        models = (_text(row.types) + "/").split("/")
        features.append(
            {
                "type": "Feature",
                "geometry": _rounded(row.geometry),
                "properties": {
                    "id": row.candidate_ids,
                    "layer": "loss",
                    "agent": row.agent,
                    "a": int(row.a),
                    "b": int(row.b),
                    "dates": row.loss_intervals.replace("_", " to "),
                    "disputed": _text(row.status) == "date_disputed",
                    "severity": _text(row.severity),
                    "source": _text(row.candidate_source),
                    "ha": round(float(row.ha), 2),
                    "confidence": int(row.confidence),
                    "outline": _text(row.geometry_quality),
                    "type_gem37": models[0],
                    "type_terra": models[1],
                    "reason_gem37": _text(row.reason_gem37),
                    "reason_terra": _text(row.reason_terra),
                },
            }
        )
    rej = rejected.to_crs("EPSG:4326")
    for row in rej.itertuples():
        features.append(
            {
                "type": "Feature",
                "geometry": _rounded(row.geometry),
                "properties": {
                    "id": row.cand_id,
                    "layer": REJECTED[row.status],
                    "status": row.status,
                    "a": int(row.a),
                    "b": int(row.b),
                    "source": _text(row.source),
                    "ha": round(float(row.area_ha), 2),
                    "reason_gem37": _text(row.reason_gem37),
                    "reason_terra": _text(row.reason_terra),
                },
            }
        )
    return features


def _dates(acq: pd.DataFrame) -> dict[int, list[str]]:
    """Acquisition dates per year, the most frequent first."""
    out: dict[int, list[str]] = {}
    for year, group in acq.groupby("year"):
        counts = group["acquisition_date"].value_counts()
        out[int(year)] = [str(d) for d in counts.index]
    return out


def _build_place(task: dict[str, Any]) -> dict[str, Any] | None:
    """Write the files of one place. Returns its entry for ``page.js``."""
    from PIL import Image
    from rasterio.transform import rowcol
    from rasterio.warp import transform_geom

    cfg: WebmapConfig = task["cfg"]
    site: dict[str, Any] = task["site"]
    unit = site["unit"]
    events, rejected = task["events"], task["rejected"]
    dates: dict[int, list[str]] = task["dates"]
    sources = {year: _naip_source(cfg, unit, year) for year in sorted(dates)}
    sources = {year: s for year, s in sources.items() if s[0] is not None}
    if not sources:
        logger.warning("%s: no image of any year, left out", unit)
        return None
    full = any(is_full for _, is_full in sources.values())
    first = next(path for path, is_full in sources.values() if is_full == full)
    res = cfg.resolution
    if not full:
        import rasterio

        with rasterio.open(first) as src:
            res = max(res, abs(src.transform.a))
    transform, width, height, bounds = _place_grid(first, res)
    out = Path(cfg.out_dir) / "p" / unit
    out.mkdir(parents=True, exist_ok=True)

    # Years whose pixels the example crops need.
    wanted = events.loc[task["examples"]] if task["examples"] else events.iloc[:0]
    keep = set(wanted["a"]) | set(wanted["b"])
    arrays: dict[int, np.ndarray] = {}
    years = []
    written = 0
    for year, (path, is_full) in sources.items():
        file = out / f"naip_{year}.webp"
        if cfg.overwrite or not file.is_file():
            rgba = _warp_rgba(path, transform, width, height, is_full)
            Image.fromarray(rgba, "RGBA").save(file, quality=cfg.quality, method=4)
            written += 1
            if year in keep:
                arrays[year] = rgba
        elif year in keep:
            arrays[year] = np.asarray(Image.open(file).convert("RGBA"))
        years.append(
            {
                "year": year,
                "date": dates[year][0],
                "other_dates": dates[year][1:],
                "image": f"p/{unit}/{file.name}",
            }
        )

    model = (
        Path(cfg.export_dir)
        / "run_layers"
        / unit
        / cfg.run_tag
        / "model"
        / f"prob_{unit}.tif"
    )
    model_files: dict[tuple[int, int], str] = {}
    if model.is_file():
        model_files = _model_images(model, transform, width, height, res, out)

    features = _features(events, rejected)
    payload = json.dumps({"type": "FeatureCollection", "features": features})
    (out / "labels.js").write_text(
        f"window.CSDV_LABELS=window.CSDV_LABELS||{{}};window.CSDV_LABELS[{json.dumps(unit)}]={payload};",
        encoding="utf-8",
    )

    listed = [y["year"] for y in years]
    intervals = []
    for a, b in zip(listed[:-1], listed[1:], strict=True):
        intervals.append(
            {
                "a": a,
                "b": b,
                "model": (
                    f"p/{unit}/{model_files[(a, b)]}" if (a, b) in model_files else None
                ),
                "loss": int(((events["a"] == a) & (events["b"] == b)).sum()),
                "background": int(
                    (
                        (rejected["a"] == a)
                        & (rejected["b"] == b)
                        & (rejected["status"] == "rejected")
                    ).sum()
                ),
            }
        )

    examples = []
    ex_dir = Path(cfg.out_dir) / "ex"
    ex_dir.mkdir(parents=True, exist_ok=True)
    for index, row in wanted.iterrows():
        if row["a"] not in arrays or row["b"] not in arrays:
            continue
        geom = transform_geom("EPSG:5070", "EPSG:3857", row.geometry.__geo_interface__)
        polygons = (
            geom["coordinates"]
            if geom["type"] == "MultiPolygon"
            else [geom["coordinates"]]
        )
        rings = []
        for polygon in polygons:
            xs, ys = zip(*polygon[0], strict=True)
            rows, cols = rowcol(transform, xs, ys)
            rings.append(list(zip(cols, rows, strict=True)))
        cols = [c for ring in rings for c, _ in ring]
        rows = [r for ring in rings for _, r in ring]
        side = max(max(cols) - min(cols), max(rows) - min(rows))
        side = int(max(side * 1.6, 160 / res * 1.2))
        cx, cy = (min(cols) + max(cols)) // 2, (min(rows) + max(rows)) // 2
        x0 = int(min(max(cx - side // 2, 0), max(width - side, 0)))
        y0 = int(min(max(cy - side // 2, 0), max(height - side, 0)))
        box = (x0, y0, min(x0 + side, width), min(y0 + side, height))
        if box[2] - box[0] != box[3] - box[1]:
            continue
        name = f"{unit}_{index}"
        _thumb(arrays[row["a"]], box, []).save(ex_dir / f"{name}_a.webp", quality=82)
        _thumb(arrays[row["b"]], box, rings).save(ex_dir / f"{name}_b.webp", quality=82)
        centre = row.geometry.centroid
        lonlat = transform_geom(
            "EPSG:5070",
            "EPSG:4326",
            {"type": "Point", "coordinates": (centre.x, centre.y)},
        )["coordinates"]
        examples.append(
            {
                "id": row["candidate_ids"],
                "unit": unit,
                "agent": row["agent"],
                "a": int(row["a"]),
                "b": int(row["b"]),
                "ha": round(float(row["ha"]), 1),
                "lat": round(lonlat[1], 6),
                "lon": round(lonlat[0], 6),
                "before": f"ex/{name}_a.webp",
                "after": f"ex/{name}_b.webp",
            }
        )

    counts = {k: int(v) for k, v in events["agent"].value_counts().items()}
    logger.info(
        "%s: %d years (%d drawn, %.1f m), %d events, %d rejected candidates",
        unit,
        len(years),
        written,
        res,
        len(events),
        len(rejected),
    )
    return {
        "unit": unit,
        "label": _text(site.get("label")),
        "target": _text(site.get("target")),
        "group": target_group(_text(site.get("target")), _text(site.get("kind"))),
        "kind": _text(site.get("kind")),
        "role": _text(site.get("role")),
        "location": _text(site.get("location")),
        "state": _text(site.get("state")).upper(),
        "note": _text(site.get("note")),
        "lat": round((bounds[0][0] + bounds[1][0]) / 2, 5),
        "lon": round((bounds[0][1] + bounds[1][1]) / 2, 5),
        "bounds": bounds,
        "resolution": res,
        "years": years,
        "intervals": intervals,
        "counts": counts,
        "background": int((rejected["status"] == "rejected").sum()),
        "unlabelled": int((rejected["status"] != "rejected").sum()),
        "examples": examples,
        "written": written,
    }


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


def build_webmap(cfg: WebmapConfig) -> WebmapResult:
    """Write the page of a label run.

    Args:
        cfg: Settings.

    Returns:
        Counts and the size of the folder.

    Raises:
        FileNotFoundError: If the label run has no labels or no site table.
        ValueError: If the page is projected to pass ``cfg.budget_gb``.
    """
    label_dir = Path(cfg.label_dir)
    sites_csv = Path(cfg.sites_csv) if cfg.sites_csv else label_dir / "sites.csv"
    if not (label_dir / "labels" / "labels.gpkg").is_file():
        raise FileNotFoundError(f"No labels/labels.gpkg in {label_dir}")
    if not sites_csv.is_file():
        raise FileNotFoundError(f"No site table at {sites_csv}")
    sites = pd.read_csv(sites_csv)
    events, rejected, acq = read_labels(label_dir)
    units = sorted(set(acq["unit"]))
    if cfg.units:
        units = [u for u in units if u in set(cfg.units)]
    chosen = choose_examples(events[events["unit"].isin(units)], cfg.examples)
    by_unit = {row["unit"]: row for row in sites.to_dict("records")}
    tasks = []
    for unit in units:
        tasks.append(
            {
                "cfg": cfg,
                "site": by_unit.get(unit, {"unit": unit}),
                "events": events[events["unit"] == unit],
                "rejected": rejected[rejected["unit"] == unit],
                "dates": _dates(acq[acq["unit"] == unit]),
                "examples": chosen.get(unit, []),
            }
        )
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done: list[dict[str, Any] | None] = []
    if cfg.budget_gb is not None and tasks:
        done.append(_build_place(tasks[0]))
        folder = out / "p" / tasks[0]["site"]["unit"]
        size = sum(f.stat().st_size for f in folder.glob("*")) if folder.is_dir() else 0
        unit_years = sum(len(task["dates"]) for task in tasks)
        projected = size / max(len(tasks[0]["dates"]), 1) * unit_years / 1e9
        logger.info(
            "Projected size of the page: %.1f GB for %d images (from %s)",
            projected,
            unit_years,
            tasks[0]["site"]["unit"],
        )
        if projected > cfg.budget_gb:
            raise ValueError(
                f"The page is projected at {projected:.1f} GB, above the limit of "
                f"{cfg.budget_gb:g} GB. Use a coarser --resolution (the size goes "
                "with its inverse square), fewer --units, or a larger --budget-gb."
            )
    rest = tasks[len(done) :]
    if cfg.workers > 1 and len(rest) > 1:
        with ProcessPoolExecutor(cfg.workers) as pool:
            done += list(pool.map(_build_place, rest))
    else:
        done += [_build_place(task) for task in rest]
    skipped = [
        task["site"]["unit"]
        for task, place in zip(tasks, done, strict=True)
        if place is None
    ]
    places = [place for place in done if place is not None]
    examples = [example for place in places for example in place.pop("examples")]
    images = sum(place.pop("written") for place in places)
    page = {
        "title": cfg.title,
        "built": date.today().isoformat(),
        "model_floor": MODEL_FLOOR,
        "places": places,
        "examples": examples,
    }
    (out / "page.js").write_text(
        "window.CSDV_PAGE=" + json.dumps(page) + ";", encoding="utf-8"
    )
    shutil.copyfile(_TEMPLATE, out / "index.html")
    # A web server reads the folder as another user.
    total = 0
    for path in [out, *out.rglob("*")]:
        path.chmod(0o755 if path.is_dir() else 0o644)
        if path.is_file():
            total += path.stat().st_size
    return WebmapResult(
        places=len(places),
        images=images,
        events=int(sum(sum(place["counts"].values()) for place in places)),
        megabytes=total / 1e6,
        skipped=skipped,
    )
