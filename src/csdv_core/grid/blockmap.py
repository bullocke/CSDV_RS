"""csdv_core.grid.blockmap: one web page for a block of adjoining quads.

Inputs: batch files of the block (``config/grid/<name>.csv``), the NAIP
mosaics of its units (``<data root>/naip/<unit>/<year>/*.tif``, or the 3 m
quicklooks of the exports when the mosaics are not at hand) and the pair
model's maps in the exports (``run_layers/<unit>/<tag>/model/``).

Output: a static folder for a web server:

- ``index.html`` and ``page.js``: a Leaflet page with one image layer per
  NAIP year, and per interval the model's probability of loss and the cause
  it names where the probability reaches the threshold.
- ``t/<layer>/<z>/<x>/<y>``: map tiles of 512 pixels. A block is one mosaic,
  whatever the UTM zones of its quads.
- ``quads.js``: the outline of every quad with its image dates and the model
  that mapped it.

Every layer is warped to one Web Mercator mosaic with ``gdalwarp`` and cut
with ``gdal2tiles.py``, both of the GDAL that the environment holds. The NAIP
tiles end at zoom 14, which is 3.3 to 4.1 m on the ground between 45 and 31
degrees north. The page shows sharper imagery of a tile service underneath
when the reader zooms further.

The page keeps the reader's notes in the browser and writes them as GeoJSON.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from csdv_core.grid.units import Unit, read_batch

logger = logging.getLogger(__name__)

__all__ = ["BlockmapConfig", "BlockmapResult", "build_blockmap", "tile_resolution"]

#: Probability (percent) below which the probability layer is transparent.
PROB_FLOOR = 20
_PROB_RGB = (255, 43, 214)
#: Pixels of a tile side. Fewer and larger files than the usual 256.
TILE_PX = 512
_WORLD_M = 40075016.685578486
_TEMPLATE = Path(__file__).with_name("blockmap.html")


@dataclass
class BlockmapConfig:
    """Settings of one page.

    Attributes:
        batches: Batch files whose units make up the block.
        out_dir: Folder of the page.
        export_dir: Exports folder with ``run_layers`` and ``quicklooks``.
        data_root: Data folder with ``naip/<unit>/<year>`` mosaics, or None
            to draw the quicklooks.
        title: Heading of the page.
        run_tag: Tag of the disturbance run that holds the model's map.
        naip_zoom: Deepest zoom of the image tiles.
        model_zoom: Deepest zoom of the model's tiles.
        min_zoom: Shallowest zoom of every layer.
        threshold: Probability (percent) from which a cause is drawn.
        years: Only these image years, or None for all.
        quality: WebP quality of the image tiles.
        workers: Processes of the tile cutter and threads of the warp.
        budget_gb: Largest size of the page. The first image year is cut,
            the page is projected from it, and the build stops there when
            the projection is larger. None for no limit.
        tmp_dir: Folder for the mosaics, removed at the end. Default
            ``_tmp`` in the page folder.
        overwrite: Cut layers again whose tiles exist.
        overwrite_model: Cut the layers of the model again and keep the
            images, for a block that was mapped again.
    """

    batches: tuple[Path, ...]
    out_dir: Path
    export_dir: Path
    data_root: Path | None = None
    title: str = "Disturbance map"
    run_tag: str = "default"
    naip_zoom: int = 14
    model_zoom: int = 13
    min_zoom: int = 8
    threshold: int = 90
    years: tuple[int, ...] | None = None
    quality: int = 80
    workers: int = 4
    budget_gb: float | None = None
    tmp_dir: Path | None = None
    overwrite: bool = False
    overwrite_model: bool = False


@dataclass
class BlockmapResult:
    """What a build wrote."""

    quads: int
    years: list[int]
    intervals: list[str]
    megabytes: float
    without_map: list[str] = field(default_factory=list)


def tile_resolution(zoom: int) -> float:
    """Web Mercator metres per pixel of a 512 pixel tile at ``zoom``."""
    return _WORLD_M / (TILE_PX * 2**zoom)


def _run(command: list[str]) -> None:
    logger.debug("%s", " ".join(command))
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(
            f"{command[0]} failed ({done.returncode}): {done.stderr.strip()[-2000:]}"
        )


def _tool(name: str) -> str:
    """A GDAL program of this environment."""
    beside = Path(sys.executable).with_name(name)
    found = str(beside) if beside.exists() else shutil.which(name)
    if not found:
        raise RuntimeError(f"{name} was not found. It comes with GDAL.")
    return found


def _mosaic(
    sources: list[Path],
    out: Path,
    zoom: int,
    *,
    resampling: str,
    bands: tuple[int, ...] | None,
    threads: int,
) -> None:
    """Warp rasters of any CRS to one Web Mercator raster on the tile grid."""
    res = repr(tile_resolution(zoom))
    command = [_tool("gdalwarp"), "-q", "-overwrite", "-t_srs", "EPSG:3857"]
    command += ["-tr", res, res, "-tap", "-r", resampling, "-multi"]
    command += ["-wo", f"NUM_THREADS={threads}", "-ovr", "AUTO"]
    for band in bands or ():
        command += ["-b", str(band)]
    if bands is not None:
        command.append("-dstalpha")
    command += ["-co", "COMPRESS=DEFLATE", "-co", "TILED=YES", "-co", "BIGTIFF=YES"]
    _run([*command, *map(str, sources), str(out)])


def _tiles(
    mosaic: Path,
    out: Path,
    zooms: tuple[int, int],
    *,
    resampling: str,
    driver: str,
    quality: int,
    workers: int,
) -> None:
    """Cut a Web Mercator raster with alpha into XYZ tiles.

    The tiles are cut into ``<out>.part`` and the folder is renamed when all
    are written, so a folder with the name of a layer is a finished layer. A
    job that ran out of time leaves a ``.part`` folder, which the next run
    cuts again.
    """
    part = out.with_name(out.name + ".part")
    command = [sys.executable, _tool("gdal2tiles.py"), "--xyz", "-q", "-x"]
    command += ["-z", f"{zooms[0]}-{zooms[1]}", f"--tilesize={TILE_PX}"]
    command += ["-r", resampling, "-w", "none", f"--processes={max(1, workers)}"]
    command += [f"--tiledriver={driver}"]
    if driver == "WEBP":
        command.append(f"--webp-quality={quality}")
    shutil.rmtree(part, ignore_errors=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    _run([*command, str(mosaic), str(part)])
    shutil.rmtree(out, ignore_errors=True)
    part.rename(out)


def _naip_source(cfg: BlockmapConfig, unit: str, year: int) -> Path | None:
    if cfg.data_root is not None:
        tifs = sorted((Path(cfg.data_root) / "naip" / unit / str(year)).glob("*.tif"))
        if len(tifs) == 1:
            return tifs[0]
    looks = sorted(
        (Path(cfg.export_dir) / "quicklooks" / unit).glob(f"naip_{year}_*_rgb.tif")
    )
    return looks[0] if looks else None


def _naip_years(cfg: BlockmapConfig, unit: str) -> set[int]:
    years: set[int] = set()
    if cfg.data_root is not None:
        root = Path(cfg.data_root) / "naip" / unit
        years |= {int(p.name) for p in root.glob("*") if p.name.isdigit()}
    for look in (Path(cfg.export_dir) / "quicklooks" / unit).glob("naip_*_rgb.tif"):
        years.add(int(look.name.split("_")[1]))
    return years


def _model_dir(cfg: BlockmapConfig, unit: str) -> Path:
    return Path(cfg.export_dir) / "run_layers" / unit / cfg.run_tag / "model"


def _interval_key(description: str) -> str:
    """``2012-06-07_2014-07-05`` as ``2012_2014``."""
    first, second = description.split("_")
    return f"{first[:4]}_{second[:4]}"


def _render_model(
    cfg: BlockmapConfig,
    unit: str,
    tmp: Path,
    bbox_4326: tuple[float, float, float, float] | None = None,
) -> tuple[dict[str, dict[str, Path]], dict[str, str]]:
    """RGBA rasters of one unit's map: per interval a probability and a cause.

    With ``bbox_4326`` only the cells whose centre lies in that box are
    drawn. The maps of adjoining quads overlap by a few hundred metres, and
    in the outer 50 m of a mosaic the model flags about twice as much as the
    neighbour's map does on the same ground. The patch layer makes the same
    cut (:func:`csdv_core.pairmodel.patches.owned_cells`).

    Returns:
        ``{interval key: {"prob": path, "cause": path}}`` and the image date
        of every year the map names.
    """
    import rasterio

    from csdv_core.pairmodel.predict import AGENT_COLOURS

    folder = _model_dir(cfg, unit)
    prob_path = folder / f"prob_forest_{unit}.tif"
    if not prob_path.is_file():
        prob_path = folder / f"prob_{unit}.tif"
    if not prob_path.is_file():
        return {}, {}
    agent_path = folder / f"agent_{unit}.tif"
    written: dict[str, dict[str, Path]] = {}
    dates: dict[str, str] = {}
    with rasterio.open(prob_path) as src:
        profile = {
            "driver": "GTiff",
            "height": src.height,
            "width": src.width,
            "count": 4,
            "dtype": "uint8",
            "crs": src.crs,
            "transform": src.transform,
            "compress": "deflate",
        }
        names = list(src.descriptions)
        prob = src.read()
        if bbox_4326 is not None:
            from csdv_core.pairmodel.patches import UnitMap, owned_cells

            own = owned_cells(
                UnitMap(
                    unit, None, None, src.transform, src.crs.to_epsg(), [],
                    bbox_4326=bbox_4326, shape=(src.height, src.width),
                )
            )  # fmt: skip
            prob[:, ~own] = 255
    agent = None
    if agent_path.is_file():
        with rasterio.open(agent_path) as src:
            agent = src.read()
    for band, name in enumerate(names):
        if not name or "_" not in name:
            continue
        first, second = name.split("_")
        dates[first[:4]], dates[second[:4]] = first, second
        key = _interval_key(name)
        share = np.where(prob[band] == 255, 0, prob[band]).astype(np.float32)
        alpha = np.clip((share - PROB_FLOOR) / (100 - PROB_FLOOR), 0, 1)
        alpha = np.where(share >= PROB_FLOOR, 70 + alpha * 170, 0).astype(np.uint8)
        rgba = np.zeros((4, *share.shape), dtype=np.uint8)
        for i, value in enumerate(_PROB_RGB):
            rgba[i] = value
        rgba[3] = alpha
        paths = {"prob": tmp / f"{unit}_prob_{key}.tif"}
        with rasterio.open(paths["prob"], "w", **profile) as dst:
            dst.write(rgba)
        if agent is not None:
            flagged = share >= cfg.threshold
            cause = np.zeros((4, *share.shape), dtype=np.uint8)
            for code, colour in AGENT_COLOURS.items():
                hit = flagged & (agent[band] == code)
                for i in range(3):
                    cause[i][hit] = colour[i]
                cause[3][hit] = 235
            paths["cause"] = tmp / f"{unit}_cause_{key}.tif"
            with rasterio.open(paths["cause"], "w", **profile) as dst:
                dst.write(cause)
        written[key] = paths
    return written, dates


def _folder_bytes(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


def _quads(units: list[Unit], info: dict[str, dict[str, Any]]) -> dict[str, Any]:
    features = []
    for unit in units:
        west, south, east, north = unit.bbox_4326
        ring = [
            [west, south],
            [east, south],
            [east, north],
            [west, north],
            [west, south],
        ]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {"unit": unit.unit_id, **info.get(unit.unit_id, {})},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def build_blockmap(cfg: BlockmapConfig) -> BlockmapResult:
    """Build the page of a block.

    Raises:
        ValueError: If the block has no unit, or none of its units has an
            image.
        RuntimeError: If a GDAL program fails, or the page is projected
            above ``budget_gb``.
    """
    from csdv_core.pairmodel.data import AGENT_NAMES
    from csdv_core.pairmodel.predict import AGENT_COLOURS

    units: list[Unit] = []
    for batch in cfg.batches:
        units += read_batch(batch)
    seen: set[str] = set()
    units = [u for u in units if not (u.unit_id in seen or seen.add(u.unit_id))]
    if not units:
        raise ValueError("The batch files name no unit")
    out = Path(cfg.out_dir)
    tiles = out / "t"
    tmp = Path(cfg.tmp_dir) if cfg.tmp_dir else out / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    years = sorted(set().union(*(_naip_years(cfg, u.unit_id) for u in units)))
    if cfg.years is not None:
        years = [y for y in years if y in set(cfg.years)]
    if not years:
        raise ValueError("None of the units has a NAIP mosaic or quicklook")

    # The model's maps first: they are small, and they give the image dates.
    info: dict[str, dict[str, Any]] = {}
    by_interval: dict[str, dict[str, list[Path]]] = {}
    without_map: list[str] = []
    for unit in units:
        box = unit.bbox_4326 if unit.kind == "cell" else None
        rendered, dates = _render_model(cfg, unit.unit_id, tmp, box)
        note = _model_dir(cfg, unit.unit_id) / "checkpoint.txt"
        info[unit.unit_id] = {
            "dates": dates,
            "model": note.read_text(encoding="utf-8").strip() if note.is_file() else "",
        }
        if not rendered:
            without_map.append(unit.unit_id)
        for key, paths in rendered.items():
            for kind, path in paths.items():
                by_interval.setdefault(key, {}).setdefault(kind, []).append(path)
    intervals = sorted(by_interval)
    kinds = sorted({kind for layers in by_interval.values() for kind in layers})
    for key in intervals:
        for kind, sources in by_interval[key].items():
            target = tiles / f"{kind}_{key}"
            again = cfg.overwrite or cfg.overwrite_model
            if target.is_dir() and not again:
                continue
            mosaic = tmp / f"{kind}_{key}.tif"
            started = time.monotonic()
            _mosaic(
                sources,
                mosaic,
                cfg.model_zoom,
                resampling="near",
                bands=None,
                threads=cfg.workers,
            )
            _tiles(
                mosaic,
                target,
                (cfg.min_zoom, cfg.model_zoom),
                resampling="near",
                driver="PNG",
                quality=cfg.quality,
                workers=cfg.workers,
            )
            mosaic.unlink()
            logger.info(
                "Model layer %s of %s cut in %.0f s",
                kind,
                key,
                time.monotonic() - started,
            )

    drawn_years: list[int] = []
    for n, year in enumerate(years):
        target = tiles / f"naip_{year}"
        sources = [s for u in units if (s := _naip_source(cfg, u.unit_id, year))]
        if not sources:
            continue
        drawn_years.append(year)
        if target.is_dir() and not cfg.overwrite:
            continue
        mosaic = tmp / f"naip_{year}.tif"
        started = time.monotonic()
        _mosaic(
            sources,
            mosaic,
            cfg.naip_zoom,
            resampling="average",
            bands=(1, 2, 3),
            threads=cfg.workers,
        )
        warped = time.monotonic()
        _tiles(
            mosaic,
            target,
            (cfg.min_zoom, cfg.naip_zoom),
            resampling="average",
            driver="WEBP",
            quality=cfg.quality,
            workers=cfg.workers,
        )
        mosaic.unlink()
        size = _folder_bytes(target)
        logger.info(
            "NAIP %d: %d quads, %.0f MB of tiles, mosaic %.0f s, tiles %.0f s",
            year,
            len(sources),
            size / 1e6,
            warped - started,
            time.monotonic() - warped,
        )
        if n == 0 and cfg.budget_gb is not None:
            projected = (size * len(years) + _folder_bytes(tiles) - size) / 1e9
            if projected > cfg.budget_gb:
                raise RuntimeError(
                    f"The page is projected at {projected:.1f} GB, above the limit "
                    f"of {cfg.budget_gb:g} GB. Lower --naip-zoom by one for a "
                    "quarter of the size, name fewer --years, or raise --budget-gb."
                )

    west = min(u.bbox_4326[0] for u in units)
    south = min(u.bbox_4326[1] for u in units)
    east = max(u.bbox_4326[2] for u in units)
    north = max(u.bbox_4326[3] for u in units)
    lat = math.radians((south + north) / 2)
    page = {
        "title": cfg.title,
        "bounds": [[south, west], [north, east]],
        "years": drawn_years,
        "intervals": intervals,
        "kinds": kinds,
        "tile_px": TILE_PX,
        "min_zoom": cfg.min_zoom,
        "naip_zoom": cfg.naip_zoom,
        "model_zoom": cfg.model_zoom,
        "naip_m": round(tile_resolution(cfg.naip_zoom) * math.cos(lat), 1),
        "threshold": cfg.threshold,
        "prob_floor": PROB_FLOOR,
        "prob_rgb": list(_PROB_RGB),
        "causes": [
            {"name": AGENT_NAMES[code - 1], "rgb": list(colour[:3])}
            for code, colour in AGENT_COLOURS.items()
        ],
        "models": sorted({v["model"] for v in info.values() if v["model"]}),
        "quads": len(units),
        "without_map": without_map,
    }
    (out / "page.js").write_text(
        "window.CSDV_BLOCK=" + json.dumps(page) + ";", encoding="utf-8"
    )
    (out / "quads.js").write_text(
        "window.CSDV_QUADS=" + json.dumps(_quads(units, info)) + ";", encoding="utf-8"
    )
    shutil.copyfile(_TEMPLATE, out / "index.html")
    shutil.rmtree(tmp, ignore_errors=True)
    # A web server reads the folder as another user.
    total = 0
    for path in [out, *out.rglob("*")]:
        path.chmod(0o755 if path.is_dir() else 0o644)
        if path.is_file():
            total += path.stat().st_size
    return BlockmapResult(
        quads=len(units),
        years=drawn_years,
        intervals=intervals,
        megabytes=total / 1e6,
        without_map=without_map,
    )
