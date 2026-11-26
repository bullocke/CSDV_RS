"""csdv_core.view.server: the web server of the map viewer.

One Starlette app serves the page (``page/``), a small JSON API and the map
tiles:

- ``/api/catalog``: the sites, the runs and the legend colours
- ``/api/site/<unit>``: the 0.6 m NAIP tile source of every year
- ``/api/point?unit=&lon=&lat=``: what every run of a site says at a position
- ``/api/labels/<unit>``: the label events and reviewed areas, as GeoJSON
- ``/api/notes`` and ``/api/notes/<id>``: the comment points
- ``/tiles/map/<run>/<layer>/<interval>/<percent>/<mask>/<z>/<x>/<y>.png``:
  a derived layer of a run (:mod:`csdv_core.view.tiles`)

- ``/tiles/naip/<unit>/<year>/<z>/<x>/<y>``: the local 3 m NAIP quicklook
- ``/tiles/chm/<unit>/<year>/<z>/<x>/<y>.png``: the canopy height quicklook
- ``/tiles/dchm/<unit>/<year0>/<year1>/<z>/<x>/<y>.png``: height change

The 0.6 m NAIP tiles (``/tiles/pc`` and ``/tiles/usda``) are the routes of the
review server (:func:`csdv_core.review.server.tile_routes`), mounted as they
are.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from csdv_core.view.catalog import Catalog, build_catalog
from csdv_core.view.notes import TAGS, NoteStore

logger = logging.getLogger(__name__)

PAGE_DIR = Path(__file__).parent / "page"
_SAFE = re.compile(r"^[A-Za-z0-9_\-]+$")
_TILE_HEADERS = {"Cache-Control": "max-age=86400"}
_NO_STORE = {"Cache-Control": "no-store"}

__all__ = ["ViewConfig", "build_app", "serve"]


@dataclass(frozen=True)
class ViewConfig:
    """Where the viewer reads maps and keeps notes."""

    export_dir: Path
    notes_path: Path
    sites_csv: Path | None = None
    labels_path: Path | None = None

    @property
    def labels(self) -> Path:
        if self.labels_path is not None:
            return self.labels_path
        return self.export_dir / "label" / "scale_test" / "labels" / "labels.gpkg"

    def labels_of(self, unit: str) -> Path:
        """Label file of a unit: places of round 2 (``p_r2_``) have their own."""
        if self.labels_path is None and unit.startswith("p_r2_"):
            return self.export_dir / "label" / "round2" / "labels" / "labels.gpkg"
        return self.labels


def _safe(*parts: str) -> None:
    for part in parts:
        if not _SAFE.match(part):
            raise ValueError(f"bad path part {part!r}")


def _endpoint(fn: Callable[[Request], Response]) -> Callable[[Request], Response]:
    def wrapped(request: Request) -> Response:
        try:
            return fn(request)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        except Exception:  # one broken request must not take the page down
            logger.exception("%s failed", request.url.path)
            return JSONResponse({"error": "server error"}, status_code=500)

    return wrapped


def _label_features(path: Path, unit: str) -> dict:
    """Events and reviewed areas of a unit in EPSG:4326."""
    import geopandas as gpd

    wanted = {
        "events": [
            "event_id",
            "agent",
            "change_type",
            "severity",
            "loss_intervals",
            "last_pre_acq",
            "first_post_acq",
            "confidence",
        ],
        "reviewed_areas": ["interval", "finding", "basis"],
    }
    features: list[dict] = []
    for layer, columns in wanted.items():
        frame = gpd.read_file(path, layer=layer, where=f"unit = '{unit}'")
        if frame.empty:
            continue
        if "candidate_ids" in frame.columns:
            # An event is stored once per window that covers it.
            frame = frame.drop_duplicates("candidate_ids")
        frame = frame[[c for c in columns if c in frame.columns] + ["geometry"]]
        frame = frame.to_crs(4326)
        frame["geometry"] = frame.geometry.simplify(0.00001)
        for column in frame.columns.drop("geometry"):
            frame[column] = frame[column].astype(object).where(frame[column].notna())
        frame.insert(0, "layer", layer)
        features.extend(json.loads(frame.to_json(default=str))["features"])
    return {"type": "FeatureCollection", "features": features}


def build_app(cfg: ViewConfig) -> Starlette:
    """The viewer's Starlette app."""
    from csdv_core.review.config import ReviewConfig
    from csdv_core.review.imagery import default_pc_cache, naip_sources
    from csdv_core.review.server import tile_routes
    from csdv_core.view import tiles

    # Keep GDAL's block cache small: WSL has little memory to spare.
    os.environ.setdefault("GDAL_CACHEMAX", "256")
    notes = NoteStore(cfg.notes_path)
    state: dict[str, Catalog] = {}

    def catalog(refresh: bool = False) -> Catalog:
        if refresh or "catalog" not in state:
            state["catalog"] = build_catalog(cfg.export_dir, cfg.sites_csv)
        return state["catalog"]

    def index(request: Request) -> Response:
        return FileResponse(PAGE_DIR / "index.html", headers=_NO_STORE)

    def get_catalog(request: Request) -> Response:
        body = catalog(refresh=True).to_json()
        body["legend"] = tiles.legend()
        body["tags"] = list(TAGS)
        body["notes_path"] = str(cfg.notes_path)
        body["labels"] = cfg.labels.exists()
        return JSONResponse(body, headers=_NO_STORE)

    def get_site(request: Request) -> Response:
        unit = request.path_params["unit"]
        _safe(unit)
        cat = catalog()
        site = cat.site(unit)
        record_path = cat.unit_dir(unit) / "unit.json"
        record = json.loads(record_path.read_text()) if record_path.exists() else {}
        years = [int(y) for y in record.get("years", {})]
        sources = naip_sources(unit, years, record, default_pc_cache())
        naip = {
            str(year): {
                "url": source.url,
                "kind": source.kind,
                "max_native_zoom": source.max_native_zoom,
            }
            for year, source in sources.items()
            if source.kind != "quicklook"
        }
        return JSONResponse({"unit": site.unit, "naip": naip}, headers=_NO_STORE)

    def get_point(request: Request) -> Response:
        unit = request.query_params["unit"]
        _safe(unit)
        lon = float(request.query_params["lon"])
        lat = float(request.query_params["lat"])
        cat = catalog()
        runs = {
            name: tiles.point_values(cat.run(name), lon, lat)
            for name in cat.site(unit).runs
        }
        body = {"unit": unit, "lon": lon, "lat": lat, "runs": runs}
        return JSONResponse(body, headers=_NO_STORE)

    @lru_cache(maxsize=32)
    def labels_json(unit: str, version: float) -> bytes:
        return json.dumps(_label_features(cfg.labels_of(unit), unit)).encode()

    def get_labels(request: Request) -> Response:
        unit = request.path_params["unit"]
        _safe(unit)
        path = cfg.labels_of(unit)
        if not path.exists():
            raise FileNotFoundError(path.name)
        body = labels_json(unit, path.stat().st_mtime)
        return Response(body, media_type="application/json", headers=_NO_STORE)

    def get_notes(request: Request) -> Response:
        return JSONResponse(notes.collection(), headers=_NO_STORE)

    async def put_note(request: Request) -> Response:
        note_id = request.path_params["note_id"]
        try:
            _safe(note_id)
            body = await request.json()
            feature = notes.put(note_id, float(body["lon"]), float(body["lat"]), body)
        except (ValueError, KeyError, TypeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        if feature is None:
            return JSONResponse({"deleted": note_id})
        return JSONResponse(feature)

    def delete_note(request: Request) -> Response:
        note_id = request.path_params["note_id"]
        _safe(note_id)
        return JSONResponse({"deleted": note_id, "existed": notes.delete(note_id)})

    def map_tile(request: Request) -> Response:
        p = request.path_params
        _safe(p["run"], p["layer"], p["interval"])
        layer = tiles.layer_of(
            catalog().run(p["run"]),
            p["layer"],
            p["interval"],
            percent=int(p["percent"]),
            forest=p["mask"] == "forest",
        )
        return _png(tiles.render_tile(layer, *_xyz(request)))

    def quicklook(unit: str, name: str) -> Path:
        _safe(unit)
        path = cfg.export_dir / "quicklooks" / unit / name
        if not path.exists():
            raise FileNotFoundError(name)
        return path

    def _xyz(request: Request) -> tuple[int, int, int]:
        p = request.path_params
        return int(p["z"]), int(p["x"]), int(p["y"])

    def naip_tile(request: Request) -> Response:
        p = request.path_params
        path = quicklook(p["unit"], f"naip_{int(p['year'])}_3m_rgb.tif")
        got = tiles.image_tile(path, *_xyz(request))
        if got is None:
            return Response(status_code=204, headers=_TILE_HEADERS)
        return Response(got[0], media_type=got[1], headers=_TILE_HEADERS)

    def _png(content: bytes | None) -> Response:
        if content is None:
            return Response(status_code=204, headers=_TILE_HEADERS)
        return Response(content, media_type="image/png", headers=_TILE_HEADERS)

    def chm_tile(request: Request) -> Response:
        p = request.path_params
        path = quicklook(p["unit"], f"chm_{int(p['year'])}_3m.tif")
        return _png(tiles.chm_tile(path, *_xyz(request)))

    def chm_change_tile(request: Request) -> Response:
        p = request.path_params
        before = quicklook(p["unit"], f"chm_{int(p['y0'])}_3m.tif")
        after = quicklook(p["unit"], f"chm_{int(p['y1'])}_3m.tif")
        return _png(tiles.chm_change_tile(before, after, *_xyz(request)))

    review_cfg = ReviewConfig(
        export_dir=cfg.export_dir,
        run_tag="default",
        store_path=cfg.notes_path,  # not used by the tile routes
        interpreter="",
    )
    xyz = "{z:int}/{x:int}/{y:int}"
    remote = ("/tiles/pc/", "/tiles/usda/")
    routes = [
        Route("/", index),
        Route("/api/catalog", _endpoint(get_catalog)),
        Route("/api/site/{unit}", _endpoint(get_site)),
        Route("/api/point", _endpoint(get_point)),
        Route("/api/labels/{unit}", _endpoint(get_labels)),
        Route("/api/notes", _endpoint(get_notes)),
        Route("/api/notes/{note_id}", put_note, methods=["PUT"]),
        Route("/api/notes/{note_id}", _endpoint(delete_note), methods=["DELETE"]),
        Route(
            "/tiles/map/{run}/{layer}/{interval}/{percent:int}/{mask}/" f"{xyz}.png",
            _endpoint(map_tile),
        ),
        Route(f"/tiles/naip/{{unit}}/{{year:int}}/{xyz}", _endpoint(naip_tile)),
        Route(f"/tiles/chm/{{unit}}/{{year:int}}/{xyz}.png", _endpoint(chm_tile)),
        Route(
            f"/tiles/dchm/{{unit}}/{{y0:int}}/{{y1:int}}/{xyz}.png",
            _endpoint(chm_change_tile),
        ),
        *(r for r in tile_routes(review_cfg) if r.path.startswith(remote)),
        Mount("/static", StaticFiles(directory=PAGE_DIR), name="static"),
    ]
    return Starlette(routes=routes)


def serve(cfg: ViewConfig, host: str = "localhost", port: int = 8767) -> None:
    """Run the viewer until interrupted."""
    import uvicorn

    app = build_app(cfg)
    logger.info("Map viewer on http://%s:%d (notes %s)", host, port, cfg.notes_path)
    uvicorn.run(app, host=host, port=port, log_level="warning")
