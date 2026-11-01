"""csdv_core.review.server: one web server for the review app and its tiles.

Solara's Starlette app serves the pages. Tile routes go in front of its
catch-all route:

- ``/tiles/ql/<unit>/<name>/<z>/<x>/<y>.png``: a quicklook, where ``name`` is
  ``naip_<year>_3m_rgb`` or ``chm_<year>_3m``
- ``/tiles/dchm/<unit>/<year0>/<year1>/<z>/<x>/<y>.png``: CHM change
- ``/tiles/run/<unit>/<onset|maxprob>/<z>/<x>/<y>.png``: run rasters
- ``/tiles/usda/<z>/<x>/<y>``: the USDA NAIP ImageServer

A tile outside a raster is answered with 204 and no content, which Leaflet
treats as an empty tile.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from csdv_core.review.config import ReviewConfig, review_config

logger = logging.getLogger(__name__)

APP_MODULE = "csdv_core.review.app"
_SAFE = re.compile(r"^[A-Za-z0-9_\-]+$")
_CACHE_HEADERS = {"Cache-Control": "max-age=86400"}

__all__ = ["build_app", "serve", "tile_routes"]


def _png(content: bytes | None) -> Response:
    if content is None:
        return Response(status_code=204)
    return Response(content, media_type="image/png", headers=_CACHE_HEADERS)


def _safe(*parts: str) -> None:
    for part in parts:
        if not _SAFE.match(part):
            raise ValueError(f"bad path part {part!r}")


def _xyz(request: Request) -> tuple[int, int, int]:
    p = request.path_params
    return int(p["z"]), int(p["x"]), int(p["y"])


def _endpoint(fn: Callable[[Request], Response]) -> Callable[[Request], Response]:
    def wrapped(request: Request) -> Response:
        try:
            return fn(request)
        except (ValueError, FileNotFoundError) as exc:
            return Response(str(exc), status_code=404)
        except Exception:  # a broken tile must not take the page down
            logger.exception("tile %s failed", request.url.path)
            return Response(status_code=500)

    return wrapped


def tile_routes(cfg: ReviewConfig | None = None) -> list[Route]:
    """Starlette routes of the review tiles."""
    from csdv_core.review import imagery, tiles

    def config() -> ReviewConfig:
        return cfg or review_config()

    def quicklook(request: Request) -> Response:
        unit, name = request.path_params["unit"], request.path_params["name"]
        _safe(unit, name)
        path = config().quicklook_root / unit / f"{name}.tif"
        if not path.exists():
            raise FileNotFoundError(path.name)
        render = tiles.chm_tile if name.startswith("chm_") else tiles.rgb_tile
        return _png(render(path, *_xyz(request)))

    def chm_change(request: Request) -> Response:
        p = request.path_params
        _safe(p["unit"])
        root = config().quicklook_root / p["unit"]
        before = root / f"chm_{int(p['y0'])}_3m.tif"
        after = root / f"chm_{int(p['y1'])}_3m.tif"
        if not (before.exists() and after.exists()):
            raise FileNotFoundError(f"{before.name} or {after.name}")
        return _png(tiles.chm_change_tile(before, after, *_xyz(request)))

    def run_raster(request: Request) -> Response:
        unit, name = request.path_params["unit"], request.path_params["name"]
        _safe(unit)
        run_dir = config().run_dir(unit)
        if name == "onset":
            return _png(tiles.onset_tile(run_dir / "onset_year_u8.tif", *_xyz(request)))
        if name == "maxprob":
            path = run_dir / "max_probability_u8.tif"
            return _png(tiles.probability_tile(path, *_xyz(request)))
        raise ValueError(name)

    def usda(request: Request) -> Response:
        content, media = imagery.usda_tile(*_xyz(request))
        return Response(content, media_type=media, headers=_CACHE_HEADERS)

    xyz = "{z:int}/{x:int}/{y:int}"
    return [
        Route(f"/tiles/ql/{{unit}}/{{name}}/{xyz}.png", _endpoint(quicklook)),
        Route(
            f"/tiles/dchm/{{unit}}/{{y0:int}}/{{y1:int}}/{xyz}.png",
            _endpoint(chm_change),
        ),
        Route(f"/tiles/run/{{unit}}/{{name}}/{xyz}.png", _endpoint(run_raster)),
        Route(f"/tiles/usda/{xyz}", _endpoint(usda)),
    ]


def build_app(cfg: ReviewConfig):
    """Solara's Starlette app with the tile routes in front."""
    cfg.to_env()
    os.environ["SOLARA_APP"] = APP_MODULE
    # Keep GDAL's block cache small: WSL has little memory to spare.
    os.environ.setdefault("GDAL_CACHEMAX", "256")
    import solara.server.starlette as solara_starlette

    app = solara_starlette.app
    app.router.routes[0:0] = tile_routes(cfg)
    return app


def serve(cfg: ReviewConfig, host: str = "localhost", port: int = 8765) -> None:
    """Run the review app until interrupted."""
    import uvicorn

    for name in ("solara", "reacton"):
        logging.getLogger(name).setLevel(logging.WARNING)
    app = build_app(cfg)
    logger.info("Review app on http://%s:%d (store %s)", host, port, cfg.store_path)
    uvicorn.run(app, host=host, port=port, log_level="warning")
