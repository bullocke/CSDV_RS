"""csdv_core.review.imagery: 0.6 m NAIP tile sources for the review map.

Each NAIP year of a unit is shown from the source the run downloaded it from
(``unit.json``):

``pc`` (Planetary Computer)
    A mosaic of the exact quads of that program year, found with
    :func:`csdv_core.download.naip_pc.search_naip_items` and registered with
    the Planetary Computer data API. The quad IDs are cached on disk, so the
    STAC search runs once per unit and year. A registered mosaic expires after
    about a day (its ID carries the expiry time, and later requests get 410),
    so the browser never sees it: the review server fetches the tiles
    (:func:`pc_tile`) and registers the mosaic again when it is about to
    expire or has expired.
``rest`` (USDA FPAC ImageServer)
    The service has no WMS or tile cache, so the review server turns each map
    tile into an ``exportImage`` request (:func:`usda_tile`). The service keeps
    only the newest collection of an area, which is the year the run took from
    it unless USDA has published a newer one since.

When a source cannot be reached, the year falls back to the 3 m quicklook of
the export, and the layer label says so.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

PC_DATA_API = "https://planetarycomputer.microsoft.com/api/data/v1"
#: Tiles beyond this zoom are upsampled by the browser.
NAIP_MAX_NATIVE_ZOOM = 19
TIMEOUT_S = 30

#: Re-register a mosaic this long before its stated expiry.
EXPIRY_MARGIN_S = 600

__all__ = [
    "ImagerySource",
    "default_pc_cache",
    "naip_sources",
    "pc_tile",
    "quicklook_source",
    "usda_tile",
]


@dataclass(frozen=True)
class ImagerySource:
    """An XYZ tile source for one NAIP year."""

    year: int
    kind: str  # "pc", "usda" or "quicklook"
    url: str
    max_native_zoom: int = NAIP_MAX_NATIVE_ZOOM

    @property
    def label(self) -> str:
        suffix = {"pc": "0.6 m", "usda": "0.6 m USDA", "quicklook": "3 m quicklook"}
        return f"{self.year} ({suffix[self.kind]})"


def quicklook_source(unit: str, year: int) -> ImagerySource:
    """The 3 m quicklook of a year, served by the review server."""
    return ImagerySource(
        year, "quicklook", f"/tiles/ql/{unit}/naip_{year}_3m_rgb/{{z}}/{{x}}/{{y}}.png"
    )


class _PcItemCache:
    """Planetary Computer item IDs per unit and year, kept in a JSON file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self.data: dict = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}

    def get(self, unit: str, year: int, bbox: tuple[float, ...]) -> list[str]:
        ids = self.data.get(unit, {}).get(str(year))
        if ids:
            return ids
        from csdv_core.download.naip_pc import search_naip_items

        ids = [item.id for item in search_naip_items(tuple(bbox), year=year)]
        self.data.setdefault(unit, {})[str(year)] = ids
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        return ids


def default_pc_cache() -> Path:
    """``<cache_root>/review/pc_items.json``."""
    from csdv_core.io.paths import project_paths

    return project_paths().cache_root / "review" / "pc_items.json"


class _Mosaics:
    """Registered mosaics by item IDs, renewed before they expire."""

    def __init__(self) -> None:
        self._ids: dict[tuple[str, ...], tuple[str, float]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _expiry(search_id: str) -> float:
        # IDs look like pc1.<expiry epoch seconds>.<hash>.
        try:
            return float(search_id.split(".")[1])
        except (IndexError, ValueError):
            return time.time() + 3600

    def search_id(self, ids: tuple[str, ...], *, renew: bool = False) -> str:
        with self._lock:
            known = self._ids.get(ids)
            if known and not renew and known[1] - EXPIRY_MARGIN_S > time.time():
                return known[0]
            body = {
                "collections": ["naip"],
                "filter-lang": "cql2-json",
                "filter": {"op": "in", "args": [{"property": "id"}, list(ids)]},
            }
            r = requests.post(
                f"{PC_DATA_API}/mosaic/register", json=body, timeout=TIMEOUT_S
            )
            r.raise_for_status()
            search_id = r.json()["searchid"]
            self._ids[ids] = (search_id, self._expiry(search_id))
            logger.info("Registered NAIP mosaic %s (%d quads)", search_id, len(ids))
            return search_id


_MOSAICS = _Mosaics()


def _pc_item_ids(unit: str, year: int, cache_path: Path | None = None) -> tuple:
    ids = (
        _PcItemCache(cache_path or default_pc_cache()).data.get(unit, {}).get(str(year))
    )
    if not ids:
        raise KeyError(f"no Planetary Computer quads cached for {unit} {year}")
    return tuple(ids)


@lru_cache(maxsize=2048)
def pc_tile(unit: str, year: int, z: int, x: int, y: int) -> tuple[bytes, str] | None:
    """One tile of a unit's NAIP mosaic for a year, or None outside the quads.

    The quad IDs come from the item cache written by :func:`naip_sources`.
    """
    ids = _pc_item_ids(unit, year)
    for attempt in range(2):
        search_id = _MOSAICS.search_id(ids, renew=attempt > 0)
        url = f"{PC_DATA_API}/mosaic/{search_id}/tiles/WebMercatorQuad/{z}/{x}/{y}"
        params = {"assets": "image", "asset_bidx": "image|1,2,3", "collection": "naip"}
        r = requests.get(url, params=params, timeout=TIMEOUT_S)
        if r.status_code == 410:  # expired after all: register again
            continue
        if r.status_code in (204, 404):
            return None
        r.raise_for_status()
        return r.content, r.headers.get("content-type", "image/jpeg")
    raise RuntimeError(f"NAIP mosaic for {unit} {year} keeps expiring")


def naip_sources(
    unit: str,
    years: list[int],
    unit_record: dict,
    cache_path: Path,
) -> dict[int, ImagerySource]:
    """Tile source of every year of a unit, falling back to the quicklooks.

    Args:
        unit: Unit ID.
        years: Years of the run.
        unit_record: The unit's ``unit.json`` (``years`` with ``source``, and
            ``bbox_4326``).
        cache_path: JSON file of cached Planetary Computer item IDs.
    """
    cache = _PcItemCache(cache_path)
    bbox = unit_record.get("bbox_4326")
    meta = unit_record.get("years", {})
    out: dict[int, ImagerySource] = {}
    for year in years:
        source = meta.get(str(year), {}).get("source", "pc")
        try:
            if source == "rest":
                out[year] = ImagerySource(year, "usda", "/tiles/usda/{z}/{x}/{y}")
            elif bbox is not None:
                cache.get(unit, year, tuple(bbox))
                url = f"/tiles/pc/{unit}/{year}/{{z}}/{{x}}/{{y}}"
                out[year] = ImagerySource(year, "pc", url)
            else:
                out[year] = quicklook_source(unit, year)
        except (requests.RequestException, ValueError, KeyError) as exc:
            logger.warning("%s %d: %s, using the 3 m quicklook", unit, year, exc)
            out[year] = quicklook_source(unit, year)
    return out


def tile_bounds_3857(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Web Mercator bounds of an XYZ tile."""
    r = 6378137.0
    size = 2 * math.pi * r / 2**z
    left = -math.pi * r + x * size
    top = math.pi * r - y * size
    return left, top - size, left + size, top


@lru_cache(maxsize=4096)
def usda_tile(z: int, x: int, y: int) -> tuple[bytes, str]:
    """One 256 px tile of the USDA NAIP ImageServer, through ``exportImage``.

    Returns:
        The image (JPEG, or PNG where the tile has no-data areas) and its
        media type.
    """
    from csdv_core.download.naip_rest import NAIP_REST_URL

    params = {
        "bbox": ",".join(f"{v:.3f}" for v in tile_bounds_3857(z, x, y)),
        "bboxSR": 3857,
        "imageSR": 3857,
        "size": "256,256",
        "format": "jpgpng",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }
    r = requests.get(f"{NAIP_REST_URL}/exportImage", params=params, timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.content, r.headers.get("content-type", "image/jpeg")
