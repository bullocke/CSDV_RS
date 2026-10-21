"""csdv_core.download.naip_rest — NAIP from the USDA FPAC image service.

The Planetary Computer archive lags the NAIP programme by a year or more. The
USDA image service carries the newest acquisitions, so it fills that gap. It
holds only the most recent collection for each area, though: once a newer year
is loaded the older one is gone, so download a year while it is there.

The service caps the size of one export, so a target grid is requested in tiles
and written tile by tile. Each tile is requested in the target CRS on the target
pixel corners, which keeps the result aligned with a canopy height model on that
grid. :func:`download_naip_rest_year` builds the same snapped UTM grid and
output name as :func:`csdv_core.download.naip_pc.download_naip_year`, so a year
from either source lines up with the others and feeds ``csdv chm-inference``.

Inputs: an EPSG:4326 box and a year. Outputs: a four-band NAIP GeoTIFF with an
internal mask over area without imagery, and a provenance JSON sidecar.
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from csdv_core.download.naip_pc import (
    NAIP_BANDS,
    SNAP_M,
    _dominant,
    _resolve_aoi,
    _resolve_years,
    grid_from_bounds,
    median_date_tag,
    naip_output_path,
    snap_bounds,
)

logger = logging.getLogger(__name__)

NAIP_REST_URL = (
    "https://apps.geo.fpac.usda.gov/geo-imagery/rest/services/"
    "naip/conus_naip/ImageServer"
)
#: EPSG code of NAD83 / UTM zone 1N. NAIP quads are delivered in NAD83 UTM.
NAD83_UTM_EPSG_BASE = 26900
#: Quarter-quad names such as ``m_4506912_se_19_030_20250815``: UTM zone,
#: resolution token and acquisition date. A trailing processing date, as in
#: ``..._20230922_20240103``, is ignored.
_QUAD_NAME_RE = re.compile(
    r"^m_\d{7}_[ns][ew]_(?P<zone>\d{1,2})_(?P<res>\d{1,3})_(?P<date>\d{8})"
)

__all__ = [
    "NAD83_UTM_EPSG_BASE",
    "NAIP_REST_URL",
    "QuadName",
    "download_naip_rest_year",
    "export_naip_to_grid",
    "find_quads",
    "parse_quad_name",
]


@dataclass(frozen=True)
class QuadName:
    """What a NAIP quarter-quad name says about the quad.

    Attributes:
        zone: UTM zone.
        gsd: Ground sample distance (m).
        date: Acquisition date, ``YYYYMMDD``.
    """

    zone: int
    gsd: float
    date: str

    @property
    def epsg(self) -> int:
        """EPSG code of the quad's NAD83 UTM zone."""
        return NAD83_UTM_EPSG_BASE + self.zone

    @property
    def iso_date(self) -> str:
        """Acquisition date as ``YYYY-MM-DD``."""
        return f"{self.date[:4]}-{self.date[4:6]}-{self.date[6:]}"


def parse_quad_name(name: str) -> QuadName:
    """Read the UTM zone, resolution and date from a NAIP quarter-quad name.

    The resolution token is metres for one digit (``1`` in older 1 m names)
    and centimetres for three (``060``, ``030``).

    Args:
        name: Catalogue name, e.g. ``m_4506912_se_19_030_20250815``.

    Returns:
        The parsed name.

    Raises:
        ValueError: If ``name`` is not a quarter-quad name.
    """
    match = _QUAD_NAME_RE.match(name)
    if match is None:
        raise ValueError(f"Not a NAIP quarter-quad name: {name!r}")
    res_token = match["res"]
    value = int(res_token)
    gsd = float(value) if len(res_token) == 1 else value / 100.0
    return QuadName(zone=int(match["zone"]), gsd=gsd, date=match["date"])


def find_quads(
    bbox_4326: tuple[float, float, float, float], *, year: int
) -> list[dict[str, Any]]:
    """Return the catalogue attributes of the quads over ``bbox_4326`` in ``year``."""
    import requests

    params = {
        "where": f"year_ts = {int(year)}",
        "geometry": ",".join(f"{v:.6f}" for v in bbox_4326),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "OBJECTID,Name,year_ts,ST,QQDATE",
        "returnGeometry": "false",
        "f": "json",
    }
    response = requests.get(f"{NAIP_REST_URL}/query", params=params, timeout=120)
    response.raise_for_status()
    return [feature["attributes"] for feature in response.json().get("features", [])]


def _export_tile(params: dict[str, str], attempts: int, label: str) -> Any:
    """Request one tile from the service, retrying, and return its pixels."""
    import rasterio
    import requests

    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(
                f"{NAIP_REST_URL}/exportImage", params=params, timeout=300
            )
            response.raise_for_status()
            with rasterio.open(io.BytesIO(response.content)) as src:
                return src.read()
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts:
                raise
            logger.warning("Tile %s failed (%s), retrying", label, exc)
            time.sleep(5 * attempt)
    raise RuntimeError("unreachable")


def export_naip_to_grid(
    out_path: Path | str,
    *,
    epsg: int,
    dst_transform: Any,
    dst_width: int,
    dst_height: int,
    object_ids: list[int] | None = None,
    tile_px: int = 2048,
    attempts: int = 4,
) -> tuple[Path, dict[str, Any]]:
    """Export the service onto a north-up target grid as a four-band GeoTIFF.

    Tiles are written as they arrive, so memory stays at one tile however
    large the grid. Pixels the service returns as zero in every band (no
    imagery) are flagged in an internal dataset mask (0 = no imagery).

    Args:
        out_path: Destination GeoTIFF.
        epsg: EPSG code of the target grid.
        dst_transform: Target affine transform.
        dst_width: Target width in pixels.
        dst_height: Target height in pixels.
        object_ids: Catalogue rasters to lock the mosaic to, from
            :func:`find_quads`. Without it the service chooses, and may return
            a different year.
        tile_px: Tile edge per request. The service caps one export at
            15000 by 4100 pixels. Keep it a multiple of 512, the GeoTIFF
            block size, so no block is written twice.
        attempts: Tries per tile before giving up.

    Returns:
        ``(path, provenance)``, with the grid written, the locked catalogue
        rasters and the share of the grid with imagery.
    """
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res = dst_transform.a
    x0, y0 = dst_transform.c, dst_transform.f

    base = {
        "bboxSR": str(epsg),
        "imageSR": str(epsg),
        "format": "tiff",
        "pixelType": "U8",
        "interpolation": "RSP_NearestNeighbor",
        "f": "image",
    }
    if object_ids:
        ids = ",".join(str(i) for i in object_ids)
        base["mosaicRule"] = (
            f'{{"mosaicMethod":"esriMosaicLockRaster","lockRasterIds":[{ids}]}}'
        )
    profile = {
        "driver": "GTiff",
        "height": dst_height,
        "width": dst_width,
        "count": 4,
        "dtype": "uint8",
        "crs": f"EPSG:{epsg}",
        "transform": dst_transform,
        "compress": "DEFLATE",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "IF_SAFER",
    }

    n_tiles = -(-dst_height // tile_px) * -(-dst_width // tile_px)
    done = 0
    covered_px = 0
    with (
        rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True),
        rasterio.open(out_path, "w", **profile) as dst,
    ):
        dst.descriptions = NAIP_BANDS
        for row in range(0, dst_height, tile_px):
            for col in range(0, dst_width, tile_px):
                h = min(tile_px, dst_height - row)
                w = min(tile_px, dst_width - col)
                minx, maxy = x0 + col * res, y0 - row * res
                params = {
                    **base,
                    "bbox": f"{minx},{maxy - h * res},{minx + w * res},{maxy}",
                    "size": f"{w},{h}",
                }
                block = _export_tile(params, attempts, f"{row},{col}")
                data = np.zeros((4, h, w), dtype="uint8")
                n_bands = min(4, block.shape[0])
                data[:n_bands] = block[:n_bands, :h, :w]
                covered = data.any(axis=0)
                window = Window(col, row, w, h)
                dst.write(data, window=window)
                dst.write_mask(covered, window=window)
                covered_px += int(covered.sum())
                done += 1
                logger.info("Tile %d of %d", done, n_tiles)

    coverage = covered_px / (dst_width * dst_height)
    if coverage < 0.999:
        logger.warning(
            "%s covers only %.3f of the target grid; the service has no imagery "
            "for the rest",
            out_path.name,
            coverage,
        )
    logger.info("Wrote %s", out_path)
    provenance = {
        "crs": f"EPSG:{epsg}",
        "transform": list(dst_transform)[:6],
        "shape": [dst_height, dst_width],
        "object_ids": list(object_ids or []),
        "coverage_fraction": coverage,
        "path": str(out_path),
    }
    return out_path, provenance


def download_naip_rest_year(
    bbox_4326: tuple[float, float, float, float],
    *,
    site: str,
    year: int,
    out_root: Path | str,
    pad_m: float = 150.0,
    overwrite: bool = False,
) -> Path:
    """Mosaic one year of NAIP from the USDA image service over a box.

    The grid and file name match
    :func:`csdv_core.download.naip_pc.download_naip_year`: the quads' own UTM
    CRS and resolution, the box widened by ``pad_m`` and snapped to
    :data:`~csdv_core.download.naip_pc.SNAP_M`, and
    ``<out_root>/<site>/<year>/NAIP_<site>_<YYYYMMDD>_<res>.tif``. A
    ``<stem>.provenance.json`` sidecar records the quads and their dates.

    Returns:
        Path of the written (or already present) GeoTIFF.

    Raises:
        ValueError: If the service has no quads for ``year`` over the box.
    """
    from rasterio.warp import transform_bounds

    year_dir = Path(out_root) / site / str(year)
    existing = sorted(year_dir.glob("*.tif"))
    if existing and not overwrite:
        logger.info("%d: %s exists, skipping", year, existing[0].name)
        return existing[0]

    quads = find_quads(bbox_4326, year=year)
    if not quads:
        raise ValueError(
            f"No NAIP imagery for {year} over {bbox_4326} in the USDA image service"
        )
    names = [parse_quad_name(quad["Name"]) for quad in quads]
    epsg = _dominant([name.epsg for name in names], "CRS", year)
    gsd = float(_dominant([name.gsd for name in names], "GSD", year))
    gsds = sorted({name.gsd for name in names})
    logger.info("%d: %d NAIP quads, ground sample distance %s", year, len(quads), gsds)
    dst_crs = f"EPSG:{epsg}"
    utm_bounds = transform_bounds("EPSG:4326", dst_crs, *bbox_4326, densify_pts=21)
    grid = grid_from_bounds(
        snap_bounds(utm_bounds, snap_m=SNAP_M, pad_m=pad_m), resolution=gsd
    )
    dates = [name.iso_date for name in names]
    date_tag = median_date_tag(dates)
    out_path = naip_output_path(out_root, site, year, date_tag, gsd)

    path, provenance = export_naip_to_grid(
        out_path,
        epsg=epsg,
        dst_transform=grid.transform,
        dst_width=grid.width,
        dst_height=grid.height,
        object_ids=[int(quad["OBJECTID"]) for quad in quads],
    )
    provenance.update(
        {
            "source": "usda_fpac_image_service",
            "service_url": NAIP_REST_URL,
            "site": site,
            "bbox_4326": list(bbox_4326),
            "n_items": len(quads),
            "item_ids": [quad["Name"] for quad in quads],
            "acq_dates": sorted(dates),
            "date_tag": date_tag,
            "gsd": gsds,
        }
    )
    out_path.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return path


@click.command("naip-rest")
@click.option("--site", required=True, help="Site label; output goes to naip/<site>/.")
@click.option(
    "--bbox",
    nargs=4,
    type=float,
    default=None,
    help="W S E N in EPSG:4326. Default: sites.yaml center +/- --half-km.",
)
@click.option("--years", default=None, help="Comma-separated. Default: sites.yaml.")
@click.option("--half-km", default=2.5, show_default=True, type=float)
@click.option(
    "--pad-m",
    default=150.0,
    show_default=True,
    type=float,
    help="Buffer added before snapping; covers the CHM edge artefact.",
)
@click.option(
    "--out-root",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Default: $CSDV_DATA_ROOT/naip.",
)
@click.option("--overwrite", is_flag=True, default=False)
def cli(
    site: str,
    bbox: tuple[float, float, float, float] | None,
    years: str | None,
    half_km: float,
    pad_m: float,
    out_root: Path | None,
    overwrite: bool,
) -> None:
    """Download NAIP from the USDA FPAC image service, one mosaic per year.

    The service holds only the newest collection for each area, which is often
    a year the Planetary Computer does not have yet. Needs outbound internet.
    On CHPC run it on a data transfer node.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s"
    )
    bbox_4326 = _resolve_aoi(site, bbox, half_km)
    year_list = _resolve_years(site, years)
    if out_root is None:
        from csdv_core.io.paths import project_paths

        out_root = project_paths().data_root / "naip"

    failed = []
    for year in year_list:
        try:
            path = download_naip_rest_year(
                bbox_4326,
                site=site,
                year=year,
                out_root=out_root,
                pad_m=pad_m,
                overwrite=overwrite,
            )
            click.echo(str(path))
        except ValueError as exc:
            logger.error("%d: %s", year, exc)
            failed.append(year)
    if failed:
        raise click.ClickException(f"No mosaic written for {failed}")
