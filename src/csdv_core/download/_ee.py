"""csdv_core.download._ee — Earth Engine initialization and AOI helpers.

Private module shared by ``naip_gee`` and ``chm_model``. All ``ee`` imports
are lazy so the package imports cleanly without ``earthengine-api``
installed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from csdv_core.config import load_sites

if TYPE_CHECKING:
    import ee  # noqa: F401

logger = logging.getLogger(__name__)

_DEFAULT_HIGH_VOLUME = "https://earthengine-highvolume.googleapis.com"


def initialize_ee(project: str = "dyce-biomass") -> None:
    """Initialize Earth Engine against the high-volume endpoint.

    Args:
        project: GEE project ID. Defaults to ``"dyce-biomass"``.
    """
    import ee

    ee.Initialize(project=project, opt_url=_DEFAULT_HIGH_VOLUME)
    logger.info("Earth Engine initialized (project=%s)", project)


def bbox_to_polygon(
    west: float, south: float, east: float, north: float
) -> ee.Geometry:
    """Convert a WGS84 bounding box to an EE polygon geometry."""
    import ee

    return ee.Geometry.Polygon(
        [[[west, north], [west, south], [east, south], [east, north]]]
    )


def site_aoi(site_code: str, half_km: float) -> ee.Geometry:
    """Build a square AOI centred on a configured site.

    Reads ``center_lonlat`` from ``config/sites.yaml`` and buffers by
    ``half_km`` kilometres on each side. Returns the bounding rectangle
    in WGS84.

    Args:
        site_code: Site code (case insensitive); must exist in
            ``sites.yaml`` with a non-null ``center_lonlat``.
        half_km: AOI half-side length in kilometres.

    Returns:
        ``ee.Geometry`` rectangle in EPSG:4326.

    Raises:
        KeyError: If the site is not configured.
        ValueError: If ``center_lonlat`` is missing for the site.
    """
    import ee

    sites = load_sites()
    entry = sites.get(site_code.upper())
    if entry.center_lonlat is None:
        raise ValueError(
            f"Site {site_code!r} has no center_lonlat in sites.yaml; "
            "cannot build AOI."
        )
    lon, lat = entry.center_lonlat
    return ee.Geometry.Point([lon, lat]).buffer(half_km * 1000.0).bounds()


def resolve_region(
    *,
    site: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    half_km: float | None = None,
) -> ee.Geometry:
    """Return an AOI geometry from either a site code or an explicit bbox.

    Exactly one of ``site`` or ``bbox`` must be provided. When ``site`` is
    used, ``half_km`` is required.
    """
    if bbox is not None and site is not None:
        raise ValueError("Provide either site or bbox, not both.")
    if bbox is not None:
        return bbox_to_polygon(*bbox)
    if site is not None:
        if half_km is None:
            raise ValueError("half_km is required when resolving an AOI from a site.")
        return site_aoi(site, half_km)
    raise ValueError("Provide either site (with half_km) or bbox.")


def export_image_to_tif(
    image: Any,
    *,
    out_dir: Any,
    description: str,
    region: ee.Geometry,
    scale: float,
    crs: str,
) -> Any:
    """Export an EE Image via wxee. Caller handles file naming and dirs."""
    import wxee  # noqa: F401  (registers the .wx accessor)

    return image.wx.to_tif(
        out_dir=str(out_dir),
        description=description,
        region=region,
        scale=scale,
        crs=crs,
    )
