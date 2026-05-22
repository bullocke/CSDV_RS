"""Unit tests for csdv_core.download._ee geometry helpers.

These tests import ``ee`` for real and skip if either the import or
``ee.Initialize`` fails. They only build geometry objects and inspect
their JSON representation locally; no Earth Engine evaluation runs.
"""

from __future__ import annotations

import pytest

ee = pytest.importorskip("ee")


def _init_ee_or_skip() -> None:
    """Initialize EE against the default project, or skip if it fails."""
    try:
        ee.Initialize()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Earth Engine not initialized in this environment: {exc}")


from csdv_core.download._ee import bbox_to_polygon, site_aoi  # noqa: E402


def test_bbox_to_polygon_round_trip() -> None:
    """bbox_to_polygon returns an ee.Geometry with the expected coordinates."""
    _init_ee_or_skip()
    geom = bbox_to_polygon(west=-78.2, south=38.85, east=-78.1, north=38.95)
    assert isinstance(geom, ee.Geometry)
    info = geom.toGeoJSONString()
    assert "-78.2" in info
    assert "38.95" in info


def test_site_aoi_uses_center_lonlat_from_sites_yaml() -> None:
    """site_aoi returns an ee.Geometry for a configured site (SCBI)."""
    _init_ee_or_skip()
    geom = site_aoi("SCBI", half_km=2.5)
    assert isinstance(geom, ee.Geometry)
    info = geom.toGeoJSONString()
    assert "-78" in info
    assert "38" in info


def test_site_aoi_unknown_site_raises() -> None:
    """Unknown site codes raise KeyError (from sites.yaml lookup)."""
    _init_ee_or_skip()
    with pytest.raises(KeyError):
        site_aoi("ZZZZ", half_km=1.0)
