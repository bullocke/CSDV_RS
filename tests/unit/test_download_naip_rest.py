"""Tests for the USDA image service download and its CLI.

No network calls. ``requests.get`` is replaced by a fake service that renders
GeoTIFF tiles with imagery east of a set easting only, so the tile writing and
the gap mask are exercised. The catalogue search is replaced where the site-year
download reaches it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
import requests
from click.testing import CliRunner
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds, from_origin

from csdv_core.download import naip_rest
from csdv_core.download.naip_rest import (
    download_naip_rest_year,
    export_naip_to_grid,
    parse_quad_name,
)


@pytest.mark.parametrize(
    "name, zone, gsd, date",
    [
        ("m_4506912_se_19_030_20250815", 19, 0.3, "20250815"),
        ("m_4506922_sw_19_030_20230922_20240103", 19, 0.3, "20230922"),
        ("m_3807753_ne_17_060_20180612", 17, 0.6, "20180612"),
        ("m_3807753_ne_17_1_20120607", 17, 1.0, "20120607"),
    ],
)
def test_parse_quad_name(name, zone, gsd, date):
    quad = parse_quad_name(name)
    assert (quad.zone, quad.gsd, quad.date) == (zone, pytest.approx(gsd), date)
    assert quad.epsg == 26900 + zone


def test_parse_quad_name_rejects_overviews():
    with pytest.raises(ValueError, match="quarter-quad"):
        parse_quad_name("Ov_i02_L01_R00000199_C0000014D.tif")


class _FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self) -> None:
        return None


def _fake_service(east_of: float, calls: list[dict]):
    """``requests.get`` stand-in: imagery (value 100) east of ``east_of`` only."""

    def _get(url, params=None, timeout=None):
        calls.append(params)
        width, height = (int(v) for v in params["size"].split(","))
        minx, miny, maxx, maxy = (float(v) for v in params["bbox"].split(","))
        xs = minx + (np.arange(width) + 0.5) * (maxx - minx) / width
        data = np.zeros((4, height, width), dtype="uint8")
        data[:, :, xs > east_of] = 100
        profile = {
            "driver": "GTiff",
            "width": width,
            "height": height,
            "count": 4,
            "dtype": "uint8",
            "crs": f"EPSG:{params['imageSR']}",
            "transform": from_bounds(minx, miny, maxx, maxy, width, height),
        }
        with MemoryFile() as memfile:
            with memfile.open(**profile) as dst:
                dst.write(data)
            return _FakeResponse(memfile.read())

    return _get


def test_export_writes_tiles_and_masks_area_without_imagery(tmp_path, monkeypatch):
    calls: list[dict] = []
    x0, y0, res = 500_000.0, 5_000_000.0, 0.6
    width, height = 1100, 600
    east_of = x0 + 400 * res  # imagery from column 400 on
    monkeypatch.setattr(requests, "get", _fake_service(east_of, calls))

    out_path = tmp_path / "naip.tif"
    path, provenance = export_naip_to_grid(
        out_path,
        epsg=26919,
        dst_transform=from_origin(x0, y0, res, res),
        dst_width=width,
        dst_height=height,
        object_ids=[7, 9],
        tile_px=512,
    )

    assert path == out_path
    assert len(calls) == 3 * 2
    assert '"lockRasterIds":[7,9]' in calls[0]["mosaicRule"]
    with rasterio.open(out_path) as src:
        data = src.read()
        mask = src.dataset_mask()
        assert src.descriptions == ("red", "green", "blue", "nir")
    assert (data[:, :, 400:] == 100).all()
    assert (data[:, :, :400] == 0).all()
    assert (mask[:, :400] == 0).all() and (mask[:, 400:] == 255).all()
    assert not list(tmp_path.glob("*.msk"))  # the mask is internal
    assert provenance["coverage_fraction"] == pytest.approx(700 / 1100)
    assert provenance["object_ids"] == [7, 9]


def test_site_year_download_matches_the_planetary_computer_layout(
    tmp_path, monkeypatch
):
    quads = [
        {"OBJECTID": 11, "Name": "m_4506912_se_19_030_20250815"},
        {"OBJECTID": 12, "Name": "m_4506913_sw_19_030_20250815"},
        {"OBJECTID": 13, "Name": "m_4506913_se_19_030_20250817"},
    ]
    exported = {}

    def _fake_export(out_path, **kwargs):
        exported.update(kwargs, out_path=Path(out_path))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).touch()
        return Path(out_path), {"coverage_fraction": 1.0}

    monkeypatch.setattr(naip_rest, "find_quads", lambda bbox, year: quads)
    monkeypatch.setattr(naip_rest, "export_naip_to_grid", _fake_export)
    path = download_naip_rest_year(
        (-69.5656, 45.6636, -69.3253, 45.7501),
        site="NorthMaine",
        year=2025,
        out_root=tmp_path,
    )

    assert (
        path == tmp_path / "NorthMaine" / "2025" / "NAIP_NorthMaine_20250815_0p3m.tif"
    )
    assert exported["epsg"] == 26919
    assert exported["object_ids"] == [11, 12, 13]
    transform = exported["dst_transform"]
    assert transform.a == pytest.approx(0.3)
    # Snapped to the shared 3 m multiple, as the Planetary Computer grid is.
    assert transform.c % 3.0 == pytest.approx(0.0)
    assert transform.f % 3.0 == pytest.approx(0.0)
    sidecar = json.loads(path.with_suffix(".provenance.json").read_text())
    assert sidecar["source"] == "usda_fpac_image_service"
    assert sidecar["n_items"] == 3
    assert sidecar["acq_dates"][-1] == "2025-08-17"


def test_site_year_download_without_quads_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(naip_rest, "find_quads", lambda bbox, year: [])
    with pytest.raises(ValueError, match="No NAIP imagery for 2024"):
        download_naip_rest_year(
            (-69.5, 45.6, -69.3, 45.7), site="X", year=2024, out_root=tmp_path
        )


def test_existing_year_is_skipped_without_searching(tmp_path, monkeypatch):
    year_dir = tmp_path / "X" / "2025"
    year_dir.mkdir(parents=True)
    existing = year_dir / "NAIP_X_20250815_0p3m.tif"
    existing.touch()

    def _no_search(*args, **kwargs):
        raise AssertionError("search should not run when the year exists")

    monkeypatch.setattr(naip_rest, "find_quads", _no_search)
    path = download_naip_rest_year(
        (-69.5, 45.6, -69.3, 45.7), site="X", year=2025, out_root=tmp_path
    )
    assert path == existing


def test_cli_downloads_each_year_and_reports_the_missing_ones(tmp_path, monkeypatch):
    calls = []

    def _fake_download(bbox_4326, *, site, year, out_root, pad_m, overwrite):
        calls.append(year)
        if year == 2024:
            raise ValueError("No NAIP imagery for 2024")
        return out_root / site / str(year) / "fake_20250815.tif"

    monkeypatch.setattr(naip_rest, "download_naip_rest_year", _fake_download)
    result = CliRunner().invoke(
        naip_rest.cli,
        [
            "--site",
            "NorthMaine",
            "--bbox",
            "-69.5656",
            "45.6636",
            "-69.3253",
            "45.7501",
            "--years",
            "2024,2025",
            "--out-root",
            str(tmp_path),
        ],
    )
    assert calls == [2024, 2025]
    assert result.exit_code != 0
    assert "No mosaic written for [2024]" in result.output
    assert "fake_20250815.tif" in result.output
