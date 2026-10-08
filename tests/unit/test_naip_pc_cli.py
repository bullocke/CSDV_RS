"""Tests for the site-year Planetary Computer download and its CLI.

No network calls. The STAC search and the mosaic are replaced where a test
reaches them, so what is checked is the AOI, the output naming the canopy
height model depends on, and how the command resolves its arguments.
"""

from __future__ import annotations

import math
import re

import pytest
from click.testing import CliRunner

from csdv_core.download import naip_pc
from csdv_core.download.naip_pc import (
    aoi_bbox_from_center,
    download_naip_year,
    naip_output_path,
    resolution_tag,
)


def test_aoi_box_is_centred_and_about_the_requested_size():
    lon, lat = -78.1454, 38.8929
    west, south, east, north = aoi_bbox_from_center((lon, lat), half_km=2.5)
    assert (west + east) / 2 == pytest.approx(lon)
    assert (south + north) / 2 == pytest.approx(lat)
    height_km = (north - south) * 111.32
    width_km = (east - west) * 111.32 * math.cos(math.radians(lat))
    assert height_km == pytest.approx(5.0)
    assert width_km == pytest.approx(5.0)


def test_aoi_box_rejects_a_non_positive_half_width():
    with pytest.raises(ValueError, match="half_km"):
        aoi_bbox_from_center((-78.0, 38.0), half_km=0)


@pytest.mark.parametrize("gsd, tag", [(0.6, "0p6m"), (1.0, "1m"), (0.3, "0p3m")])
def test_resolution_tag(gsd, tag):
    assert resolution_tag(gsd) == tag


def test_output_name_carries_the_date_token_the_model_reads(tmp_path):
    path = naip_output_path(tmp_path, "SantaClara", 2018, "20180612", 0.6)
    assert path == tmp_path / "SantaClara" / "2018" / (
        "NAIP_SantaClara_20180612_0p6m.tif"
    )
    # Same pattern csdv_core.chm_inference.infer validates.
    assert re.search(r"\d{8}", path.stem).group() == "20180612"


def test_existing_year_is_skipped_without_searching(tmp_path, monkeypatch):
    year_dir = tmp_path / "SCBI" / "2018"
    year_dir.mkdir(parents=True)
    existing = year_dir / "m_3807753_ne_17_060_20180612.tif"
    existing.touch()

    def _no_search(*args, **kwargs):
        raise AssertionError("search should not run when the year exists")

    monkeypatch.setattr(naip_pc, "search_naip_items", _no_search)
    path = download_naip_year(
        (-78.2, 38.8, -78.1, 38.9), site="SCBI", year=2018, out_root=tmp_path
    )
    assert path == existing


def _run(args, monkeypatch, tmp_path):
    calls = []

    def _fake_download(bbox_4326, *, site, year, out_root, pad_m, overwrite):
        calls.append({"bbox": bbox_4326, "site": site, "year": year})
        return out_root / site / str(year) / "fake_20180612.tif"

    monkeypatch.setattr(naip_pc, "download_naip_year", _fake_download)
    result = CliRunner().invoke(
        naip_pc.cli, [*args, "--out-root", str(tmp_path)], catch_exceptions=False
    )
    return result, calls


def test_cli_uses_the_bbox_for_a_site_not_in_the_config(monkeypatch, tmp_path):
    result, calls = _run(
        [
            "--site",
            "SantaClara",
            "--bbox",
            "-86.31",
            "39.06",
            "-86.25",
            "39.12",
            "--years",
            "2016,2018",
        ],
        monkeypatch,
        tmp_path,
    )
    assert result.exit_code == 0, result.output
    assert [c["year"] for c in calls] == [2016, 2018]
    assert calls[0]["bbox"] == pytest.approx((-86.31, 39.06, -86.25, 39.12))


def test_cli_falls_back_to_sites_yaml_for_aoi_and_years(monkeypatch, tmp_path):
    result, calls = _run(["--site", "SCBI"], monkeypatch, tmp_path)
    assert result.exit_code == 0, result.output
    assert [c["year"] for c in calls] == [2014, 2016, 2018, 2021, 2023]
    west, south, east, north = calls[0]["bbox"]
    assert west < -78.1454 < east and south < 38.8929 < north


def test_cli_asks_for_a_bbox_when_the_site_is_unknown(monkeypatch, tmp_path):
    result, calls = _run(
        ["--site", "Nowhere", "--years", "2018"], monkeypatch, tmp_path
    )
    assert result.exit_code != 0
    assert "--bbox" in result.output
    assert calls == []
