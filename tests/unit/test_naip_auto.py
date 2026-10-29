"""Tests for choosing a NAIP source per year (``csdv download naip-auto``).

No network calls. The catalogue lookups and the per-year downloads are
replaced, so what is checked is the routing between the Planetary Computer and
the USDA service, the coverage threshold and the command's exit status.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner
from shapely.geometry import box

from csdv_core.download import naip_auto, naip_pc, naip_rest
from csdv_core.download.naip_auto import plan_years
from csdv_core.download.naip_pc import footprint_coverage

BBOX = (-71.75, 44.0, -71.625, 44.125)


def test_planetary_computer_is_preferred_and_usda_fills_newer_years():
    plan = plan_years({2018: 1.0, 2021: 1.0, 2023: 1.0}, {2023: 1.0, 2025: 1.0})
    assert [(p.year, p.source) for p in plan.years] == [
        (2018, "pc"),
        (2021, "pc"),
        (2023, "pc"),
        (2025, "rest"),
    ]
    assert plan.skipped == {}


def test_low_coverage_years_are_skipped_with_a_reason():
    plan = plan_years({2021: 1.0, 2023: 0.52}, {2025: 1.0}, min_coverage=0.9)
    assert plan.year_list == [2021, 2025]
    assert "52%" in plan.skipped[2023]


def test_usda_is_used_when_the_planetary_computer_year_is_partial():
    plan = plan_years({2023: 0.4}, {2023: 1.0}, min_coverage=0.9)
    assert [(p.year, p.source) for p in plan.years] == [(2023, "rest")]


def test_start_year_and_explicit_years():
    pc = {2008: 1.0, 2010: 1.0, 2012: 1.0}
    assert plan_years(pc, {}, start_year=2010).year_list == [2010, 2012]
    plan = plan_years(pc, {}, years=[2008, 2014])
    assert plan.year_list == [2008]
    assert plan.skipped[2014] == "no imagery in either catalogue"


def test_footprint_coverage():
    west, south, east, north = BBOX
    half = box(west, south, (west + east) / 2, north)
    outside = box(east, south, east + 1, north)
    assert footprint_coverage(BBOX, [half]) == pytest.approx(0.5)
    assert footprint_coverage(BBOX, [half, half, outside]) == pytest.approx(0.5)
    assert footprint_coverage(BBOX, []) == 0.0


def _fake_sources(monkeypatch, calls):
    monkeypatch.setattr(naip_pc, "available_years", lambda bbox: {2021: 1.0, 2023: 0.3})
    monkeypatch.setattr(naip_rest, "available_years", lambda bbox: {2025: 1.0})

    def fake(source):
        def download(bbox, *, site, year, out_root, pad_m, overwrite):
            calls.append((source, year))
            return out_root / site / str(year) / f"NAIP_{site}_{year}0701_0p6m.tif"

        return download

    monkeypatch.setattr(naip_pc, "download_naip_year", fake("pc"))
    monkeypatch.setattr(naip_rest, "download_naip_rest_year", fake("rest"))


def test_cli_downloads_each_year_from_its_source(monkeypatch, tmp_path):
    calls: list[tuple[str, int]] = []
    _fake_sources(monkeypatch, calls)
    args = ["--site", "t", "--bbox", *map(str, BBOX), "--out-root", str(tmp_path)]
    result = CliRunner().invoke(naip_auto.cli, args)
    assert result.exit_code == 0, result.output
    assert calls == [("pc", 2021), ("rest", 2025)]


def test_cli_fails_when_a_named_year_is_unusable(monkeypatch, tmp_path):
    calls: list[tuple[str, int]] = []
    _fake_sources(monkeypatch, calls)
    args = ["--site", "t", "--bbox", *map(str, BBOX), "--out-root", str(tmp_path)]
    result = CliRunner().invoke(naip_auto.cli, [*args, "--years", "2021,2023"])
    assert result.exit_code != 0
    assert "2023" in result.output
    assert calls == [("pc", 2021)]
