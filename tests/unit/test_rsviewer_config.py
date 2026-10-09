"""Tests for rsviewer.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from rsviewer.config import expand_env, load_catalog
from rsviewer.layers import Category, RasterLayer, VectorLayer

YAML = """
title: Test site
zoom: 15
vars:
  cache: ${RSV_TEST_CACHE:-cache}
  naip: "{cache}/naip"
layers:
  - type: raster
    name: NAIP {year}
    path: "{naip}/naip_{year}.tif"
    foreach: {year: [2016, 2018]}
    visible: {year: [2018]}
    group: NAIP
    style: {bands: [1, 2, 3]}
  - type: raster
    name: Onset
    path: onset.tif
    style:
      categories: {1: [2014, "#111111"], 2: {label: "2016", color: "#222222"}}
  - type: vector
    name: Reference
    path: ref.gpkg
    layer: disturbance
    visible: false
    style:
      by: chg_type
      categories: {stand_replacing: [Stand-replacing, "#e41a1c"]}
    popup: [ref_id]
"""


def test_load_catalog_expands_loops_vars_and_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("RSV_TEST_CACHE", raising=False)
    (tmp_path / "cat.yaml").write_text(YAML)
    cat = load_catalog(tmp_path / "cat.yaml")
    assert cat.title == "Test site" and cat.zoom == 15
    names = [lyr.name for lyr in cat.layers]
    assert names == ["NAIP 2016", "NAIP 2018", "Onset", "Reference"]
    n16, n18, onset, ref = cat.layers
    assert isinstance(n16, RasterLayer) and isinstance(ref, VectorLayer)
    assert n16.path == tmp_path / "cache" / "naip" / "naip_2016.tif"
    assert (n16.visible, n18.visible) == (False, True)
    assert n16.style.bands == (1, 2, 3) and n16.style.is_rgb
    assert onset.path == tmp_path / "onset.tif"
    assert onset.style.categories == {
        1: Category("2014", "#111111"),
        2: Category("2016", "#222222"),
    }
    assert ref.visible is False and ref.layer == "disturbance"
    assert ref.style.categories["stand_replacing"].color == "#e41a1c"


def test_env_and_override_variables(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RSV_TEST_CACHE", "/data/c")
    (tmp_path / "cat.yaml").write_text(YAML)
    cat = load_catalog(tmp_path / "cat.yaml")
    assert cat.layers[0].path == Path("/data/c/naip/naip_2016.tif")
    cat = load_catalog(tmp_path / "cat.yaml", variables={"naip": "/elsewhere"})
    assert cat.layers[0].path == Path("/elsewhere/naip_2016.tif")


def test_expand_env_requires_unset_vars_to_have_defaults(monkeypatch) -> None:
    monkeypatch.delenv("RSV_MISSING", raising=False)
    assert expand_env("${RSV_MISSING:-x}/y") == "x/y"
    with pytest.raises(KeyError):
        expand_env("${RSV_MISSING}/y")


def test_unknown_layer_type(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text("layers:\n  - {type: mesh, name: x, path: a}\n")
    with pytest.raises(ValueError, match="unknown layer type"):
        load_catalog(tmp_path / "bad.yaml")
