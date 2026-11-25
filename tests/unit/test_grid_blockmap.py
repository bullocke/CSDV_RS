"""The page of a block of quads: layers rendered from a model map."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from csdv_core.grid.blockmap import (
    PROB_FLOOR,
    BlockmapConfig,
    _interval_key,
    _render_model,
    tile_resolution,
)


def test_tile_resolution_halves_per_zoom() -> None:
    assert tile_resolution(13) == 2 * tile_resolution(14)
    assert 4.7 < tile_resolution(14) < 4.8


def test_interval_key() -> None:
    assert _interval_key("2012-06-07_2014-07-05") == "2012_2014"


def test_render_model_colours_by_threshold(tmp_path: Path) -> None:
    unit = "q0000001"
    folder = tmp_path / "exports/run_layers" / unit / "default/model"
    folder.mkdir(parents=True)
    profile = {
        "driver": "GTiff", "height": 2, "width": 3, "count": 1, "dtype": "uint8",
        "crs": "EPSG:26916", "transform": from_origin(500000, 4300000, 9.6, 9.6),
        "nodata": 255,
    }  # fmt: skip
    prob = np.array([[[0, PROB_FLOOR, 95], [255, 60, 100]]], dtype=np.uint8)
    agent = np.array([[[1, 1, 3], [1, 2, 1]]], dtype=np.uint8)
    for name, data in (("prob_forest", prob), ("agent", agent)):
        with rasterio.open(folder / f"{name}_{unit}.tif", "w", **profile) as dst:
            dst.write(data)
            dst.set_band_description(1, "2018-08-12_2020-06-12")
    cfg = BlockmapConfig((), tmp_path / "site", tmp_path / "exports", threshold=90)
    out = tmp_path / "tmp"
    out.mkdir()
    rendered, dates = _render_model(cfg, unit, out)
    assert dates == {"2018": "2018-08-12", "2020": "2020-06-12"}
    with rasterio.open(rendered["2018_2020"]["prob"]) as src:
        alpha = src.read(4)
    # Below the floor and without a value: transparent. Rising with probability.
    assert alpha[0, 0] == 0 and alpha[1, 0] == 0
    assert 0 < alpha[0, 1] < alpha[1, 1] < alpha[0, 2] < alpha[1, 2]
    with rasterio.open(rendered["2018_2020"]["cause"]) as src:
        cause = src.read()
    assert (cause[3] > 0).tolist() == [[False, False, True], [False, False, True]]
    assert cause[:3, 0, 2].tolist() == [227, 73, 72]  # fire
    assert cause[:3, 1, 2].tolist() == [237, 161, 0]  # harvest
