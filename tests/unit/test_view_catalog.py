"""Tests for the map viewer's catalogue, its tiles and its server."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from csdv_core.view import tiles
from csdv_core.view.catalog import build_catalog, interval_years, run_family

UNIT = "q4107127"
INTERVALS = ["2012-08-17_2014-07-18", "2014-07-18_2016-07-03"]
# A 40 by 30 cell grid of 9.6 m cells in UTM 19N, inside the unit's quad.
ORIGIN = (277000.0, 4607000.0)
CELL = 9.6


def _write(path: Path, data: np.ndarray, names: list[str], cell: float = CELL) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype="uint8",
        crs="EPSG:26919",
        transform=from_origin(*ORIGIN, cell, cell),
        nodata=255,
    ) as dst:
        dst.write(data)
        dst.descriptions = tuple(names)


def _run(
    export: Path, name: str, threshold: float | None, forest: bool = False
) -> None:
    prob = np.zeros((2, 30, 40), dtype=np.uint8)
    prob[0, 5:10, 5:10] = 95
    prob[1, 20:25, 20:30] = 90
    prob[:, 0, :] = 255
    agent = np.full((2, 30, 40), 1, dtype=np.uint8)
    agent[1] = 3
    agent[:, 0, :] = 255
    maps = export / "pairmodel" / name / "map"
    _write(maps / f"prob_{UNIT}.tif", prob, INTERVALS)
    _write(maps / f"agent_{UNIT}.tif", agent, INTERVALS)
    if forest:
        _write(maps / f"prob_forest_{UNIT}.tif", prob, INTERVALS)
    types = np.where(prob[:1] >= 89, agent[:1], 0).astype(np.uint8)
    text = f"Agent of loss at or above 0.89, {INTERVALS[0]}"
    _write(maps / f"type_{UNIT}_{INTERVALS[0]}.tif", types, [text])
    if threshold is not None:
        compare = {
            "threshold": threshold,
            "model": {
                "average_precision": 0.7797,
                "precision": 0.8,
                "recall": 0.6,
                "f1": 0.7,
            },
            "di": {"average_precision": 0.84},
        }
        (maps.parent / "compare").mkdir()
        (maps.parent / "compare" / "compare.json").write_text(json.dumps(compare))


@pytest.fixture
def export(tmp_path: Path) -> Path:
    root = tmp_path / "chpc_exports"
    _run(root, f"holdout_{UNIT}", 0.64, forest=True)
    _run(root, f"look_{UNIT}", None)
    _run(root, f"holdout_sf_{UNIT}", 0.91)
    (root / "pairmodel" / "not_a_run").mkdir()
    unit_dir = root / "runs" / "default" / UNIT
    onset = np.zeros((1, 15, 20), dtype=np.uint8)
    onset[0, 2:5, 2:5] = 14
    onset[0, 14, :] = 255
    _write(unit_dir / "onset_year_u8.tif", onset, ["onset"], cell=2 * CELL)
    record = {
        "bbox_4326": [-71.75, 41.5, -71.625, 41.625],
        "years": {
            "2012": {"source": "pc"},
            "2014": {"source": "pc"},
            "2025": {"source": "rest"},
        },
    }
    (unit_dir / "unit.json").write_text(json.dumps(record))
    quick = root / "quicklooks" / UNIT
    quick.mkdir(parents=True)
    rgb = np.full((3, 96, 128), 120, dtype=np.uint8)
    rgb[0, :48] = 200
    _write(quick / "naip_2014_3m_rgb.tif", rgb, ["r", "g", "b"], cell=3.0)
    low = np.full((1, 96, 128), 20, dtype=np.uint8)  # 5 m at 0.25 m steps
    high = np.full((1, 96, 128), 80, dtype=np.uint8)  # 20 m
    _write(quick / "chm_2014_3m.tif", high, ["chm"], cell=3.0)
    _write(quick / "chm_2016_3m.tif", low, ["chm"], cell=3.0)
    return root


@pytest.fixture
def sites_csv(tmp_path: Path) -> Path:
    path = tmp_path / "sites.csv"
    path.write_text(
        f"# a comment\nunit,batch,place,notes\n{UNIT},yes,Rhode Island oak forest,Moth\n"
    )
    return path


def test_run_family_and_interval_years() -> None:
    assert run_family("holdout_sf0_q3708104", "q3708104") == "holdout_sf0"
    assert run_family("look_q3708104", "q3708104") == "look"
    assert run_family("first_look_blocks2", "q4412328") == "first_look_blocks2"
    assert interval_years("2012-08-17_2014-07-18") == (2012, 2014)
    with pytest.raises(ValueError):
        interval_years("2012_2014")


def test_catalog_lists_runs_in_order_with_the_di_last(
    export: Path, sites_csv: Path
) -> None:
    cat = build_catalog(export, sites_csv)
    site = cat.site(UNIT)
    assert site.runs == (
        f"look_{UNIT}",
        f"holdout_{UNIT}",
        f"holdout_sf_{UNIT}",
        f"di_{UNIT}",
    )
    assert site.place == "Rhode Island oak forest"
    assert site.intervals == tuple(INTERVALS)
    assert site.years == (
        (2012, "2012-08-17"),
        (2014, "2014-07-18"),
        (2016, "2016-07-03"),
        (2025, None),
    )
    assert site.quicklook_years == (2014,)
    assert site.chm_years == (2014, 2016)
    assert site.bbox == (-71.75, 41.5, -71.625, 41.625)


def test_catalog_reads_thresholds_scores_and_layers(export: Path) -> None:
    cat = build_catalog(export)
    holdout = cat.run(f"holdout_{UNIT}")
    assert holdout.threshold == 0.64
    assert holdout.scores["model"]["average_precision"] == 0.78
    assert holdout.has_forest and not holdout.has_skip
    assert holdout.layers == ("date", "type", "prob")
    # Without a comparison the threshold comes from a type raster.
    assert cat.run(f"look_{UNIT}").threshold == 0.89
    di = cat.run(f"di_{UNIT}")
    assert di.family == "di" and di.layers == ("date",) and di.threshold is None
    json.dumps(cat.to_json())
    with pytest.raises(FileNotFoundError):
        cat.run("nothing")


def _centre_lonlat(row: float, col: float) -> tuple[float, float]:
    from pyproj import Transformer

    x = ORIGIN[0] + (col + 0.5) * CELL
    y = ORIGIN[1] - (row + 0.5) * CELL
    return Transformer.from_crs(26919, 4326, always_xy=True).transform(x, y)


def test_point_values_read_the_cell_under_a_position(export: Path) -> None:
    cat = build_catalog(export)
    lon, lat = _centre_lonlat(7, 7)
    values = tiles.point_values(cat.run(f"holdout_{UNIT}"), lon, lat)
    assert values["prob"] == [95, 0]
    assert values["agent"] == [1, 3]
    assert values["prob_forest"] == [95, 0]
    lon, lat = _centre_lonlat(0, 7)
    assert tiles.point_values(cat.run(f"holdout_{UNIT}"), lon, lat)["prob"] == [
        None,
        None,
    ]
    lon, lat = _centre_lonlat(7, 7)  # cell (3, 3) of the 19.2 m onset grid
    assert tiles.point_values(cat.run(f"di_{UNIT}"), lon, lat) == {"onset_year": 2014}
    assert tiles.point_values(cat.run(f"holdout_{UNIT}"), 0.0, 0.0)["prob"] == [
        None,
        None,
    ]


def _tile_of(lon: float, lat: float, z: int) -> tuple[int, int]:
    import math

    n = 2**z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def test_tile_samples_the_cells_at_their_place(export: Path) -> None:
    from pyproj import Transformer

    cat = build_catalog(export)
    layer = tiles.layer_of(cat.run(f"holdout_{UNIT}"), "type", "all", percent=64)
    assert layer.data[7, 7] == 1 and layer.data[22, 25] == 3 and layer.data[15, 15] == 0
    z = 17
    lon, lat = _centre_lonlat(15, 20)
    x, y = _tile_of(lon, lat, z)
    values, inside = tiles.sample_tile(layer, z, x, y)
    assert inside.any()
    # Every tile pixel holds the value of the cell its centre falls in.
    left, bottom, right, top = tiles.tile_bounds_3857(z, x, y)
    size = tiles.TILE_SIZE
    px = left + (np.arange(size) + 0.5) / size * (right - left)
    py = top - (np.arange(size) + 0.5) / size * (top - bottom)
    gx, gy = np.meshgrid(px, py)
    ux, uy = Transformer.from_crs(3857, 26919, always_xy=True).transform(gx, gy)
    col = np.floor((ux - ORIGIN[0]) / CELL).astype(int)
    row = np.floor((ORIGIN[1] - uy) / CELL).astype(int)
    ok = (col >= 0) & (col < 40) & (row >= 0) & (row < 30)
    # The coarse projection grid may move a pixel across a cell edge by a hair.
    assert (inside == ok).mean() > 0.999
    both = inside & ok
    assert (values[both] == layer.data[row[both], col[both]]).mean() > 0.995
    png = tiles.render_tile(layer, z, x, y)
    assert png is not None and png[:8] == b"\x89PNG\r\n\x1a\n"
    assert tiles.render_tile(layer, z, x + 50, y) is None


def test_layers_follow_threshold_interval_and_source(export: Path) -> None:
    cat = build_catalog(export)
    run = cat.run(f"holdout_{UNIT}")
    assert tiles.layer_of(run, "date", "all", percent=64).data[7, 7] == 14
    assert tiles.layer_of(run, "date", "all", percent=64).data[22, 25] == 16
    assert tiles.layer_of(run, "date", "all", percent=92).data[22, 25] == 0
    assert tiles.layer_of(run, "date", INTERVALS[1], percent=64).data[7, 7] == 0
    assert tiles.layer_of(run, "prob", "all").data[0, 0] == 255
    di = cat.run(f"di_{UNIT}")
    assert tiles.layer_of(di, "date", INTERVALS[0]).data[3, 3] == 14
    assert tiles.layer_of(di, "date", INTERVALS[1]).data[3, 3] == 0
    with pytest.raises(FileNotFoundError):
        tiles.layer_of(di, "type")
    with pytest.raises(FileNotFoundError):
        tiles.layer_of(run, "date", "2001-01-01_2002-01-01")


def test_server_routes(export: Path, sites_csv: Path, tmp_path: Path) -> None:
    from starlette.testclient import TestClient

    from csdv_core.view.server import ViewConfig, build_app

    notes = tmp_path / "notes" / "pair.geojson"
    client = TestClient(build_app(ViewConfig(export, notes, sites_csv)))
    assert "CSDV map viewer" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    body = client.get("/api/catalog").json()
    assert [s["unit"] for s in body["sites"]] == [UNIT]
    assert len(body["runs"]) == 4 and body["labels"] is False
    assert set(body["legend"]) == {"years", "agents", "prob", "prob_min"}

    lon, lat = _centre_lonlat(22, 25)
    x, y = _tile_of(lon, lat, 15)
    url = f"/tiles/map/holdout_{UNIT}/type/all/64/all/15/{x}/{y}.png"
    tile = client.get(url)
    assert tile.status_code == 200 and tile.headers["content-type"] == "image/png"
    assert client.get(url.replace(f"/{x}/", f"/{x + 40}/")).status_code == 204
    assert client.get(url.replace("/type/", "/nonsense/")).status_code == 404
    assert client.get(url.replace("holdout_", "missing_")).status_code == 404
    point = client.get(
        "/api/point", params={"unit": UNIT, "lon": lon, "lat": lat}
    ).json()
    assert point["runs"][f"holdout_{UNIT}"]["prob"] == [0, 90]
    assert client.get(f"/api/labels/{UNIT}").status_code == 404

    # Quicklook tiles: the centre of the 384 by 288 m quicklook at zoom 18.
    qlon, qlat = _centre_lonlat(15, 20)
    qx, qy = _tile_of(qlon, qlat, 18)
    naip = client.get(f"/tiles/naip/{UNIT}/2014/18/{qx}/{qy}")
    assert naip.status_code == 200 and naip.headers["content-type"] == "image/jpeg"
    assert client.get(f"/tiles/naip/{UNIT}/2014/18/{qx + 9}/{qy}").status_code == 204
    assert client.get(f"/tiles/naip/{UNIT}/1999/18/{qx}/{qy}").status_code == 404
    assert client.get(f"/tiles/chm/{UNIT}/2014/18/{qx}/{qy}.png").status_code == 200
    change = client.get(f"/tiles/dchm/{UNIT}/2014/2016/18/{qx}/{qy}.png")
    assert change.status_code == 200
    red, _, blue, alpha = np.asarray(Image.open(BytesIO(change.content)))[128, 128]
    assert red > blue and alpha > 0  # a 15 m loss of height is drawn red

    put = client.put(
        "/api/notes/n1", json={"lon": lon, "lat": lat, "tag": "good", "unit": UNIT}
    )
    assert put.status_code == 200 and put.json()["properties"]["tag"] == "good"
    assert len(client.get("/api/notes").json()["features"]) == 1
    assert json.loads(notes.read_text())["features"][0]["properties"]["unit"] == UNIT
    assert client.put("/api/notes/n1", json={"lon": lon, "lat": lat}).json() == {
        "deleted": "n1"
    }
    assert client.put("/api/notes/n2", json={"lon": lon}).status_code == 400
    assert (
        client.put("/api/notes/bad id", json={"lon": lon, "lat": lat}).status_code
        == 400
    )
    client.put("/api/notes/n3", json={"lon": lon, "lat": lat, "comment": "x"})
    assert client.delete("/api/notes/n3").json() == {"deleted": "n3", "existed": True}
    assert client.get("/api/notes").json()["features"] == []
