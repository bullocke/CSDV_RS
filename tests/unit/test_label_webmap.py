"""Tests of the static map page of a label run (csdv_core.label.webmap)."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from csdv_core.label.webmap import (
    WebmapConfig,
    _years_of,
    build_webmap,
    target_group,
)

UNIT = "p_test_harvest_me1"
YEARS = (2019, 2021, 2023)
# A 360 m square in UTM zone 19N (Maine).
X0, Y0, SIDE = 500_000.0, 5_000_000.0, 360.0


def _write(path: Path, data: np.ndarray, res: float, names: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": data.shape[0],
        "height": data.shape[1],
        "width": data.shape[2],
        "crs": "EPSG:26919",
        "transform": from_origin(X0, Y0 + SIDE, res, res),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        if names:
            dst.descriptions = names


@pytest.fixture
def label_run(tmp_path: Path) -> dict[str, Path]:
    rng = np.random.default_rng(0)
    data_root, exports = tmp_path / "data", tmp_path / "exports"
    for year in YEARS:
        image = rng.integers(0, 255, (4, 600, 600), dtype=np.uint8)
        _write(data_root / "naip" / UNIT / str(year) / f"{year}.tif", image, 0.6)
    prob = np.full((2, 38, 38), 5, dtype=np.uint8)
    prob[0, 10:20, 10:20] = 90
    _write(
        exports / "run_layers" / UNIT / "default" / "model" / f"prob_{UNIT}.tif",
        prob,
        9.6,
        ["2019-07-01_2021-07-02", "2021-07-02_2023-07-03"],
    )
    # One event of 0.64 ha, stored twice (two windows), and one rejection.
    square = gpd.GeoSeries([box(X0 + 100, Y0 + 100, X0 + 180, Y0 + 180)], crs=26919)
    geom = square.to_crs(5070).iloc[0]
    event = {
        "unit": UNIT,
        "candidate_ids": f"{UNIT}-di-2021-1",
        "candidate_source": "di",
        "loss_intervals": "2019-07-01_2021-07-02",
        "severity": "stand_replacing",
        "agent": "harvest",
        "geometry_quality": "precise",
        "confidence": 3,
    }
    events = gpd.GeoDataFrame(
        [{**event, "window_id": "W1"}, {**event, "window_id": "W2"}],
        geometry=[geom, geom],
        crs=5070,
    )
    cand = {
        "unit": UNIT,
        "judged": True,
        "types": "harvest/harvest",
        "reason_gem37": "Canopy removed.",
        "reason_terra": "Clearcut.",
        "area_ha": 0.64,
    }
    cands = gpd.GeoDataFrame(
        [
            {
                **cand,
                "window_id": "W1",
                "cand_id": event["candidate_ids"],
                "source": "di",
                "interval": "2019-2021",
                "status": "accepted",
            },
            {
                **cand,
                "window_id": "W1",
                "cand_id": f"{UNIT}-model-2023-2",
                "source": "model",
                "interval": "2021-2023",
                "status": "rejected",
            },
        ],
        geometry=[geom, geom],
        crs=5070,
    )
    acq = gpd.GeoDataFrame(
        {
            "unit": UNIT,
            "window_id": "W1",
            "year": list(YEARS),
            "acquisition_date": ["2019-07-01", "2021-07-02", "2023-07-03"],
        }
    )
    label_dir = tmp_path / "label"
    (label_dir / "labels").mkdir(parents=True)
    gpkg = label_dir / "labels" / "labels.gpkg"
    events.to_file(gpkg, layer="events", driver="GPKG")
    cands.to_file(gpkg, layer="candidates", driver="GPKG")
    acq.to_file(gpkg, layer="acquisitions", driver="GPKG")
    pd.DataFrame(
        [
            {
                "unit": UNIT,
                "label": "test_me1",
                "target": "harvest_north",
                "kind": "loss",
                "role": "test",
                "location": "somewhere in Maine",
                "state": "me",
                "note": "",
            }
        ]
    ).to_csv(label_dir / "sites.csv", index=False)
    return {
        "label": label_dir,
        "data": data_root,
        "exports": exports,
        "out": tmp_path / "page",
    }


def _page(out: Path) -> dict:
    text = (out / "page.js").read_text()
    return json.loads(text[len("window.CSDV_PAGE=") : -1])


def test_years_and_groups():
    assert _years_of("2012-06-29_2014-07-24") == (2012, 2014)
    assert _years_of("2012-2014") == (2012, 2014)
    assert target_group("harvest_south_pine_off", "loss") == "harvest"
    assert target_group("beetle_rockies", "loss") == "biotic_mortality"
    assert target_group("other_loss", "loss") == "other"
    assert target_group("snow_conifer", "background") == "background"


def test_build_page(label_run):
    cfg = WebmapConfig(
        label_dir=label_run["label"],
        out_dir=label_run["out"],
        export_dir=label_run["exports"],
        data_root=label_run["data"],
        resolution=1.2,
    )
    result = build_webmap(cfg)
    assert result.places == 1 and result.images == 3
    # The event lies in two windows and is counted once.
    assert result.events == 1
    out = label_run["out"]
    page = _page(out)
    place = page["places"][0]
    assert place["group"] == "harvest" and place["role"] == "test"
    assert [y["year"] for y in place["years"]] == list(YEARS)
    assert [
        (v["a"], v["b"], v["loss"], v["background"]) for v in place["intervals"]
    ] == [
        (2019, 2021, 1, 0),
        (2021, 2023, 0, 1),
    ]
    assert all(v["model"] for v in place["intervals"])
    (south, west), (north, east) = place["bounds"]
    assert 45.0 < south < north < 45.3 and -69.1 < west < east < -68.9
    for year in YEARS:
        assert (out / "p" / UNIT / f"naip_{year}.webp").stat().st_size > 0
    labels = (out / "p" / UNIT / "labels.js").read_text()
    features = json.loads(labels[labels.index("]=") + 2 : -1])["features"]
    assert sorted(f["properties"]["layer"] for f in features) == ["background", "loss"]
    lon, lat = features[0]["geometry"]["coordinates"][0][0]
    assert west < lon < east and south < lat < north
    assert len(page["examples"]) == 1
    assert (out / page["examples"][0]["after"]).is_file()
    assert (out / "index.html").is_file()
    # A second build draws nothing again.
    assert build_webmap(cfg).images == 0


def test_quicklooks_without_mosaics(label_run):
    looks = label_run["exports"] / "quicklooks" / UNIT
    rng = np.random.default_rng(1)
    for year in YEARS:
        _write(
            looks / f"naip_{year}_3m_rgb.tif",
            rng.integers(0, 255, (3, 120, 120), dtype=np.uint8),
            3.0,
        )
    result = build_webmap(
        WebmapConfig(
            label_dir=label_run["label"],
            out_dir=label_run["out"],
            export_dir=label_run["exports"],
            data_root=None,
        )
    )
    assert result.images == 3
    assert _page(label_run["out"])["places"][0]["resolution"] == 3.0


def test_budget_stops_the_build(label_run):
    with pytest.raises(ValueError, match="projected"):
        build_webmap(
            WebmapConfig(
                label_dir=label_run["label"],
                out_dir=label_run["out"],
                export_dir=label_run["exports"],
                data_root=label_run["data"],
                budget_gb=1e-6,
            )
        )
