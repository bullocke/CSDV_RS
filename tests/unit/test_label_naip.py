"""Acquisition date of a window from the quads a mosaic was built from."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from csdv_core.grid.cells import quarter_quad_bounds
from csdv_core.label.naip import (
    parse_item_id,
    provenance_date,
    read_naip,
    window_quad,
)

QUAD = "4407159"
EPSG = 26919
UNIT = f"q{QUAD}"


def _window(quarter: str, dx: float = 0.0, half: float = 150.0):
    """A square in UTM, ``dx`` metres east of the centre of a quarter quad."""
    west, south, east, north = quarter_quad_bounds(QUAD, quarter)
    to_utm = Transformer.from_crs(4326, EPSG, always_xy=True)
    x, y = to_utm.transform((west + east) / 2, (south + north) / 2)
    return (x + dx - half, y - half, x + dx + half, y + half)


def _seam_x() -> float:
    """Easting of the seam between the south-west and south-east quarters."""
    west, south, _, north = quarter_quad_bounds(QUAD, "se")
    to_utm = Transformer.from_crs(4326, EPSG, always_xy=True)
    return to_utm.transform(west, (south + north) / 2)[0]


ITEMS = [
    f"nh_m_{QUAD}_sw_19_060_20180915_20181211",
    f"nh_m_{QUAD}_se_19_060_20180702_20181211",
    f"nh_m_{QUAD}_nw_19_060_20180702_20181211",
    f"nh_m_{QUAD}_ne_19_060_20180702_20181211",
]


@pytest.mark.parametrize(
    "item_id, expected",
    [
        (
            "nh_m_4407152_nw_19_1_20110926_20111114",
            ("4407152", "nw", "2011-09-26", 1.0),
        ),
        (
            "ri_m_4107127_se_19_.6_20160703_20160914",
            ("4107127", "se", "2016-07-03", 0.6),
        ),
        (
            "ny_m_4407462_se_18_.5_20150523_20151109",
            ("4407462", "se", "2015-05-23", 0.5),
        ),
        ("nh_m_4407152_nw_19_h_20160823", ("4407152", "nw", "2016-08-23", None)),
        ("me_m_4506921_nw_19_060_20210816", ("4506921", "nw", "2021-08-16", 0.6)),
        ("m_4506921_nw_19_030_20250815", ("4506921", "nw", "2025-08-15", 0.3)),
    ],
)
def test_item_ids_of_both_services_parse(item_id: str, expected: tuple) -> None:
    assert parse_item_id(item_id) == expected


def test_a_name_without_a_quad_is_not_an_item() -> None:
    assert parse_item_id("NAIP_q4407159_20180801_0p6m.tif") is None


def test_window_date_comes_from_its_quad() -> None:
    found = window_quad(ITEMS, EPSG, _window("se"))
    assert found.date == "2018-07-02" and found.item_id == ITEMS[1]
    assert found.native_m == 0.6 and found.date_share == 1.0
    assert found.other_dates == ()
    assert window_quad(ITEMS, EPSG, _window("sw")).date == "2018-09-15"


def test_the_first_listed_quad_fills_the_overlap() -> None:
    """A mosaic keeps the first quad that reaches a pixel, past its nominal edge."""
    x = _seam_x()
    y0, y1 = _window("se")[1], _window("se")[3]
    beside = (x + 20.0, y0, x + 120.0, y1)  # east of the seam, inside the overlap
    assert window_quad(ITEMS, EPSG, beside).date == "2018-09-15"
    assert window_quad(ITEMS, EPSG, beside, overlap_m=0.0).date == "2018-07-02"
    # The other order gives the other flight.
    assert window_quad(ITEMS[::-1], EPSG, beside).date == "2018-07-02"


def test_a_window_on_a_seam_records_both_dates() -> None:
    x = _seam_x()
    y0, y1 = _window("se")[1], _window("se")[3]
    across = (x - 150.0, y0, x + 450.0, y1)
    found = window_quad(ITEMS, EPSG, across, overlap_m=0.0)
    assert found.date == "2018-07-02" and found.other_dates == ("2018-09-15",)
    assert 0.6 < found.date_share < 0.9


def test_points_outside_the_mosaic_do_not_count() -> None:
    x = _seam_x()
    y0, y1 = _window("se")[1], _window("se")[3]
    across = (x - 300.0, y0, x + 300.0, y1)
    east_only = (x, y0 - 1000.0, x + 5000.0, y1 + 1000.0)
    found = window_quad(ITEMS, EPSG, across, overlap_m=0.0, extent=east_only)
    assert found.date == "2018-07-02" and found.date_share == 1.0
    assert window_quad(ITEMS, EPSG, across, extent=(0.0, 0.0, 1.0, 1.0)) is None


def test_no_listed_quad_reaches_the_window() -> None:
    far = [i.replace(QUAD, "3511151") for i in ITEMS]
    assert window_quad(far, EPSG, _window("se")) is None
    assert window_quad(["not an item"], EPSG, _window("se")) is None


def _mosaic(data_root: Path, bounds, sidecar: dict | None) -> Path:
    folder = data_root / "naip" / UNIT / "2018"
    folder.mkdir(parents=True)
    path = folder / f"NAIP_{UNIT}_20180801_0p6m.tif"
    x0, y0, x1, y1 = bounds
    width, height = round((x1 - x0) / 0.6) + 40, round((y1 - y0) / 0.6) + 40
    data = np.random.default_rng(0).integers(40, 200, (4, height, width), np.uint8)
    transform = from_origin(x0 - 12.0, y1 + 12.0, 0.6, 0.6)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=4,
        dtype="uint8",
        crs=f"EPSG:{EPSG}",
        transform=transform,
    ) as dst:
        dst.write(data)
    if sidecar is not None:
        full = {
            "crs": f"EPSG:{EPSG}",
            "transform": list(transform)[:6],
            "shape": [height, width],
            **sidecar,
        }
        path.with_suffix(".provenance.json").write_text(json.dumps(full))
    return path


def _snap(bounds):
    return tuple(round(v / 0.6) * 0.6 for v in bounds)


def test_a_mosaic_window_takes_the_date_of_its_quad(tmp_path: Path) -> None:
    """The file name holds the unit's median date. The sidecar holds the quad's."""
    bounds = _snap(_window("se"))
    path = _mosaic(tmp_path, bounds, {"item_ids": ITEMS, "date_tag": "20180801"})
    read = read_naip(
        "local", unit=UNIT, year=2018, epsg=EPSG, bounds=bounds, data_root=tmp_path
    )
    assert read.date == "2018-07-02" and read.date_rule == "quad"
    assert read.quad == ITEMS[1] and read.native_m == 0.6
    assert read.item_ids == (path.name,) and not read.mixed_dates
    assert provenance_date(path, EPSG, bounds).date == "2018-07-02"


@pytest.mark.parametrize("sidecar", [None, {"item_ids": ["unreadable"]}, {}])
def test_without_a_usable_sidecar_the_file_name_gives_the_date(
    tmp_path: Path, sidecar: dict | None, caplog: pytest.LogCaptureFixture
) -> None:
    bounds = _snap(_window("se"))
    _mosaic(tmp_path, bounds, sidecar)
    with caplog.at_level("WARNING"):
        read = read_naip(
            "local", unit=UNIT, year=2018, epsg=EPSG, bounds=bounds, data_root=tmp_path
        )
    assert read.date == "2018-08-01" and read.date_rule == "mosaic_name"
    assert read.date_share is None and not read.mixed_dates


def test_pixels_outside_keep_stay_empty(tmp_path: Path) -> None:
    bounds = _snap(_window("se"))
    _mosaic(tmp_path, bounds, {"item_ids": ITEMS})
    size = round((bounds[2] - bounds[0]) / 0.6)
    keep = np.zeros((size, size), bool)
    keep[:, : size // 2] = True
    read = read_naip(
        "local",
        unit=UNIT,
        year=2018,
        epsg=EPSG,
        bounds=bounds,
        data_root=tmp_path,
        keep=keep,
    )
    assert read.data[:, :, size // 2 :].max() == 0
    assert read.data[:, :, : size // 2].any(axis=0).all()
    assert read.coverage == pytest.approx(0.5, abs=0.01)
