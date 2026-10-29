"""Tests for the download set of a grid run (``csdv grid export``).

A synthetic run folder and synthetic NAIP and CHM years are written to a
temporary store, so the byte encodings, the COG output and the copied files
are checked without CHPC data.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from csdv_core.grid.export import (
    NODATA_U8,
    encode_chm,
    encode_max_probability,
    encode_onset,
    export_run,
    onset_colormap,
)
from csdv_core.grid.units import GridPaths

CRS = "EPSG:26919"


def test_encode_onset():
    onset = np.array([[2018, 0, -1], [2025, 2011, 0]], dtype=np.int16)
    np.testing.assert_array_equal(
        encode_onset(onset), [[18, 0, NODATA_U8], [25, 11, 0]]
    )


def test_onset_colormap_marks_none_and_never_monitored():
    table = onset_colormap([0, 18, 21, NODATA_U8])
    assert table[NODATA_U8][3] == 0
    assert table[0][3] == 255
    assert table[18] != table[21]


def test_encode_max_probability():
    prob = np.array(
        [[[0.2, np.nan], [np.nan, 0.999]], [[0.7, np.nan], [0.1, np.nan]]],
        dtype="float32",
    )
    np.testing.assert_array_equal(
        encode_max_probability(prob), [[70, NODATA_U8], [10, 100]]
    )


def test_encode_chm_steps_clips_and_masks():
    heights = np.ma.masked_array([0.1, 12.3, 80.0, 5.0], mask=[0, 0, 0, 1])
    np.testing.assert_array_equal(encode_chm(heights), [0, 49, 254, NODATA_U8])


def _write(path, data, *, nodata=None, transform=None, mask=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = data if data.ndim == 3 else data[np.newaxis]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        crs=CRS,
        transform=transform or from_origin(290000, 4890000, 19.2, 19.2),
        nodata=nodata,
    ) as dst:
        dst.write(data)
        if mask is not None:
            dst.write_mask(mask)


@pytest.fixture()
def store(tmp_path):
    paths = GridPaths(
        grid_dir=tmp_path / "grid",
        export_dir=tmp_path / "exports",
        data_root=tmp_path / "data",
        results_root=tmp_path / "results",
    )
    run = paths.run_dir("q4407159", "default")
    onset = np.zeros((40, 40), dtype=np.int16)
    onset[:10] = 2018
    onset[-5:] = -1
    _write(run / "detection" / "onset_year.tif", onset)
    prob = np.full((2, 40, 40), np.nan, dtype="float32")
    prob[0, :20] = 0.8
    _write(run / "detection" / "probability.tif", prob, nodata=np.nan)
    (run / "detection" / "objects.csv").write_text("id\n1\n", encoding="utf-8")
    (run / "figures").mkdir()
    (run / "figures" / "onset.png").write_bytes(b"png")
    for name in ("summary.json", "run_params.json"):
        (run / name).write_text("{}", encoding="utf-8")
    (run / "report.md").write_text("# report\n", encoding="utf-8")

    fine = from_origin(290000, 4890000, 0.6, 0.6)
    chm = np.full((100, 100), 1500, dtype=np.uint16)
    chm[:, :10] = 65535
    _write(
        paths.chm_year_dir("q4407159", 2021) / "NAIP_q_20210701_0p6m_chm.tif",
        chm,
        nodata=65535,
        transform=fine,
    )
    naip = np.full((4, 100, 100), 120, dtype=np.uint8)
    mask = np.full((100, 100), 255, dtype=np.uint8)
    mask[:20] = 0
    _write(
        paths.naip_year_dir("q4407159", 2021) / "NAIP_q_20210701_0p6m.tif",
        naip,
        transform=fine,
        mask=mask,
    )
    return paths


def test_export_run_writes_cogs_copies_and_quicklooks(store):
    result = export_run(store, "q4407159", "default")
    out = store.export_run_dir("q4407159", "default")
    assert result.export_dir == out

    with rasterio.open(out / "onset_year_u8.tif") as src:
        assert src.dtypes[0] == "uint8"
        assert src.nodata == NODATA_U8
        assert src.colormap(1)[18][3] == 255
        codes = src.read(1)
    assert set(np.unique(codes)) == {0, 18, NODATA_U8}
    with rasterio.open(out / "max_probability_u8.tif") as src:
        assert src.read(1)[0, 0] == 80
        assert src.read(1)[30, 0] == NODATA_U8

    for name in ("summary.json", "report.md", "objects.csv", "figures/onset.png"):
        assert (out / name).exists(), name
    assert json.loads((out / "export.json").read_text())["unit"] == "q4407159"

    looks = store.quicklook_dir("q4407159")
    with rasterio.open(looks / "chm_2021_3m.tif") as src:
        assert src.res == pytest.approx((3.0, 3.0))
        assert src.scales[0] == pytest.approx(0.25)
        chm = src.read(1)
    assert chm[0, 0] == NODATA_U8
    assert chm[0, 10] == 60  # 15 m in 0.25 m steps
    with rasterio.open(looks / "naip_2021_3m_rgb.tif") as src:
        assert src.count == 3
        assert src.profile["compress"].lower() == "jpeg"
        valid = src.dataset_mask()
    assert valid[0].max() == 0 and valid[-1].min() == 255


def test_export_needs_a_finished_run(store):
    with pytest.raises(FileNotFoundError):
        export_run(store, "q4407159", "missing")
