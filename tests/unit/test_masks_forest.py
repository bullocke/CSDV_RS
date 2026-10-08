"""Tests for csdv_core.masks.forest and csdv_core.io.forest_masks."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from affine import Affine

from csdv_core.config import ForestMaskProfile
from csdv_core.io.forest_masks import export_forest_masks
from csdv_core.masks import forest as fm


def test_block_cover_matches_hand_count():
    chm = np.zeros((4, 4), "float32")
    chm[:2, :2] = 10.0  # top-left block fully canopy
    chm[0, 2] = 10.0  # top-right block 1 of 4
    res = fm.chm_forest_mask(chm, block_px=2, min_cover=0.5)
    assert res.mask.tolist() == [[1, 0], [0, 0]]
    cover, _ = fm.block_fraction(chm > 5, np.isfinite(chm), 2)
    np.testing.assert_allclose(cover, [[1.0, 0.25], [0.0, 0.0]])


def test_cover_ignores_nodata_pixels():
    chm = np.full((2, 2), 10.0, "float32")
    chm[0, 0] = np.nan
    chm[0, 1] = 1.0
    cover, valid = fm.block_fraction(np.nan_to_num(chm) > 5, np.isfinite(chm), 2)
    assert valid.all()
    assert cover[0, 0] == pytest.approx(2 / 3)


def test_nodata_block_is_coded_255():
    chm = np.full((2, 4), np.nan, "float32")
    chm[:, 2:] = 20.0
    res = fm.chm_forest_mask(chm, block_px=2)
    assert res.mask.tolist() == [[fm.NODATA, fm.FOREST]]
    assert res.forest_fraction == 1.0


def test_pixel_mode_is_height_threshold():
    chm = np.array([[4.9, 5.1], [np.nan, 30.0]], "float32")
    res = fm.chm_forest_mask(chm, block_px=1, height_m=5.0)
    assert res.mask.tolist() == [[0, 1], [fm.NODATA, 1]]


def test_clean_mask_removes_specks_and_fills_holes():
    m = np.zeros((10, 10), bool)
    m[2:8, 2:8] = True
    m[4, 4] = False  # one-pixel hole
    m[0, 9] = True  # one-pixel speck
    out = fm.clean_mask(m, min_patch_px=4, fill_holes_px=4)
    assert out[4, 4] and not out[0, 9]


def test_robust_z_uses_reference_only():
    values = np.array([0.5, 0.6, 0.7, -1.0, -1.0], "float32")
    ref = np.array([True, True, True, False, False])
    s = fm.robust_stats(values, ref)
    assert s.median == pytest.approx(0.6)
    assert s.scale == pytest.approx(1.4826 * 0.1, rel=1e-4)
    res = fm.ndvi_forest_mask(
        values[None, :], threshold=-3, mode="z", stats=s, block_px=1
    )
    assert res.mask.tolist() == [[1, 1, 1, 0, 0]]


def test_ndvi_matches_cell_builder_formula():
    red = np.array([[0, 50]], "uint8")
    nir = np.array([[0, 150]], "uint8")
    np.testing.assert_allclose(fm.ndvi(red, nir), [[0.0, 0.5]])


def test_combine_masks_propagates_nodata():
    a = np.array([[1, 1, 0, 255]], "uint8")
    b = np.array([[1, 0, 0, 1]], "uint8")
    assert fm.combine_masks(a, b, "and").tolist() == [[1, 0, 0, 255]]
    assert fm.combine_masks(a, b, "or").tolist() == [[1, 1, 0, 255]]


def test_params_tag_is_stable_and_order_free():
    assert fm.params_tag({"a": 1, "b": 2}) == fm.params_tag({"b": 2, "a": 1})
    assert fm.params_tag({"a": 1}) != fm.params_tag({"a": 2})


def test_profile_requires_threshold_unless_otsu():
    with pytest.raises(ValueError):
        ForestMaskProfile(ndvi_mode="raw", ndvi_threshold=None)
    ForestMaskProfile(ndvi_mode="otsu", ndvi_threshold=None)


def _write(path, data, nodata=None):
    count = 1 if data.ndim == 2 else data.shape[0]
    h, w = data.shape[-2:]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=count,
        dtype=data.dtype,
        crs="EPSG:26916",
        transform=Affine(0.6, 0, 500000, 0, -0.6, 4300000),
        nodata=nodata,
    ) as dst:
        dst.write(data if data.ndim == 3 else data[None])


def test_export_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    chm_paths, naip_paths = {}, {}
    for year in (2016, 2018):
        chm = np.full((32, 32), 20.0, "float32")
        chm[:16, :16] = 1.0  # one open cell
        chm[-1, -1] = -9999.0
        naip = rng.integers(20, 60, (4, 32, 32)).astype("uint8")
        naip[3] = np.where(chm > 5, 150, 40)
        chm_paths[year] = tmp_path / f"chm_{year}.tif"
        naip_paths[year] = tmp_path / f"naip_{year}.tif"
        _write(chm_paths[year], chm, nodata=-9999.0)
        _write(naip_paths[year], naip)
    prof = ForestMaskProfile(
        source="and", block_px=16, ndvi_mode="raw", ndvi_threshold=0.3
    )
    out = export_forest_masks(chm_paths, naip_paths, tmp_path / "out", prof)
    layers = {(e.layer, e.year) for e in out}
    assert (
        ("chm", 2016) in layers and ("ndvi", 2018) in layers and ("and", 2018) in layers
    )
    chm_file = next(e.path for e in out if e.layer == "chm" and e.year == 2016)
    with rasterio.open(chm_file) as src:
        arr = src.read(1)
        assert src.nodata == 255
        assert src.transform.a == pytest.approx(9.6)
    assert arr.tolist() == [[0, 1], [1, 1]]
    assert (tmp_path / "out" / "masks_index.csv").exists()
