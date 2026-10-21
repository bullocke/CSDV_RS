"""Tests for masking NAIP gaps in a predicted CHM.

Small synthetic rasters stand in for a NAIP mosaic with a gap (zero in every
band, as the downloaders write it) and the CHM the model predicted over it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from csdv_core.chm_inference.naip_gaps import (
    ORIGINAL_SUFFIX,
    chm_nodata_value,
    coarse_block_px,
    edge_band,
    mask_naip_gaps,
)

PIXEL_M = 0.6
ROWS, COLS = 200, 300
GAP_COLS = 101  # not a multiple of any coarse cell, so cell edges are exercised
CHM_VALUE = 1500
NODATA = 65535
ORIGIN = (500_000.0, 5_000_000.0)


def _write(path: Path, data: np.ndarray, *, pixel_m: float = PIXEL_M, **extra) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "count": data.shape[0],
        "dtype": data.dtype.name,
        "crs": "EPSG:26919",
        "transform": from_origin(*ORIGIN, pixel_m, pixel_m),
        "compress": "DEFLATE",
        "tiled": True,
        "blockxsize": 16,
        "blockysize": 16,
        **extra,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


def _naip(tmp_path: Path, gap_cols: int = GAP_COLS) -> Path:
    rng = np.random.default_rng(0)
    data = rng.integers(1, 256, size=(4, ROWS, COLS), dtype="uint8")
    data[:, :, :gap_cols] = 0
    return _write(tmp_path / "naip" / "NAIP_Site_20230922_0p6m.tif", data)


def _chm(tmp_path: Path, **extra) -> Path:
    data = np.full((1, ROWS, COLS), CHM_VALUE, dtype="uint16")
    data[0, 0, -1] = NODATA  # already nodata before masking
    path = tmp_path / "chm" / "NAIP_Site_20230922_0p6m_chm.tif"
    _write(path, data, nodata=NODATA, **extra)
    with rasterio.open(path, "r+") as dst:
        dst.update_tags(model="naip-chm")
        dst.update_tags(1, units="cm")
    return path


def _read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def test_gap_pixels_become_nodata_and_imagery_is_untouched(tmp_path):
    chm = _chm(tmp_path)
    result = mask_naip_gaps(_naip(tmp_path), chm, edge_m=0.0)

    out = _read(chm)
    assert (out[:, :GAP_COLS] == NODATA).all()
    assert (out[1:, GAP_COLS:] == CHM_VALUE).all()
    assert result.rewritten
    assert result.gap_pixels == ROWS * GAP_COLS
    assert result.edge_pixels == 0
    assert result.newly_masked == ROWS * GAP_COLS
    assert result.gap_share == pytest.approx(GAP_COLS / COLS)


def test_masked_chm_keeps_its_profile_and_tags(tmp_path):
    chm = _chm(tmp_path)
    mask_naip_gaps(_naip(tmp_path), chm, edge_m=0.0)
    with rasterio.open(chm) as src:
        assert src.dtypes[0] == "uint16"
        assert src.nodata == NODATA
        assert src.compression.name.upper() == "DEFLATE"
        assert src.block_shapes[0] == (16, 16)
        assert src.tags()["model"] == "naip-chm"
        assert src.tags(1)["units"] == "cm"
    assert not list(chm.parent.glob("*.partial"))


def test_edge_band_masks_imagery_near_a_gap(tmp_path):
    chm = _chm(tmp_path)
    edge_m = 12.0
    result = mask_naip_gaps(_naip(tmp_path), chm, edge_m=edge_m)

    out = _read(chm)
    band_px = int(edge_m / PIXEL_M)  # 20 columns
    cell_px = coarse_block_px(edge_m, PIXEL_M)
    # Inside the band, allowing one coarse cell of slack at its far side.
    assert (out[:, GAP_COLS : GAP_COLS + band_px - cell_px] == NODATA).all()
    # Well beyond the band nothing changes.
    assert (out[1:, GAP_COLS + band_px + 2 * cell_px :] == CHM_VALUE).all()
    assert result.edge_pixels == pytest.approx(ROWS * band_px, rel=0.15)


def test_a_stray_black_pixel_does_not_open_a_hole(tmp_path):
    naip = _naip(tmp_path, gap_cols=0)
    with rasterio.open(naip, "r+") as dst:
        dst.write(np.zeros((4, 1, 1), dtype="uint8"), window=((100, 101), (150, 151)))
    chm = _chm(tmp_path)
    result = mask_naip_gaps(naip, chm, edge_m=12.0)

    out = _read(chm)
    assert out[100, 150] == NODATA
    assert result.newly_masked == 1
    assert result.edge_pixels == 0


def test_naip_without_gaps_leaves_the_chm_alone(tmp_path):
    chm = _chm(tmp_path)
    before = chm.read_bytes()
    result = mask_naip_gaps(_naip(tmp_path, gap_cols=0), chm, edge_m=50.0)
    assert not result.rewritten
    assert result.newly_masked == 0
    assert chm.read_bytes() == before


def test_keep_original_moves_the_unmasked_chm_out_of_tif_globs(tmp_path):
    chm = _chm(tmp_path)
    naip = _naip(tmp_path)
    result = mask_naip_gaps(naip, chm, edge_m=0.0, keep_original=True)

    original = chm.with_name(chm.name + ORIGINAL_SUFFIX)
    assert result.original_path == original
    assert (_read(original)[1:, :] == CHM_VALUE).all()
    assert list(chm.parent.glob("*.tif")) == [chm]
    with pytest.raises(FileExistsError, match="masked already"):
        mask_naip_gaps(naip, chm, edge_m=0.0, keep_original=True)


def test_out_path_leaves_the_input_chm_unchanged(tmp_path):
    chm = _chm(tmp_path)
    out_path = tmp_path / "masked.tif"
    result = mask_naip_gaps(_naip(tmp_path), chm, edge_m=0.0, out_path=out_path)
    assert result.chm_path == out_path
    assert (_read(chm)[1:, :] == CHM_VALUE).all()
    assert (_read(out_path)[:, :GAP_COLS] == NODATA).all()


def test_chm_on_a_coarser_grid_is_masked_through_a_warp(tmp_path):
    data = np.full((1, ROWS // 2, COLS // 2), CHM_VALUE, dtype="uint16")
    chm = _write(tmp_path / "chm_coarse.tif", data, pixel_m=2 * PIXEL_M, nodata=NODATA)
    mask_naip_gaps(_naip(tmp_path), chm, edge_m=0.0)

    out = _read(chm)
    assert (out[:, : GAP_COLS // 2] == NODATA).all()
    assert (out[:, GAP_COLS // 2 + 1 :] == CHM_VALUE).all()


def test_strip_height_does_not_change_the_result(tmp_path):
    """30 m gives 5-pixel coarse cells, which straddle the 48-row strips."""
    naip = _naip(tmp_path)
    with rasterio.open(naip, "r+") as dst:  # horizontal gap edge inside a cell
        dst.write(np.zeros((4, 47, 100), dtype="uint8"), window=((0, 47), (101, 201)))
    chm = _chm(tmp_path)
    single = mask_naip_gaps(naip, chm, edge_m=30.0, out_path=tmp_path / "a.tif")
    strips = mask_naip_gaps(
        naip, chm, edge_m=30.0, out_path=tmp_path / "b.tif", strip_rows=48
    )
    assert np.array_equal(_read(single.chm_path), _read(strips.chm_path))
    assert single.edge_pixels == strips.edge_pixels


def test_negative_edge_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="edge_m"):
        mask_naip_gaps(_naip(tmp_path), _chm(tmp_path), edge_m=-1.0)


def test_edge_band_needs_a_mostly_gap_cell_to_start():
    share = np.zeros((5, 5))
    share[2, 2] = 0.4
    assert not edge_band(share, cell_m=1.0, edge_m=1.5).any()
    share[2, 2] = 0.6
    band = edge_band(share, cell_m=1.0, edge_m=1.0)
    assert band.sum() == 5  # the cell and its four neighbours at 1 m


@pytest.mark.parametrize(
    "dtype, nodata, expected",
    [("uint16", 65535.0, 65535.0), ("uint16", None, 65535.0), ("uint8", None, 255.0)],
)
def test_chm_nodata_value(dtype, nodata, expected):
    assert chm_nodata_value(dtype, nodata) == expected


def test_chm_nodata_value_is_nan_for_an_undeclared_float():
    assert np.isnan(chm_nodata_value("float32", None))
