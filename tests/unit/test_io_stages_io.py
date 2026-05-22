"""Tests for csdv_core.io.stages_io: multi-date cube readers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from csdv_core.io.paths import ProjectPaths
from csdv_core.io.stages_io import read_metric_cube, read_stage_cube


def _make_paths(tmp_path: Path) -> ProjectPaths:
    return ProjectPaths(
        data_root=tmp_path / "data",
        results_root=tmp_path / "results",
        cache_root=tmp_path / "cache",
    )


def _write_uint8(path: Path, arr: np.ndarray, transform: Affine) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="uint8",
        transform=transform,
        crs="EPSG:5070",
        nodata=0,
    ) as dst:
        dst.write(arr.astype("uint8"), 1)


def _write_float(path: Path, arr: np.ndarray, transform: Affine) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        transform=transform,
        crs="EPSG:5070",
        nodata=float("nan"),
    ) as dst:
        dst.write(arr.astype("float32"), 1)


def _t() -> Affine:
    return Affine.translation(0.0, 0.0) * Affine.scale(0.6, -0.6)


def test_read_stage_cube_stacks_in_order(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    a = np.array([[1, 2], [3, 4]], dtype="uint8")
    b = np.array([[4, 4], [4, 4]], dtype="uint8")
    _write_uint8(paths.stages_dir("SCBI", 2014, 100) / "stage.tif", a, _t())
    _write_uint8(paths.stages_dir("SCBI", 2018, 100) / "stage.tif", b, _t())

    cube, grid = read_stage_cube(paths, "SCBI", [2014, 2018], 100)
    assert cube.shape == (2, 2, 2)
    assert cube.dtype == np.uint8
    np.testing.assert_array_equal(cube[0], a)
    np.testing.assert_array_equal(cube[1], b)
    assert grid.crs == "EPSG:5070"


def test_read_stage_cube_requires_two_years(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    with pytest.raises(ValueError, match="at least 2 years"):
        read_stage_cube(paths, "SCBI", [2014], 100)


def test_read_stage_cube_missing_year_raises(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    a = np.zeros((2, 2), dtype="uint8")
    _write_uint8(paths.stages_dir("SCBI", 2014, 100) / "stage.tif", a, _t())
    with pytest.raises(FileNotFoundError):
        read_stage_cube(paths, "SCBI", [2014, 2018], 100)


def test_read_stage_cube_shape_mismatch_raises(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    _write_uint8(
        paths.stages_dir("SCBI", 2014, 100) / "stage.tif",
        np.zeros((2, 2), dtype="uint8"),
        _t(),
    )
    _write_uint8(
        paths.stages_dir("SCBI", 2018, 100) / "stage.tif",
        np.zeros((3, 3), dtype="uint8"),
        _t(),
    )
    with pytest.raises(ValueError):
        read_stage_cube(paths, "SCBI", [2014, 2018], 100)


def test_read_metric_cube_stacks_each_metric(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    for year, val in [(2014, 0.10), (2018, 0.20)]:
        d = paths.metrics_dir("SCBI", year, 100)
        _write_float(d / "gap_fraction.tif", np.full((2, 2), val), _t())
        _write_float(d / "crown_width_cv.tif", np.full((2, 2), val * 2), _t())

    cubes, grid = read_metric_cube(
        paths, "SCBI", [2014, 2018], 100, ["gap_fraction", "crown_width_cv"]
    )
    assert set(cubes) == {"gap_fraction", "crown_width_cv"}
    assert cubes["gap_fraction"].shape == (2, 2, 2)
    assert cubes["gap_fraction"][0, 0, 0] == pytest.approx(0.10)
    assert cubes["gap_fraction"][1, 0, 0] == pytest.approx(0.20)
    assert grid.crs == "EPSG:5070"


def test_read_metric_cube_empty_metric_list_returns_empty(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    d = paths.metrics_dir("SCBI", 2014, 100)
    _write_float(d / "gap_fraction.tif", np.zeros((2, 2)), _t())
    cubes, grid = read_metric_cube(paths, "SCBI", [2014, 2018], 100, [])
    assert cubes == {}
    assert grid.crs == "EPSG:5070"
