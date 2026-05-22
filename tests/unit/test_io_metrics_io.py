"""Tests for :mod:`csdv_core.io.metrics_io`."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from csdv_core.io.metrics_io import (
    MANIFEST_NAME,
    read_manifest,
    read_metric_stack,
    write_metric_stack,
)
from csdv_core.metrics._result import make_result


def _make(name: str, transform=None, shape=(4, 4)):
    if transform is None:
        transform = Affine.translation(0.0, 0.0) * Affine.scale(30.0, -30.0)
    arr = np.full(shape, 0.5, dtype="float32")
    return make_result(
        name=name,
        array=arr,
        transform=transform,
        crs="EPSG:5070",
        window_m=30.0,
        params={"k": 1},
        units="fraction",
    )


def test_write_then_read_roundtrip(tmp_path) -> None:
    results = [_make("gap_fraction"), _make("crown_width_cv")]
    write_metric_stack(results, tmp_path)
    arrays, grid = read_metric_stack(tmp_path)
    assert set(arrays.keys()) == {"gap_fraction", "crown_width_cv"}
    assert arrays["gap_fraction"].dtype == np.float32
    assert arrays["gap_fraction"].shape == (4, 4)
    assert grid.crs == "EPSG:5070"
    assert grid.pixel_size_m == pytest.approx(30.0)


def test_manifest_written(tmp_path) -> None:
    results = [_make("gap_fraction")]
    write_metric_stack(results, tmp_path)
    assert (tmp_path / MANIFEST_NAME).exists()
    manifest = read_manifest(tmp_path)
    assert "metrics" in manifest
    entries = manifest["metrics"]
    assert entries[0]["metric"] == "gap_fraction"
    assert entries[0]["window_m"] == 30.0
    assert entries[0]["units"] == "fraction"


def test_read_manifest_missing_returns_empty(tmp_path) -> None:
    assert read_manifest(tmp_path) == {}


def test_read_metric_stack_no_files_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        read_metric_stack(tmp_path)


def test_read_metric_stack_grid_mismatch_raises(tmp_path) -> None:
    t1 = Affine.translation(0.0, 0.0) * Affine.scale(30.0, -30.0)
    t2 = Affine.translation(100.0, 0.0) * Affine.scale(30.0, -30.0)
    write_metric_stack([_make("gap_fraction", transform=t1)], tmp_path)
    write_metric_stack([_make("crown_width_cv", transform=t2)], tmp_path)
    with pytest.raises(ValueError, match="disagrees"):
        read_metric_stack(tmp_path)


def test_read_metric_stack_subset(tmp_path) -> None:
    write_metric_stack([_make("gap_fraction"), _make("crown_width_cv")], tmp_path)
    arrays, _ = read_metric_stack(tmp_path, metrics=["gap_fraction"])
    assert list(arrays.keys()) == ["gap_fraction"]
