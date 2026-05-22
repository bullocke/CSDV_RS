"""Tests for the Pass-1 metric orchestrator and ``csdv compute-metrics``."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from click.testing import CliRunner
from rasterio.transform import from_origin
from shapely.geometry import Point


def _write_chm(path: Path, *, size: int = 200, pixel_size: float = 0.5) -> None:
    rng = np.random.default_rng(42)
    data = rng.uniform(0.0, 25.0, size=(size, size)).astype("float32")
    transform = from_origin(0.0, size * pixel_size, pixel_size, pixel_size)
    profile = {
        "driver": "GTiff",
        "height": size,
        "width": size,
        "count": 1,
        "dtype": "float32",
        "transform": transform,
        "crs": "EPSG:5070",
        "nodata": -9999.0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _write_naip(path: Path, *, size: int = 200, pixel_size: float = 0.5) -> None:
    rng = np.random.default_rng(7)
    data = rng.integers(0, 255, size=(4, size, size), dtype="uint8")
    transform = from_origin(0.0, size * pixel_size, pixel_size, pixel_size)
    profile = {
        "driver": "GTiff",
        "height": size,
        "width": size,
        "count": 4,
        "dtype": "uint8",
        "transform": transform,
        "crs": "EPSG:5070",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def _write_crowns(path: Path) -> None:
    rng = np.random.default_rng(11)
    rows = []
    # Spread points across a 100x100 m extent to get crowns in each 50m window.
    for x, y in [(25.0, 75.0), (75.0, 75.0), (25.0, 25.0), (75.0, 25.0)]:
        for _ in range(5):
            rows.append(
                {
                    "crown_diam_m": float(rng.uniform(2.0, 8.0)),
                    "geometry": Point(x + rng.uniform(-5, 5), y + rng.uniform(-5, 5)),
                }
            )
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:5070")
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GPKG")


@pytest.fixture()
def fake_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Materialise minimal CSDV data + results trees for one site/year."""
    data = tmp_path / "data"
    results = tmp_path / "results"
    cache = tmp_path / "cache"
    monkeypatch.setenv("CSDV_DATA_ROOT", str(data))
    monkeypatch.setenv("CSDV_RESULTS_ROOT", str(results))
    monkeypatch.setenv("CSDV_CACHE_ROOT", str(cache))

    _write_chm(data / "naip_chm" / "SCBI" / "2018" / "chm.tif")
    _write_naip(data / "naip" / "SCBI" / "2018" / "naip.tif")
    _write_crowns(results / "crowns" / "SCBI" / "2018" / "crowns.gpkg")
    return tmp_path


def test_pass1_metrics_listed():
    from csdv_core.metrics.orchestrator import PASS1_METRICS

    assert set(PASS1_METRICS) == {
        "gap_fraction",
        "crown_fraction",
        "crown_cv",
        "glcm_texture",
    }


def test_unsupported_metric_raises(fake_project):
    from csdv_core.metrics.orchestrator import compute_for_window

    with pytest.raises(NotImplementedError, match="not yet wired"):
        compute_for_window("SCBI", 2018, 50.0, metric_names=["shrub_fraction"])


def test_compute_for_window_produces_chm_metrics(fake_project):
    from csdv_core.metrics.orchestrator import compute_for_window

    results = compute_for_window(
        "SCBI",
        2018,
        50.0,
        metric_names=["gap_fraction", "crown_fraction"],
    )
    assert [r.name for r in results] == ["gap_fraction", "crown_fraction"]
    for r in results:
        assert r.window_m == 50.0
        assert r.array.ndim == 2
        # 200 px * 0.5 m = 100 m, /50 m = 2x2 windows.
        assert r.array.shape == (2, 2)
        finite = np.isfinite(r.array)
        assert finite.all()
        assert ((r.array[finite] >= 0.0) & (r.array[finite] <= 1.0)).all()
    gap, crown = results
    # gap + crown == 1 where both finite.
    sums = gap.array + crown.array
    assert np.allclose(sums, 1.0, atol=1e-5)


def test_compute_for_window_all_pass1_metrics(fake_project):
    from csdv_core.metrics.orchestrator import PASS1_METRICS, compute_for_window

    results = compute_for_window("SCBI", 2018, 50.0)
    assert [r.name for r in results] == list(PASS1_METRICS)
    for r in results:
        assert r.array.shape == (2, 2)


def test_missing_chm_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CSDV_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("CSDV_RESULTS_ROOT", str(tmp_path / "results"))
    from csdv_core.metrics.orchestrator import compute_for_window

    with pytest.raises(FileNotFoundError, match="CHM required"):
        compute_for_window("SCBI", 2018, 50.0, metric_names=["gap_fraction"])


def test_cli_writes_metric_stack(fake_project):
    from csdv_core.io.paths import project_paths
    from csdv_core.metrics.cli import cli as compute_metrics_cli

    runner = CliRunner()
    result = runner.invoke(
        compute_metrics_cli,
        [
            "--site",
            "SCBI",
            "--year",
            "2018",
            "--window-m",
            "50",
            "--metrics",
            "gap_fraction,crown_fraction",
        ],
    )
    assert result.exit_code == 0, result.output
    out_dir = project_paths().metrics_dir("SCBI", 2018, 50)
    assert (out_dir / "manifest.yaml").exists()
    assert (out_dir / "gap_fraction.tif").exists()
    assert (out_dir / "crown_fraction.tif").exists()


def test_cli_refuses_to_overwrite_without_force(fake_project):
    from csdv_core.io.paths import project_paths
    from csdv_core.metrics.cli import cli as compute_metrics_cli

    out_dir = project_paths().metrics_dir("SCBI", 2018, 50)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.yaml").write_text("metrics: []\n")

    runner = CliRunner()
    result = runner.invoke(
        compute_metrics_cli,
        [
            "--site",
            "SCBI",
            "--year",
            "2018",
            "--window-m",
            "50",
            "--metrics",
            "gap_fraction",
        ],
    )
    assert result.exit_code != 0
    assert "already exists" in result.output

    # With --force, it should succeed.
    result2 = runner.invoke(
        compute_metrics_cli,
        [
            "--site",
            "SCBI",
            "--year",
            "2018",
            "--window-m",
            "50",
            "--metrics",
            "gap_fraction",
            "--force",
        ],
    )
    assert result2.exit_code == 0, result2.output
