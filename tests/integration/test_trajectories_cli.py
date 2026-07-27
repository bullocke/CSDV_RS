"""CLI integration test for ``csdv classify-trajectories``.

Writes a tiny multi-date stage + metric stack to disk under a temp
results root, swaps in a minimal trajectories config, invokes the CLI,
and asserts outputs exist with expected dtypes and codes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import yaml
from affine import Affine
from click.testing import CliRunner

from csdv_core.config import reload_config
from csdv_core.trajectories.cli import cli as traj_cli


def _t() -> Affine:
    return Affine.translation(0.0, 0.0) * Affine.scale(0.6, -0.6)


def _write_uint8(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="uint8",
        transform=_t(),
        crs="EPSG:5070",
        nodata=0,
    ) as dst:
        dst.write(arr.astype("uint8"), 1)


def _write_float(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        transform=_t(),
        crs="EPSG:5070",
        nodata=float("nan"),
    ) as dst:
        dst.write(arr.astype("float32"), 1)


def test_classify_trajectories_cli_smoke(tmp_path: Path, monkeypatch) -> None:
    # Point all roots at tmp_path.
    monkeypatch.setenv("CSDV_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("CSDV_RESULTS_ROOT", str(tmp_path / "results"))
    monkeypatch.setenv("CSDV_CACHE_ROOT", str(tmp_path / "cache"))

    site = "TEST"
    window_m = 100
    years = [2014, 2018]
    results = tmp_path / "results"

    # Stage rasters: year-0 ESE (3) everywhere, year-1 LSE (4) everywhere.
    for year, code in zip(years, [3, 4], strict=True):
        _write_uint8(
            results / "stages" / site / str(year) / "100m" / "stage.tif",
            np.full((4, 4), code, dtype="uint8"),
        )

    # Metric rasters: stable low gap and crown_cv across both years (DS1 hits),
    # plus crown_fraction high enough that LC1 cannot match.
    for year in years:
        mdir = results / "metrics" / site / str(year) / "100m"
        _write_float(mdir / "gap_fraction.tif", np.full((4, 4), 0.05))
        _write_float(mdir / "crown_cv.tif", np.full((4, 4), 0.05))
        _write_float(mdir / "glcm_texture.tif", np.full((4, 4), 0.10))
        _write_float(mdir / "crown_fraction.tif", np.full((4, 4), 0.80))
        # LC1 also references ndvi_seasonal_amplitude (placeholder), but its
        # value is null in trajectories.yaml so the predicate never fires;
        # however the metric must still exist on disk because the loader
        # reads every required name. Provide a stand-in raster.
        _write_float(mdir / "ndvi_seasonal_amplitude.tif", np.zeros((4, 4)))
        # Other metrics referenced by placeholder rules:
        for name in [
            "ndvi_trend",
            "shrub_fraction",
            "wetness_persistence",
            "impervious_fraction",
            "ndvi_mean",
            "linearity_index",
            "gap_persistence",
            "row_directionality",
        ]:
            _write_float(mdir / f"{name}.tif", np.zeros((4, 4)))

    reload_config()
    runner = CliRunner()
    result = runner.invoke(
        traj_cli,
        [
            "--site",
            site,
            "--window-m",
            str(window_m),
            "--years",
            "2014,2018",
        ],
    )
    if result.exit_code != 0:
        raise AssertionError(
            f"CLI failed: exit={result.exit_code}\n"
            f"output:\n{result.output}\n"
            f"exception: {result.exception!r}"
        )

    out_dir = results / "trajectories" / site / "100m"
    assert (out_dir / "trajectory.tif").exists()
    assert (out_dir / "trajectory_n_predicates.tif").exists()
    assert (out_dir / "trajectory_n_dates.tif").exists()
    assert (out_dir / "trajectories_manifest.yaml").exists()

    with rasterio.open(out_dir / "trajectory.tif") as src:
        traj = src.read(1)
    # DS1 has code 1 in trajectories.yaml. All pixels should match DS1.
    assert (traj == 1).all()

    manifest = yaml.safe_load((out_dir / "trajectories_manifest.yaml").read_text())
    assert manifest["site"] == site
    assert manifest["years"] == years
    assert manifest["window_m"] == 100.0
