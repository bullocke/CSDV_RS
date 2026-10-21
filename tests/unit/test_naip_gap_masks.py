"""Tests that NAIP gaps are flagged by the Planetary Computer mosaic and
carried into the CHM by ``predict_chm``.

No network, no GPU. Local GeoTIFFs stand in for the Planetary Computer quads,
and the upstream inference subprocess is replaced by one that writes a CHM of
constant height, as the model does over black fill.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from csdv_core.chm_inference import infer
from csdv_core.download.naip_pc import naip_mosaic

X0, Y0, RES = 500_000.0, 5_000_000.0, 0.6
ROWS, COLS = 120, 200


def _write(path: Path, data: np.ndarray, transform, **extra) -> Path:
    profile = {
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "count": data.shape[0],
        "dtype": data.dtype.name,
        "crs": "EPSG:26919",
        "transform": transform,
        **extra,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


def test_mosaic_masks_the_part_of_the_grid_no_quad_reaches(tmp_path):
    # One quad covering the eastern half of the target grid.
    quad = np.full((4, ROWS, COLS // 2), 90, dtype="uint8")
    quad_path = _write(
        tmp_path / "quad.tif", quad, from_origin(X0 + RES * COLS / 2, Y0, RES, RES)
    )
    item = SimpleNamespace(
        id="q1",
        assets={"image": SimpleNamespace(href=str(quad_path))},
        properties={"datetime": "2023-09-22T00:00:00Z", "gsd": RES},
    )
    out_path = tmp_path / "naip" / "NAIP_X_20230922_0p6m.tif"
    _, provenance = naip_mosaic(
        [item],
        out_path,
        dst_crs="EPSG:26919",
        dst_transform=from_origin(X0, Y0, RES, RES),
        dst_width=COLS,
        dst_height=ROWS,
    )

    with rasterio.open(out_path) as src:
        mask = src.dataset_mask()
        data = src.read()
    assert (mask[:, : COLS // 2] == 0).all()
    assert (mask[:, COLS // 2 :] == 255).all()
    assert (data[:, :, : COLS // 2] == 0).all()
    assert provenance["coverage_fraction"] == pytest.approx(0.5)
    assert not list(out_path.parent.glob("*.msk"))


@pytest.fixture()
def fake_inference(tmp_path, monkeypatch):
    """A NAIP with a western gap, a fake upstream repo, and a fake model run."""
    naip = np.full((4, ROWS, COLS), 120, dtype="uint8")
    naip[:, :, :80] = 0
    naip_path = _write(
        tmp_path / "NAIP_X_20230922_0p6m.tif", naip, from_origin(X0, Y0, RES, RES)
    )
    repo = tmp_path / "naip-chm"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "inference.py").touch()

    def _fake_run(cmd, cwd=None):
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        chm = np.full((1, ROWS, COLS), 900, dtype="uint16")
        _write(
            output_dir / f"{naip_path.stem}_chm.tif",
            chm,
            from_origin(X0, Y0, RES, RES),
            nodata=65535,
        )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(infer.subprocess, "run", _fake_run)
    return SimpleNamespace(naip=naip_path, repo=repo, out=tmp_path / "chm")


def _predicted(fake, **kwargs) -> np.ndarray:
    infer.predict_chm(
        fake.naip,
        fake.out,
        repo_dir=fake.repo,
        conditioning_dir=fake.repo,
        **kwargs,
    )
    with rasterio.open(fake.out / f"{fake.naip.stem}_chm.tif") as src:
        return src.read(1)


def test_predict_chm_sets_naip_gaps_to_nodata(fake_inference):
    chm = _predicted(fake_inference, gap_edge_m=0.0)
    assert (chm[:, :80] == 65535).all()
    assert (chm[:, 80:] == 900).all()


def test_predict_chm_edge_band_defaults_to_the_config(fake_inference):
    chm = _predicted(fake_inference)
    # metrics.yaml sets 50 m, about 83 columns at 0.6 m: the rest of the grid.
    assert (chm[:, : 80 + 70] == 65535).all()


def test_predict_chm_can_leave_the_chm_unmasked(fake_inference):
    chm = _predicted(fake_inference, mask_gaps=False)
    assert (chm == 900).all()
