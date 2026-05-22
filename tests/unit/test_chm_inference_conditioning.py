"""Unit tests for csdv_core.chm_inference.conditioning."""

from __future__ import annotations

from pathlib import Path

import pytest

from csdv_core.chm_inference.conditioning import (
    REQUIRED_RASTERS,
    ConditioningPaths,
    conditioning_paths,
    resolve_conditioning_dir,
    validate,
)
from csdv_core.io.paths import ProjectPaths


def _make_paths(root: Path) -> ProjectPaths:
    """Build a ProjectPaths with all three roots under ``root``."""
    return ProjectPaths(
        data_root=root / "data",
        results_root=root / "results",
        cache_root=root / "cache",
    )


def test_validate_raises_when_rasters_missing(tmp_path: Path) -> None:
    """validate() lists missing rasters in the error message."""
    paths = _make_paths(tmp_path)
    cp = conditioning_paths(paths)
    cp.root.mkdir(parents=True, exist_ok=True)
    (cp.root / "elevation.tif").write_bytes(b"\x00")
    (cp.root / "nlcd.tif").write_bytes(b"\x00")

    with pytest.raises(FileNotFoundError) as excinfo:
        validate(paths)
    msg = str(excinfo.value)
    assert "climate_pca.tif" in msg
    assert "soil_pca.tif" in msg
    assert "ecoregion.tif" in msg


def test_validate_succeeds_when_all_present(tmp_path: Path) -> None:
    """validate() returns a ConditioningPaths when all 5 rasters exist."""
    paths = _make_paths(tmp_path)
    cp = conditioning_paths(paths)
    cp.root.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_RASTERS:
        (cp.root / name).write_bytes(b"\x00")

    out = validate(paths)
    assert isinstance(out, ConditioningPaths)
    assert out.root == cp.root
    assert out.elevation.name == "elevation.tif"


def test_required_rasters_count_and_resolver(tmp_path: Path) -> None:
    """Sanity checks on REQUIRED_RASTERS and the directory resolver."""
    assert len(REQUIRED_RASTERS) == 5
    paths = _make_paths(tmp_path)
    assert resolve_conditioning_dir(paths) == paths.chm_model_dir() / "conditioning"
