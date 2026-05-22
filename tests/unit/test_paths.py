"""Tests for csdv_core.io.paths."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from csdv_core.io.paths import ProjectPaths, project_paths


def test_defaults_resolve_under_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("CSDV_DATA_ROOT", "CSDV_RESULTS_ROOT", "CSDV_CACHE_ROOT"):
        monkeypatch.delenv(var, raising=False)
    p = project_paths()
    assert isinstance(p, ProjectPaths)
    assert p.data_root.name == "data"
    assert p.results_root.name == "results"
    assert p.cache_root.name == ".cache"


def test_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CSDV_DATA_ROOT", str(tmp_path / "d"))
    monkeypatch.setenv("CSDV_RESULTS_ROOT", str(tmp_path / "r"))
    monkeypatch.setenv("CSDV_CACHE_ROOT", str(tmp_path / "c"))
    p = project_paths()
    assert p.data_root == tmp_path / "d"
    assert p.results_root == tmp_path / "r"
    assert p.cache_root == tmp_path / "c"


def test_subpath_helpers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CSDV_DATA_ROOT", str(tmp_path / "d"))
    monkeypatch.setenv("CSDV_RESULTS_ROOT", str(tmp_path / "r"))
    p = project_paths()
    assert p.naip_dir("SCBI", 2023) == tmp_path / "d" / "naip" / "SCBI" / "2023"
    assert p.naip_chm_dir("HARV", 2021) == tmp_path / "d" / "naip_chm" / "HARV" / "2021"
    assert (
        p.neon_chm_dir("SCBI", 2023)
        == tmp_path / "d" / "neon" / "SCBI" / "als_chm" / "2023"
    )
    assert (
        p.metrics_dir("SCBI", 2023, 50)
        == tmp_path / "r" / "metrics" / "SCBI" / "2023" / "50m"
    )
    assert p.trajectories_dir("SCBI") == tmp_path / "r" / "trajectories" / "SCBI"
    assert (
        p.trajectories_dir("SCBI", 100)
        == tmp_path / "r" / "trajectories" / "SCBI" / "100m"
    )
    assert p.figures_dir() == tmp_path / "r" / "figures"


def test_project_paths_is_frozen() -> None:
    p = project_paths()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.data_root = Path("/tmp")  # type: ignore[misc]
