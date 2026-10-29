"""Tests for submitting a unit's Slurm chain and reading its progress.

No network and no Slurm: the catalogue lookup, ``sbatch`` and ``sacct`` are
replaced, so what is checked is which steps are submitted, their
dependencies, the CHM manifest, the ledger and the status rows.
"""

from __future__ import annotations

import pytest

from csdv_core.grid import status as status_mod
from csdv_core.grid import submit as submit_mod
from csdv_core.grid.status import collect_status, read_ledger
from csdv_core.grid.submit import SubmitError, SubmitOptions, submit_unit
from csdv_core.grid.units import GridPaths, Unit, load_unit

UNIT = "q4407159"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    paths = GridPaths(
        grid_dir=tmp_path / "grid",
        export_dir=tmp_path / "exports",
        data_root=tmp_path / "data",
        results_root=tmp_path / "results",
    )
    for name in ("CSDV_REPO", "CSDV_LOG_DIR", "CSDV_MANIFEST_DIR"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    monkeypatch.setattr(
        submit_mod,
        "discover_years",
        lambda bbox: ({2018: 1.0, 2021: 1.0, 2023: 0.4}, {2025: 1.0}),
    )
    calls: list[list[str]] = []

    def fake_sbatch(command):
        calls.append(command)
        return str(1000 + len(calls))

    monkeypatch.setattr(submit_mod, "_sbatch", fake_sbatch)
    monkeypatch.setattr(status_mod, "sacct_states", lambda ids: {})
    monkeypatch.setattr(submit_mod, "sacct_states", lambda ids: {})
    return paths, calls


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_full_chain_with_dependencies_manifest_and_ledger(env):
    paths, calls = env
    opts = SubmitOptions(detect_args=("--rule", "D6s"))
    sub = submit_unit(paths, Unit.cell(UNIT, label="Lincoln"), opts)

    assert [j.step for j in sub.jobs] == [
        "download",
        "chm",
        "detect",
        "chips",
        "export",
    ]
    download, chm, detect, chips, export = (j.command for j in sub.jobs)
    assert "--years" in download and "2018,2021,2025" in download
    assert "--dependency=afterok:1001" in chm and "--array=1-3" in chm
    assert "--dependency=afterok:1002" in detect
    assert detect[-4:] == ["--years", "2018,2021,2025", "--rule", "D6s"]
    assert "--dependency=afterok:1003" in chips
    assert "--dependency=afterok:1003,afterany:1004" in export

    rows = sub.manifest.read_text().splitlines()
    assert rows[0] == "site,year,naip_path,crowns_name"
    assert rows[1] == f"{UNIT},2018,{paths.naip_year_dir(UNIT, 2018)},"

    ledger = read_ledger(paths)
    assert ledger[-1]["detect_job"] == "1003"
    assert ledger[-1]["detect_args"] == "--rule D6s"
    unit = load_unit(paths, UNIT)
    assert unit.years[2025]["source"] == "rest"
    assert "40%" in unit.skipped_years[2023]


def test_second_run_tag_submits_only_detection_onward(env):
    paths, calls = env
    submit_unit(paths, Unit.cell(UNIT), SubmitOptions())
    for year in (2018, 2021, 2025):
        _touch(paths.naip_year_dir(UNIT, year) / "n.tif")
        _touch(paths.chm_year_dir(UNIT, year) / "n_chm.tif")
    _touch(paths.run_dir(UNIT, "default") / "summary.json")

    with pytest.raises(SubmitError, match="exists"):
        submit_unit(paths, Unit.cell(UNIT), SubmitOptions())
    sub = submit_unit(
        paths, Unit.cell(UNIT), SubmitOptions(run_tag="D6s", steps=("detect", "export"))
    )
    assert [j.step for j in sub.jobs] == ["detect", "export"]
    assert not any(a.startswith("--dependency") for a in sub.jobs[0].command)
    assert "--dependency=afterok:" in " ".join(sub.jobs[1].command)


def test_dry_run_writes_nothing(env):
    paths, calls = env
    sub = submit_unit(paths, Unit.cell(UNIT), SubmitOptions(dry_run=True))
    assert calls == []
    assert len(sub.jobs) == 5
    assert "--dependency=afterok:<download>" in sub.jobs[1].command
    assert not paths.ledger.exists()
    assert load_unit(paths, UNIT) is None


def test_too_few_years_is_refused(env, monkeypatch):
    paths, _ = env
    monkeypatch.setattr(submit_mod, "discover_years", lambda bbox: ({2021: 1.0}, {}))
    with pytest.raises(SubmitError, match="needs 2"):
        submit_unit(paths, Unit.cell(UNIT), SubmitOptions())


def test_status_follows_the_files(env, monkeypatch):
    paths, _ = env
    submit_unit(paths, Unit.cell(UNIT, label="Lincoln"), SubmitOptions())
    monkeypatch.setattr(
        status_mod,
        "sacct_states",
        lambda ids: {"1002": ["COMPLETED", "RUNNING", "PENDING"]},
    )
    for year in (2018, 2021, 2025):
        _touch(paths.naip_year_dir(UNIT, year) / "n.tif")
    _touch(paths.chm_year_dir(UNIT, 2018) / "n_chm.tif")

    (row,) = collect_status(paths)
    assert (row.unit, row.label, row.run_tag) == (UNIT, "Lincoln", "default")
    assert (row.naip, row.chm, row.detect) == ("3/3", "1/3", "-")
    assert row.state.startswith("chm: ") and "RUNNING" in row.state

    for year in (2021, 2025):
        _touch(paths.chm_year_dir(UNIT, year) / "n_chm.tif")
    run = paths.run_dir(UNIT, "default")
    _touch(run / "summary.json")
    _touch(run / "inspection" / "index.md")
    _touch(paths.export_run_dir(UNIT, "default") / "export.json")
    (row,) = collect_status(paths)
    assert row.state == "done"


def test_sacct_array_specs_are_counted(monkeypatch):
    monkeypatch.setattr(status_mod.shutil, "which", lambda name: "/usr/bin/sacct")

    class Out:
        stdout = "77_1|COMPLETED\n77_[2-4%2]|PENDING\n78|CANCELLED by 5\n"

    monkeypatch.setattr(status_mod.subprocess, "run", lambda *a, **k: Out())
    states = status_mod.sacct_states(["77", "78"])
    assert states == {
        "77": ["COMPLETED", "PENDING", "PENDING", "PENDING"],
        "78": ["CANCELLED"],
    }
