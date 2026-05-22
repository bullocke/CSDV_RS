"""Tests for ``csdv check`` preflight subcommand."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner


@pytest.fixture()
def isolated_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data = tmp_path / "data"
    results = tmp_path / "results"
    cache = tmp_path / "cache"
    data.mkdir()
    results.mkdir()
    cache.mkdir()
    monkeypatch.setenv("CSDV_DATA_ROOT", str(data))
    monkeypatch.setenv("CSDV_RESULTS_ROOT", str(results))
    monkeypatch.setenv("CSDV_CACHE_ROOT", str(cache))
    return tmp_path


def _run_check(args: list[str]) -> object:
    from csdv_core.check import cli

    runner = CliRunner()
    return runner.invoke(cli, args, standalone_mode=False)


def test_check_no_args_passes(isolated_roots):
    result = _run_check([])
    assert result.exit_code == 0, result.output
    assert "[FAIL]" not in result.output
    assert "# environment" in result.output
    assert "# paths" in result.output
    assert "# configs" in result.output


def test_check_unknown_site_fails(isolated_roots):
    result = _run_check(["--site", "ZZZZ"])
    assert result.exit_code != 0
    assert "[FAIL]" in result.output
    assert "ZZZZ" in result.output


def test_check_known_site_passes(isolated_roots):
    result = _run_check(["--site", "SCBI"])
    assert result.exit_code == 0, result.output
    assert "[FAIL]" not in result.output


def test_check_unknown_year_warns(isolated_roots):
    result = _run_check(["--site", "SCBI", "--years", "1999"])
    assert "[WARN]" in result.output
    # Non-strict: WARN does not fail.
    assert result.exit_code == 0


def test_check_unknown_year_strict_fails(isolated_roots):
    result = _run_check(["--site", "SCBI", "--years", "1999", "--strict"])
    assert result.exit_code != 0
    assert "[WARN]" in result.output


def test_check_unknown_window_warns(isolated_roots):
    result = _run_check(["--window-m", "37"])
    assert "[WARN]" in result.output
    assert result.exit_code == 0


def test_check_missing_data_root_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("CSDV_DATA_ROOT", str(tmp_path / "does_not_exist"))
    monkeypatch.setenv("CSDV_RESULTS_ROOT", str(tmp_path / "results"))
    (tmp_path / "results").mkdir()
    result = _run_check([])
    assert result.exit_code != 0
    assert "MISSING" in result.output


def test_check_pipeline_state_reports_missing(isolated_roots):
    result = _run_check(["--site", "SCBI", "--years", "2014,2018", "--window-m", "50"])
    # All pipeline outputs are missing in a fresh data root: WARN rows.
    assert "# pipeline-state" in result.output
    assert "[WARN]" in result.output
    assert result.exit_code == 0
