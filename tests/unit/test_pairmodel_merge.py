"""Tests for merging label runs and for the draw share of a named run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from csdv_core.pairmodel import data as D
from csdv_core.pairmodel.cli import pair
from csdv_core.pairmodel.merge import merge_chips
from tests.unit.test_pairmodel_data import write_chip


def _run(tmp_path: Path, name: str, chips: list[tuple[str, int | None]]) -> Path:
    """A label folder ``tmp_path/name`` whose chips/ holds the given chips."""
    folder = tmp_path / name / "chips"
    folder.mkdir(parents=True)
    rows = [
        write_chip(folder, chip_id, loss_interval=loss, seed=i)
        for i, (chip_id, loss) in enumerate(chips)
    ]
    for row in rows:
        row["window_id"] = "W0001"
    pd.DataFrame(rows).to_parquet(folder / "chips.parquet", index=False)
    return folder.parent


def test_merge_keeps_runs_apart_and_drops_duplicates(tmp_path: Path) -> None:
    first = _run(tmp_path, "scale", [("qa_c0", 1), ("qa_c1", None)])
    second = _run(tmp_path, "stable", [("qb_c0", None), ("qa_c1", None)])
    result = merge_chips(tmp_path / "merged", {"scale": first, "stable": second})
    assert result.chips == {"scale": 2, "stable": 1}
    assert result.dropped == {"scale": 0, "stable": 1}
    table = pd.read_parquet(result.chips_dir / "chips.parquet")
    assert sorted(table["chip_id"]) == ["qa_c0", "qa_c1", "qb_c0"]
    assert set(table["window_id"]) == {"scale:W0001", "stable:W0001"}
    # Every file of every kept chip is linked, and the loader reads the folder.
    assert result.files == 3 * 4
    assert (result.chips_dir / "qb_c0_naip.tif").exists()
    index = D.load_index(result.chips_dir)
    assert set(index["run"]) == {"scale", "stable"}
    with pytest.raises(FileExistsError):
        merge_chips(tmp_path / "merged", {"scale": first})


def test_merge_rejects_a_colon_in_a_run_name(tmp_path: Path) -> None:
    first = _run(tmp_path, "scale", [("qa_c0", 1)])
    with pytest.raises(ValueError, match="colon"):
        merge_chips(tmp_path / "merged", {"sc:ale": first})


def _index() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit": ["qa"] * 4 + ["qb"] * 4 + ["qc"] * 6,
            "loss_share": [0.2] * 4 + [0.0] * 4 + [0.0] * 6,
            "run": ["scale"] * 8 + ["stable"] * 6,
        }
    )


def test_run_draws_take_their_share_from_the_samples_without_loss() -> None:
    index = _index()
    weights = D.sample_weights(index, loss_draws=0.5, run_draws={"stable": 0.15})
    assert weights.sum() == pytest.approx(1.0)
    assert weights[:4].sum() == pytest.approx(0.5)
    assert weights[8:].sum() == pytest.approx(0.15)
    assert weights[4:8].sum() == pytest.approx(0.35)


def test_no_run_draws_leave_the_weights_unchanged() -> None:
    index = _index()
    plain = D.sample_weights(index.drop(columns="run"), loss_draws=0.5)
    assert np.array_equal(D.sample_weights(index, loss_draws=0.5), plain)
    assert np.array_equal(D.sample_weights(index, loss_draws=0.5, run_draws={}), plain)


def test_run_draws_reject_unknown_runs_and_too_large_shares() -> None:
    index = _index()
    with pytest.raises(ValueError, match="No sample"):
        D.sample_weights(index, run_draws={"other": 0.1})
    with pytest.raises(ValueError, match="nothing"):
        D.sample_weights(index, loss_draws=0.5, run_draws={"stable": 0.5})


def test_merge_chips_command(tmp_path: Path) -> None:
    first = _run(tmp_path, "scale", [("qa_c0", 1)])
    second = _run(tmp_path, "stable", [("qb_c0", None)])
    result = CliRunner().invoke(
        pair,
        [
            "merge-chips",
            "--out",
            str(tmp_path / "m"),
            "--run",
            f"scale={first}",
            "--run",
            f"stable={second}",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "stable: 1 chips" in result.output
    bad = CliRunner().invoke(
        pair, ["merge-chips", "--out", str(tmp_path / "n"), "--run", "nofolder"]
    )
    assert bad.exit_code != 0
