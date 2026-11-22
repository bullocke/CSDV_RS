"""Tests for merging label runs and for the draw share of a named run."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import click
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


def test_a_share_of_zero_draws_as_if_the_run_were_absent() -> None:
    """The control of a run with a share: same chips, none of the run drawn."""
    index = _index()
    weights = D.sample_weights(index, loss_draws=0.5, run_draws={"stable": 0.0})
    assert (weights[8:] == 0).all()
    alone = D.sample_weights(index.iloc[:8].drop(columns="run"), loss_draws=0.5)
    assert weights[:8] == pytest.approx(alone)


def test_run_draws_option_accepts_zero_and_rejects_one() -> None:
    from csdv_core.pairmodel.cli import _run_draws

    assert _run_draws(("stable=0",)) == {"stable": 0.0}
    assert _run_draws(("stable=0.15",)) == {"stable": 0.15}
    for bad in ("stable=1", "stable=-0.1", "stable", "=0.1"):
        with pytest.raises(click.BadParameter):
            _run_draws((bad,))


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


def test_an_excluded_run_is_never_drawn() -> None:
    """The control of a model trained with a run: the others as without it."""
    index = _index()
    weights = D.sample_weights(index, loss_draws=0.5, exclude_runs=["stable"])
    assert (weights[8:] == 0).all() and weights.sum() == pytest.approx(1.0)
    alone = D.sample_weights(index.iloc[:8].drop(columns="run"), loss_draws=0.5)
    assert weights[:8] == pytest.approx(alone)
    # A run with loss leaves with all of its samples, not only the quiet ones.
    weights = D.sample_weights(index, loss_draws=0.5, exclude_runs=["scale"])
    assert (weights[:8] == 0).all() and weights[8:].sum() == pytest.approx(1.0)
    # A share for an excluded run is dropped with it.
    both = D.sample_weights(index, run_draws={"stable": 0.15}, exclude_runs=["stable"])
    assert (both[8:] == 0).all()
    with pytest.raises(ValueError, match="No sample from the label runs"):
        D.sample_weights(index, exclude_runs=["other"])
    with pytest.raises(ValueError, match="leave no sample"):
        D.sample_weights(index, exclude_runs=["scale", "stable"])


def _sites(folder: Path, rows: list[str]) -> None:
    text = "unit,role,holdout_with,target\n" + "\n".join(rows) + "\n"
    (folder / "sites.csv").write_text(text)


def test_merge_carries_the_sites_of_its_runs(tmp_path: Path) -> None:
    first = _run(tmp_path, "scale", [("qa_c0", 1), ("qb_c0", None)])
    second = _run(tmp_path, "places", [("pa_c0", 1), ("pb_c0", 1), ("pc_c0", None)])
    _sites(second, ["pa,train,qa,wind", "pb,test,,fire", "pc,train,,snow"])
    result = merge_chips(tmp_path / "merged", {"scale": first, "places": second})
    assert result.sites == 3
    sites = D.load_sites(result.chips_dir)
    assert list(sites["unit"]) == ["pa", "pb", "pc"] and set(sites["run"]) == {"places"}
    assert D.role_units(sites, ["test"]) == ["pb"]
    assert D.tied_units(sites, ["qa"]) == ["pa"]
    assert D.tied_units(sites, ["qb"]) == [] and D.tied_units(None, ["qa"]) == []
    assert D.load_sites(first / "chips") is None


def test_training_holds_out_tied_units_and_roles(tmp_path: Path) -> None:
    from csdv_core.pairmodel import train as T

    first = _run(tmp_path, "scale", [("qa_c0", 1), ("qb_c0", 1), ("qc_c0", None)])
    second = _run(tmp_path, "places", [("pa_c0", 1), ("pb_c0", 1), ("pc_c0", None)])
    _sites(second, ["pa,train,qa,wind", "pb,test,,fire", "pc,train,,snow", "pz,test,,"])
    chips = merge_chips(tmp_path / "merged", {"scale": first, "places": second})
    cfg = T.TrainConfig(
        chips_dir=chips.chips_dir,
        out_dir=tmp_path / "run",
        holdout_units=("qa",),
        val_units=("qb",),
    )
    planned = T.plan(cfg)
    # The place that saw the event of the held-out unit is held out with it.
    assert planned.split.test == ("pa", "qa") and planned.held_with == ("pa",)
    assert "pa" not in set(planned.train_index["unit"])
    assert "ties them to a held-out unit: pa" in T.describe(planned)
    # With the test places held out as well. A place without chips is no unit.
    with_role = T.plan(T.dataclasses.replace(cfg, hold_roles=("test",)))
    assert with_role.split.test == ("pa", "pb", "qa")
    assert set(with_role.train_index["unit"]) == {"qc", "pc"}
    # The draws of a control that leaves a run out.
    control = T.dataclasses.replace(cfg, exclude_runs=("places",))
    text = T.describe_draws(T.plan(control), control)
    table = pd.read_csv(
        io.StringIO(text.split("\n", 1)[1]), sep=r"\s{2,}", engine="python"
    )
    assert table.loc["places", ["draws with loss", "draws without"]].sum() == 0
    assert table.loc["scale", ["draws with loss", "draws without"]].sum() == 1
    # A folder without a sites table cannot hold out a role.
    alone = T.TrainConfig(
        chips_dir=first / "chips", out_dir=tmp_path / "r", hold_roles=("test",)
    )
    with pytest.raises(ValueError, match="roles cannot be held out"):
        T.plan(alone)


def test_score_writes_samples_units_and_pooled_scores(tmp_path: Path) -> None:
    import json

    import torch

    from csdv_core.pairmodel.model import SiamesePairNet
    from csdv_core.pairmodel.score import counts_of, score

    first = _run(tmp_path, "scale", [("qa_c0", 1)])
    second = _run(tmp_path, "places", [("pa_c0", 1), ("pb_c0", 0), ("pc_c0", None)])
    _sites(second, ["pa,train,,wind", "pb,test,,fire", "pc,test,,snow", "pz,test,,"])
    chips = merge_chips(tmp_path / "merged", {"scale": first, "places": second})
    torch.manual_seed(0)
    model = SiamesePairNet(encoder="resnet18", cell_px=16, n_agents=0)
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "encoder": "resnet18",
            "cell_px": 16,
            "n_agents": 0,
            "n_dates": 2,
            "mean": [60.0] * 4,
            "std": [30.0] * 4,
            "best_threshold": 0.4,
        },
        checkpoint,
    )
    out = tmp_path / "score"
    result = score(
        checkpoint, chips.chips_dir, out, ["pb", "pc", "pz"], workers=0, device="cpu"
    )
    assert result.units == ("pb", "pc") and result.threshold == 0.4
    samples = pd.read_parquet(out / "samples.parquet")
    # Two chips of three dates: two intervals each.
    assert len(samples) == 4 and set(samples["run"]) == {"places"}
    cells = (128 // 16) ** 2
    loss = samples[(samples["unit"] == "pb") & (samples["k"] == 0)].iloc[0]
    assert sum(loss["pos"]) > 0 and sum(loss["pos"]) + sum(loss["neg"]) <= cells
    quiet = samples[samples["unit"] == "pc"]
    assert all(sum(p) == 0 for p in quiet["pos"])
    assert all(sum(n) == cells for n in quiet["neg"])
    units = pd.read_csv(out / "units.csv").set_index("unit")
    assert units.loc["pc", "loss_cells"] == 0 and units.loc["pb", "loss_cells"] > 0
    record = json.loads((out / "score.json").read_text())
    assert record["units"] == 2 and record["samples"] == 4
    assert record["pooled"]["loss_cells"] == counts_of(samples).n_pos
    assert record["ends_november_to_april"]["background_cells"] == 0

    # The command takes the units from the roles of the sites table.
    cli = CliRunner()
    args = ["score", "--checkpoint", str(checkpoint), "--chips-dir"]
    args += [str(chips.chips_dir), "--out-dir", str(tmp_path / "again")]
    done = cli.invoke(
        pair, [*args, "--role", "test", "--workers", "0", "--device", "cpu"]
    )
    assert done.exit_code == 0, done.output
    assert "4 samples of 2 units" in done.output
    none = cli.invoke(pair, args)
    assert none.exit_code != 0 and "--units or --role" in none.output
