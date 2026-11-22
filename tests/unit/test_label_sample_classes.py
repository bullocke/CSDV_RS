"""The sample by class: windows spread over model and detection agreement."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from click.testing import CliRunner
from rasterio.transform import from_origin
from shapely.geometry import box

from csdv_core.cli import main
from csdv_core.label.pipeline import LabelRun, window_parts
from csdv_core.label.sample import SampleDesign
from csdv_core.label.sample_classes import (
    ClassDesign,
    allocate_stage,
    candidate_class,
    class_counts,
    draw_class_windows,
    image_season,
    offseason_intervals,
    seed_seasons,
)

UNIT = "q4407159"
#: Seven blocks of 6 by 6 map cells (0.33 ha) on forest that nothing flagged.
BLOCKS = [(14, 40), (14, 50), (14, 60), (30, 40), (30, 50), (30, 60), (42, 40)]


def _write_map(export: Path) -> None:
    """A model map: it flags the detected object of 2015-2018 and, in
    2018-2021, seven small blocks of forest. It is silent on the CHM-loss
    and the weak object."""
    from csdv_core.label.inputs import load_unit_inputs

    unit = load_unit_inputs(export / "runs" / "default" / UNIT)
    cell = 9.6
    x0, _, _, y1 = unit.grid.bounds
    data = np.zeros((2, 84, 84), dtype="uint8")
    data[0, 19:31, 19:31] = 90
    for row, col in BLOCKS:
        data[1, row : row + 6, col : col + 6] = 60
    folder = unit.layers_root / "model"
    folder.mkdir(exist_ok=True)
    with rasterio.open(
        folder / f"prob_forest_{UNIT}.tif",
        "w",
        driver="GTiff",
        height=84,
        width=84,
        count=2,
        dtype="uint8",
        crs="EPSG:26919",
        transform=from_origin(
            np.floor(x0 / cell) * cell, np.ceil(y1 / cell) * cell, cell, cell
        ),
        nodata=255,
    ) as dst:
        dst.write(data)


@pytest.fixture()
def run(label_export: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from csdv_core.label.requests import LOCATIONS

    monkeypatch.setitem(LOCATIONS, UNIT, "a synthetic forest")
    _write_map(label_export)
    root = tmp_path / "label"
    root.mkdir()
    run = LabelRun(root=root, runs_dir=label_export / "runs" / "default")
    # Windows of up to two chips a side, as in round 2. In a unit this small
    # several seeds then share one window, which is allowed.
    run.cfg.chips.max_window_chips = 2
    return run


def _design(**more) -> ClassDesign:
    base = {
        "draw_per_stratum": 5,
        "stage1_per_unit": 2,
        "priority": {UNIT: "2018-2021"},
        "edge_m": 60.0,
    }
    return ClassDesign(**{**base, **more})


def test_classes_follow_the_share_the_model_flags() -> None:
    assert candidate_class("model", 1.0, plain=False) == "model_only"
    assert candidate_class("di", 0.05, plain=False) == "cand_only"
    assert candidate_class("chm", 0.3, plain=False) == "partly"
    assert candidate_class("weak", 0.5, plain=False) == "agree"
    # Without a map, or where the model must have no say, the source counts.
    assert candidate_class("weak", float("nan"), plain=False) == "weak"
    assert candidate_class("di", 0.9, plain=True) == "di"
    assert candidate_class("model", 1.0, plain=True) is None


def test_strata_are_class_by_priority_interval(run: LabelRun) -> None:
    counts = class_counts(run, [UNIT], _design()).set_index("sub_stratum")
    assert counts["candidates"].to_dict() == {
        "model_only-priority": 7,
        "cand_only-priority": 1,  # the CHM-loss object the model does not see
        "cand_only-other": 1,  # the weak object
        "agree-other": 1,  # the detected object
    }


def test_the_draw_orders_each_stratum_and_stages_the_first_windows(
    run: LabelRun,
) -> None:
    table, judged = draw_class_windows(run, [UNIT], _design())
    seeded = table[table["stratum"] != "quiet"]
    model = seeded[seeded["sub_stratum"] == "model_only-priority"]
    assert len(model) == 5 and list(model["rank"]) == [1, 2, 3, 4, 5]
    assert (model["stratum_n"] == 7).all() and (model["stratum"] == "other").all()
    # Stage 1: one window of each of the first two strata, priority first.
    stage1 = seeded[seeded["stage"] == 1].set_index("sub_stratum")
    assert set(stage1.index) == {"model_only-priority", "cand_only-priority"}
    assert (stage1["rank"] == 1).all()
    assert stage1.loc["model_only-priority", "incl_prob"] == pytest.approx(1 / 7)
    assert stage1.loc["cand_only-priority", "incl_prob"] == pytest.approx(1.0)
    assert seeded.loc[seeded["stage"] == 0, "incl_prob"].isna().all()
    assert table["window_id"].is_unique
    # The same design gives the same sample, and a larger draw contains it.
    again, _ = draw_class_windows(run, [UNIT], _design())
    assert table.equals(again)
    larger, _ = draw_class_windows(run, [UNIT], _design(draw_per_stratum=7))
    assert set(model["seed_cand"]) < set(
        larger.loc[larger["sub_stratum"] == "model_only-priority", "seed_cand"]
    )
    # Every candidate is judged in one window only.
    assert judged["cand_id"].is_unique
    seeds = judged[judged["role"] == "seed"]
    assert set(seeds["cand_id"]) == set(seeded["seed_cand"])


def test_only_the_listed_candidates_of_a_window_are_judged(run: LabelRun) -> None:
    table, judged = draw_class_windows(run, [UNIT], _design(judged_per_window=1))
    assert (judged["role"] == "seed").all()
    table.to_csv(run.root / "windows.csv", index=False)
    judged.to_csv(run.root / "judged.csv", index=False)
    listed = LabelRun(root=run.root, runs_dir=run.runs_dir, cfg=run.cfg)
    for row in table[table["stratum"] != "quiet"].itertuples():
        parts = window_parts(listed, row, with_stack=False)
        asked = parts.candidates.loc[parts.candidates["judged"], "cand_id"]
        assert list(asked) == [row.seed_cand]
        # Model objects that are not judged are not in the window at all.
        others = parts.candidates[~parts.candidates["judged"]]
        assert "model" not in set(others["source"])


def test_a_seed_that_its_window_cannot_show_is_passed_over(run: LabelRun) -> None:
    # With windows of one chip, a block that straddles two chips shows under
    # 0.3 ha in either and cannot be judged. The draw takes the next one.
    run.cfg.chips.max_window_chips = 1
    table, judged = draw_class_windows(run, [UNIT], _design(draw_per_stratum=7))
    listed = judged.groupby("window_id")["cand_id"].apply(list).to_dict()
    for row in table[table["stratum"] != "quiet"].itertuples():
        parts = window_parts(run, row, with_stack=False)
        visible = parts.candidates.set_index("cand_id")["visible_ha"]
        assert visible[row.seed_cand] >= 0.3 and listed[row.window_id] == [
            row.seed_cand
        ] + [c for c in listed[row.window_id] if c != row.seed_cand]
    model = table[table["sub_stratum"] == "model_only-priority"]
    assert 0 < len(model) < 7 and list(model["rank"]) == list(range(1, len(model) + 1))


def test_ground_shared_by_two_windows_uses_the_verdicts_of_both(
    run: LabelRun,
) -> None:
    from csdv_core.label.pipeline import build_window
    from csdv_core.label.requests import VERDICT_PROMPT
    from csdv_core.label.rules import Reason

    table, judged = draw_class_windows(run, [UNIT], _design(judged_per_window=1))
    table.to_csv(run.root / "windows.csv", index=False)
    judged.to_csv(run.root / "judged.csv", index=False)
    listed = LabelRun(root=run.root, runs_dir=run.runs_dir, cfg=run.cfg)
    model = table[table["sub_stratum"] == "model_only-priority"]
    # Two windows of which the first also holds the seed of the second.
    cands = listed.candidates(UNIT).set_index("cand_id")
    first = second = None
    for _, a in model.iterrows():
        area = box(*listed.window(a).bounds)
        for _, b in model.iterrows():
            seen = cands.loc[b["seed_cand"], "geometry"].intersection(area).area
            if a["window_id"] != b["window_id"] and seen / 1e4 >= 0.3:
                first, second = a, b
                break
        if first is not None:
            break
    assert first is not None, "no two model windows share a seed"

    def verdict(window_id: str, cand_id: str, states: list[str]) -> list[dict]:
        answer = {"states": states, "confidence": 3, "reason": "test"}
        if "full_loss" in states:
            answer.update(
                change_type="clearcut",
                outline_fit="about_right",
                affected_share="over_75",
            )
        return [
            {
                "window_id": window_id,
                "cand_id": cand_id,
                "kind": "verdict",
                "model": name,
                "dates": ["2015-08-01", "2018-08-02", "2021-08-03"],
                "view": [0, 0, 1, 1, 640],
                "extra": {"prompt": VERDICT_PROMPT},
                "answer": answer,
                "cost": 0.0,
                "cached": False,
                "error": "",
            }
            for name in ("gem37", "terra")
        ]

    # The second window's seed was judged as a loss. The first window, whose
    # labels the shared chips take, shows it although it never asked.
    records = verdict(
        second["window_id"], second["seed_cand"], ["forest", "forest", "full_loss"]
    )
    built = build_window(listed, first, records, with_stack=False)
    status = {c.cand_id: c.routed.status for c in built.judged}
    assert status[second["seed_cand"]] == "accepted"
    assert (built.labels.reason[1] == Reason.ACCEPTED_FULL).any()
    # Its own seed has no verdict yet and is unlabelled in its interval.
    assert status[first["seed_cand"]] == "unjudged"
    # A model object nobody judged is not there at all.
    assert set(status) & set(model["seed_cand"]) == {
        first["seed_cand"],
        second["seed_cand"],
    }
    # Without the list, a window knows only what was asked through it.
    (run.root / "judged.csv").unlink()
    alone = LabelRun(root=run.root, runs_dir=run.runs_dir, cfg=run.cfg)
    built = build_window(alone, first, records, with_stack=False)
    status = {c.cand_id: c.routed.status for c in built.judged}
    assert status[second["seed_cand"]] == "unjudged"


def test_units_set_aside_for_testing_are_sampled_without_the_model(
    run: LabelRun,
) -> None:
    design = _design(plain_units=frozenset({UNIT}))
    counts = class_counts(run, [UNIT], design).set_index("sub_stratum")
    assert counts["candidates"].to_dict() == {
        "chm-priority": 1,
        "weak-other": 1,
        "di-other": 1,
    }
    table, judged = draw_class_windows(run, [UNIT], design)
    assert not judged["cand_id"].str.contains("-model-").any()
    assert set(table.loc[table["stratum"] != "quiet", "sub_stratum"]) == set(
        counts.index
    )


def test_a_seed_near_the_edge_of_its_unit_is_left_out(run: LabelRun) -> None:
    # The CHM-loss object starts 96 m from the west edge and is 96 m wide.
    wide = class_counts(run, [UNIT], _design(edge_m=160.0))
    assert "cand_only-priority" not in set(wide["sub_stratum"])
    near = class_counts(run, [UNIT], _design(edge_m=60.0))
    assert "cand_only-priority" in set(near["sub_stratum"])


def test_model_objects_do_not_stop_a_chip_from_being_quiet(run: LabelRun) -> None:
    quiet = SampleDesign(
        n_di=0,
        n_other=0,
        n_quiet=4,
        quiet_intervals={UNIT: ("2018-2021",)},
        quiet_raised_share=1.0,
        quiet_forest_share=0.5,
    )
    table, _ = draw_class_windows(run, [UNIT], _design(quiet=quiet))
    chips = table[table["stratum"] == "quiet"]
    # The chip north-east of the centre holds model blocks of 2018-2021 and
    # no other candidate of that interval.
    assert len(chips) >= 1 and (chips["stage"] == 1).all()
    assert (chips["sub_stratum"] == "2018-2021").all()


def test_a_stage_takes_the_next_windows_of_each_stratum(run: LabelRun) -> None:
    table, _ = draw_class_windows(run, [UNIT], _design())
    plan = pd.DataFrame(
        {"unit": [UNIT], "sub_stratum": ["model_only-priority"], "n": [2]}
    )
    out = allocate_stage(table, plan, 2)
    model = out[out["sub_stratum"] == "model_only-priority"].sort_values("rank")
    assert list(model["stage"]) == [1, 2, 2, 0, 0]
    assert model["incl_prob"].tolist()[:3] == pytest.approx([3 / 7] * 3)
    assert model["incl_prob"].iloc[3:].isna().all()
    # Other strata are untouched, and no window ID or order changes.
    assert list(out["window_id"]) == list(table["window_id"])
    rest = out["sub_stratum"] != "model_only-priority"
    assert out.loc[rest, "stage"].equals(table.loc[rest, "stage"])

    # Given again with another plan, the stage is redone, except for
    # windows that already hold answers.
    held = set(model["window_id"].iloc[1:2])
    plan["n"] = 1
    redone = allocate_stage(out, plan, 2, judged_windows=held)
    model = redone[redone["sub_stratum"] == "model_only-priority"].sort_values("rank")
    assert list(model["stage"]) == [1, 2, 2, 0, 0]
    plan["n"] = 0
    redone = allocate_stage(out, plan, 2, judged_windows=held)
    assert (redone["stage"] == 2).sum() == 1

    plan["n"] = 9
    with pytest.raises(ValueError, match="9 asked, 4 left"):
        allocate_stage(table, plan, 2)
    with pytest.raises(ValueError, match="Stage 1 is set"):
        allocate_stage(table, plan, 1)
    plan["sub_stratum"] = "agree-priority"
    with pytest.raises(ValueError, match="No stratum agree-priority"):
        allocate_stage(table, plan, 2)


def test_a_plan_can_take_the_windows_of_one_season(run: LabelRun) -> None:
    assert image_season("2018-10-31") == "may-oct"
    assert image_season("2018-11-01") == "nov-apr"
    assert image_season("2019-04-30") == "nov-apr"
    assert image_season("2019-05-01") == "may-oct"

    # One stratum whose seeds end in two seasons: the seven blocks the model
    # flags in 2018-2021 and, as a stand-in for a second interval, the same
    # candidates listed with a season of their own.
    table, _ = draw_class_windows(run, [UNIT], _design())
    sub = "model_only-priority"
    seeds = seed_seasons(run, [UNIT], _design())
    assert set(seeds["season"]) == {"may-oct"}
    pool = seeds[seeds["sub_stratum"] == sub]
    assert len(pool) == 7
    drawn = table[table["sub_stratum"] == sub].sort_values("rank")
    # Ranks 2 and 4 of the draw, and one candidate that was not drawn, end
    # on a winter image. The rest end in summer.
    winter = set(drawn["seed_cand"].iloc[[1, 3]])
    winter.add(next(c for c in pool["cand_id"] if c not in set(drawn["seed_cand"])))
    seeds.loc[seeds["cand_id"].isin(winter), "season"] = "nov-apr"

    plan = pd.DataFrame(
        {"unit": [UNIT], "sub_stratum": [sub], "season": ["nov-apr"], "n": [2]}
    )
    out = allocate_stage(table, plan, 2, seeds=seeds)
    model = out[out["sub_stratum"] == sub].sort_values("rank")
    # Both winter windows are taken out of turn. Rank 1 was stage 1.
    assert list(model["stage"]) == [1, 2, 0, 2, 0]
    assert list(model["season"]) == ["may-oct", "nov-apr"] * 2 + ["may-oct"]
    assert list(model["season_n"]) == [4, 3, 4, 3, 4]
    # Inclusion is counted per season: 2 of 3 in winter, 1 of 4 in summer.
    assert model["incl_prob"].tolist()[:2] == pytest.approx([1 / 4, 2 / 3])
    assert model["incl_prob"].iloc[3] == pytest.approx(2 / 3)
    assert model["incl_prob"].iloc[[2, 4]].isna().all()
    # A stratum the plan does not split keeps its count over the stratum.
    other = out[out["sub_stratum"] == "cand_only-priority"]
    assert other["season_n"].isna().all() and (other["incl_prob"] == 1.0).all()
    assert list(out["window_id"]) == list(table["window_id"])

    # A later row without a season takes the next windows of the stratum,
    # and the stratum stays split.
    plan = pd.DataFrame(
        {
            "unit": [UNIT, UNIT],
            "sub_stratum": [sub, sub],
            "season": ["nov-apr", "may-oct"],
            "n": [1, 2],
        }
    )
    out = allocate_stage(table, plan, 2, seeds=seeds)
    model = out[out["sub_stratum"] == sub].sort_values("rank")
    assert list(model["stage"]) == [1, 2, 2, 0, 2]
    assert model["incl_prob"].tolist()[:3] == pytest.approx([3 / 4, 1 / 3, 3 / 4])
    again = allocate_stage(
        out, pd.DataFrame({"unit": [UNIT], "sub_stratum": [sub], "n": [1]}), 3
    )
    model = again[again["sub_stratum"] == sub].sort_values("rank")
    assert list(model["stage"]) == [1, 2, 2, 3, 2]
    assert model["incl_prob"].iloc[3] == pytest.approx(2 / 3)

    # A row with a season takes what is left when it asks for more: the
    # plan is written without the season of every seed.
    plan = pd.DataFrame(
        {"unit": [UNIT], "sub_stratum": [sub], "season": ["nov-apr"], "n": [3]}
    )
    out = allocate_stage(table, plan, 2, seeds=seeds)
    model = out[out["sub_stratum"] == sub].sort_values("rank")
    assert list(model["stage"]) == [1, 2, 0, 2, 0]
    assert out.attrs["clipped"] == [(UNIT, f"{sub} nov-apr", 3, 2)]
    with pytest.raises(ValueError, match="the seeds are needed"):
        allocate_stage(table, plan, 2)
    plan["season"] = "winter"
    with pytest.raises(ValueError, match="names the seasons"):
        allocate_stage(table, plan, 2, seeds=seeds)
    # Seeds listed with another design do not match the draw.
    plan["season"] = "nov-apr"
    with pytest.raises(ValueError, match="6 candidates listed, 7 when"):
        allocate_stage(table, plan, 2, seeds=seeds.drop(pool.index[-1]))


def test_the_season_of_a_seed_is_that_of_its_quad(
    run: LabelRun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run dates a year by the median over its quads. The mosaic's sidecar
    says which quad, flown when, covers each candidate."""
    from types import SimpleNamespace

    from csdv_core.label import sample_classes

    plain = seed_seasons(run, [UNIT], _design())
    assert set(plain["date"]) == {
        run.unit(UNIT).dates[2018],
        run.unit(UNIT).dates[2021],
    }
    assert plain.attrs["run_dates"] == 2  # no data root: both years

    def mosaic(year: int, items: list[str]) -> None:
        folder = tmp_path / "naip" / UNIT / str(year)
        folder.mkdir(parents=True)
        path = folder / f"NAIP_{UNIT}_{year}0803_0p6m.tif"
        path.write_bytes(b"")
        sidecar = {"item_ids": items}
        path.with_suffix(".provenance.json").write_text(json.dumps(sidecar))

    # 2021: every quad was flown on one winter day, whatever the run says.
    mosaic(2021, ["nh_m_4407159_sw_19_060_20220112", "nh_m_4407159_se_19_060_20220112"])
    seeds = seed_seasons(run, [UNIT], _design(), tmp_path)
    late = seeds[seeds["sub_stratum"].str.endswith("-priority")]
    assert set(late["date"]) == {"2022-01-12"} and set(late["season"]) == {"nov-apr"}
    # 2018 has no mosaic here: its seeds keep the run's date.
    early = seeds[~seeds["sub_stratum"].str.endswith("-priority")]
    assert set(early["date"]) == {run.unit(UNIT).dates[2018]}
    assert seeds.attrs["run_dates"] == 1

    # 2018: two flights. The quad that covers the candidate decides, and a
    # candidate no listed quad reaches keeps the run's date.
    mosaic(2018, ["nh_m_4407159_sw_19_060_20181026", "nh_m_4407159_se_19_060_20190102"])
    seen = []

    def quad_of(path, epsg, bounds):
        seen.append((path.name, epsg, bounds))
        return SimpleNamespace(date="2019-01-02") if len(seen) == 1 else None

    monkeypatch.setattr(sample_classes, "provenance_date", quad_of)
    seeds = seed_seasons(run, [UNIT], _design(), tmp_path)
    early = seeds[~seeds["sub_stratum"].str.endswith("-priority")]
    assert list(early["date"]) == ["2019-01-02", run.unit(UNIT).dates[2018]]
    assert list(early["season"]) == ["nov-apr", "may-oct"]
    assert seeds.attrs["run_dates"] == 0
    cands = run.candidates(UNIT).set_index("cand_id")
    first = cands.loc[early["cand_id"].iloc[0]]
    assert seen[0] == (f"NAIP_{UNIT}_20180803_0p6m.tif", 26919, first.geometry.bounds)


def test_quiet_chips_are_judged_on_off_season_intervals(run: LabelRun) -> None:
    unit = run.unit(UNIT)
    assert offseason_intervals(run, [UNIT]) == {UNIT: tuple(unit.intervals)}
    unit.dates[2018] = "2018-11-20"
    assert offseason_intervals(run, [UNIT]) == {UNIT: ("2015-2018", "2018-2021")}
    unit.dates[2018] = "2018-08-02"
    unit.dates[2021] = "2021-04-03"
    assert offseason_intervals(run, [UNIT]) == {UNIT: ("2018-2021",)}


def test_cli_draws_by_class_and_allocates_a_stage(
    run: LabelRun, label_export: Path, tmp_path: Path
) -> None:
    sites = tmp_path / "sites.csv"
    sites.write_text(
        "# the places of the run\n"
        "unit,role,interval,location\n"
        f'{UNIT},train,2018-2021,"somewhere, a region"\n'
        "p_missing,train,2016-2018,elsewhere\n"
    )
    config = tmp_path / "config.yaml"
    config.write_text("chips:\n  max_window_chips: 2\n")
    where = ["--label-dir", str(run.root), "--export-dir", str(label_export)]
    where += ["--config", str(config)]
    sample = ["label", "sample", *where, "--design", "classes", "--sites", str(sites)]
    sample += ["--draw-per-stratum", "5", "--stage1-per-unit", "2", "--edge-m", "60"]
    sample += ["--n-quiet", "0"]
    cli = CliRunner()
    result = cli.invoke(main, [*sample, "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "1 units have no run here and are left out: p_missing" in result.output
    assert "Stage 1: 2 seeded windows" in result.output
    assert not (run.root / "windows.csv").exists()
    result = cli.invoke(main, sample)
    assert result.exit_code == 0, result.output
    table = pd.read_csv(run.root / "windows.csv")
    assert {"rank", "stratum_n", "stage"} <= set(table.columns)
    assert (run.root / "judged.csv").exists()
    assert (run.root / "units.csv").read_text().split() == ["unit", UNIT]

    plan = tmp_path / "plan.csv"
    plan.write_text(f"unit,sub_stratum,n\n{UNIT},model_only-priority,3\n")
    allocate = ["label", "allocate", *where, "--plan", str(plan)]
    result = cli.invoke(main, [*allocate, "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Stage 2: 3 windows in 1 units" in result.output
    assert (pd.read_csv(run.root / "windows.csv")["stage"] == 2).sum() == 0
    result = cli.invoke(main, allocate)
    assert result.exit_code == 0, result.output
    assert (pd.read_csv(run.root / "windows.csv")["stage"] == 2).sum() == 3
    assert len(list(run.root.glob("windows.2*.csv"))) == 1  # the copy kept

    # A plan with seasons needs the sites file, and counts per season.
    plan.write_text(
        f"unit,sub_stratum,season,n\n{UNIT},model_only-priority,may-oct,2\n"
    )
    result = cli.invoke(main, [*allocate, "--dry-run"])
    assert result.exit_code != 0 and "give --sites" in result.output
    seasons = [*allocate, "--sites", str(sites), "--edge-m", "60"]
    result = cli.invoke(main, seasons)
    assert result.exit_code == 0, result.output
    table = pd.read_csv(run.root / "windows.csv")
    model = table[table["sub_stratum"] == "model_only-priority"]
    assert (model["season"] == "may-oct").all() and (model["season_n"] == 7).all()
    assert (table["stage"] == 2).sum() == 2
    # With another edge the strata are not those of the draw.
    result = cli.invoke(main, [*seasons, "--edge-m", "400"])
    assert result.exit_code != 0 and "when the sample was drawn" in result.output

    # A priority interval the unit does not have is an error, not a silent miss.
    sites.write_text(f"unit,role,interval\n{UNIT},train,2016-2019\n")
    result = cli.invoke(main, [*sample, "--dry-run"])
    assert result.exit_code != 0 and "2016-2019 is not among" in result.output
