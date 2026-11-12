"""The labelling steps over a label run folder, with local synthetic mosaics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from click.testing import CliRunner
from rasterio.transform import from_origin

from csdv_core.cli import main
from csdv_core.label import judge as J
from csdv_core.label.lattice import Window
from csdv_core.label.pipeline import (
    LabelRun,
    build_window,
    load_requests,
    render_window,
    window_parts,
)
from csdv_core.label.rules import Reason
from csdv_core.label.sample import SampleDesign, draw_windows

UNIT = "q4407159"
WINDOW = Window("w1", UNIT, 26919, 977, 14646, 2, 2)
ROW = pd.Series(
    {
        "window_id": "w1",
        "unit": UNIT,
        "epsg": 26919,
        "ix0": 977,
        "iy0": 14646,
        "nx": 2,
        "ny": 2,
    }
)


@pytest.fixture()
def run(
    label_export: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> LabelRun:
    """A label run with one window and local mosaics for its three years."""
    from csdv_core.label.requests import LOCATIONS

    monkeypatch.setitem(LOCATIONS, UNIT, "a synthetic forest")
    x0, _, _, y1 = WINDOW.stack_bounds(64)
    for year in (2015, 2018, 2021):
        folder = tmp_path / "data" / "naip" / UNIT / str(year)
        folder.mkdir(parents=True)
        rng = np.random.default_rng(year)
        data = rng.integers(40, 200, (4, 1400, 1400), dtype=np.uint8)
        with rasterio.open(
            folder / f"NAIP_{UNIT}_{year}0801_0p6m.tif",
            "w",
            driver="GTiff",
            height=1400,
            width=1400,
            count=4,
            dtype="uint8",
            crs="EPSG:26919",
            transform=from_origin(x0 - 60, y1 + 60, 0.6, 0.6),
        ) as dst:
            dst.write(data)
    root = tmp_path / "label"
    root.mkdir()
    pd.DataFrame([ROW]).to_csv(root / "windows.csv", index=False)
    return LabelRun(root=root, runs_dir=label_export / "runs" / "default")


def _answer(states: list[str], **more) -> dict:
    return {"states": states, "confidence": 3, "reason": "test", **more}


def _write_answers(run: LabelRun, fit: str = "about_right") -> None:
    requests = load_requests(run.root / "requests" / "w1")
    by_source = {q.cand_id.split("-")[1]: q for q in requests}
    loss = _answer(
        ["forest", "full_loss", "open"],
        change_type="clearcut",
        outline_fit=fit,
        affected_share="over_75",
    )
    none = _answer(["forest", "forest", "forest"])
    answers = {
        "di": (loss, loss),
        "chm": (none, none),
        "weak": (_answer(["forest", "partial_loss", "open"], change_type="wind"), none),
    }
    records = []
    for source, pair in answers.items():
        q = by_source[source]
        for model, answer in zip(run.models, pair, strict=True):
            records.append(
                {
                    "window_id": "w1",
                    "cand_id": q.cand_id,
                    "kind": "verdict",
                    "model": model,
                    "dates": q.dates,
                    "view": [*q.view.bounds, q.view.px],
                    "extra": q.extra,
                    "answer": answer,
                    "cost": 0.0,
                    "cached": False,
                    "error": "",
                }
            )
    if run.answers.exists():
        run.answers.unlink()
    J.append_records(run.answers, records)


def test_render_then_build(run: LabelRun, tmp_path: Path) -> None:
    n = render_window(run, ROW, source="local", data_root=tmp_path / "data")
    assert n == 3  # the detected object, the CHM-loss object, the weak object
    folder = run.root / "requests" / "w1"
    requests = load_requests(folder)
    assert {q.cand_id.split("-")[1] for q in requests} == {"di", "chm", "weak"}
    assert all(len(q.images) == 3 and q.images[0][:2] == b"\xff\xd8" for q in requests)
    assert "2015-08-01" in requests[0].prompt and "yellow outline" in requests[0].prompt
    meta = json.loads((run.root / "stacks" / "w1" / "meta.json").read_text())
    assert meta["years"]["2018"]["source"] == "local"

    _write_answers(run)
    records = J.read_records(run.answers)
    built = build_window(run, ROW, records)
    status = {
        c.source: c.routed.status for c in built.judged if c.routed.status != "unjudged"
    }
    assert status == {"di": "accepted", "chm": "rejected", "weak": "disagree"}
    reason = built.labels.reason
    assert (reason == Reason.ACCEPTED_FULL).any()
    assert (reason[1] == Reason.REJECTED_FLAGGED).any()
    assert (reason[0] == Reason.DISAGREE).any()
    assert not (reason == Reason.NO_DATA).any()
    # The labels do not need the imagery: the same rasters without it.
    blind = build_window(run, ROW, records, with_stack=False)
    assert (blind.labels.reason == reason).all()
    assert blind.labels.intervals == built.labels.intervals


def test_answers_to_an_older_prompt_are_not_used(run: LabelRun, tmp_path: Path) -> None:
    render_window(run, ROW, source="local", data_root=tmp_path / "data")
    _write_answers(run)
    records = J.read_records(run.answers)
    assert len(J.verdicts_of(records, "w1")) == 3
    assert J.verdicts_of(records, "w1", prompt="verdict_v2") == {}
    old = [{**r, "extra": {"prompt": "verdict_v2"}} for r in records]
    run.answers.unlink()
    J.append_records(run.answers, old)
    kept = J.read_records(run.answers)
    assert len(kept) == 6 and J.verdicts_of(kept, "w1") == {}
    built = build_window(run, ROW, kept)
    assert {c.routed.status for c in built.judged} == {"unjudged"}
    # Old and new answers to the same call live side by side.
    J.append_records(run.answers, records)
    assert len(J.read_records(run.answers)) == 12


def test_polygon_requests_follow_the_verdicts(run: LabelRun, tmp_path: Path) -> None:
    render_window(run, ROW, source="local", data_root=tmp_path / "data")
    _write_answers(run)
    assert render_window(run, ROW, kind="polygon", source="local") == 0
    _write_answers(run, fit="too_small")
    assert render_window(run, ROW, kind="polygon", source="local") == 1
    polygon = load_requests(run.root / "requests" / "w1", "polygons.jsonl")[0]
    assert len(polygon.images) == 2 and polygon.extra["pre"] == 0
    assert "2015-08-01" in polygon.prompt and "2018-08-01" in polygon.prompt


def test_a_unit_without_location_text_is_refused(
    run: LabelRun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from csdv_core.label.requests import LOCATIONS

    monkeypatch.delitem(LOCATIONS, UNIT)
    with pytest.raises(KeyError):
        render_window(run, ROW, source="local", data_root=tmp_path / "data")


def test_candidates_in_the_collar_are_kept_but_not_judged(run: LabelRun) -> None:
    parts = window_parts(run, ROW, with_stack=False)
    table = parts.candidates.set_index("cand_id")
    small = table.loc[f"{UNIT}-di-2021-2"]
    assert not small["judged"] and small["area_ha"] < 0.3
    assert table["judged"].sum() == 3


def test_sample(run: LabelRun) -> None:
    design = SampleDesign(n_di=5, n_other=5, n_quiet=2, seed=1)
    table = draw_windows(run, [UNIT], design)
    counts = table["stratum"].value_counts().to_dict()
    assert counts["di"] == 1 and counts["other"] == 2
    # Each of the four chips inside the small grid touches a candidate, so
    # none of them qualifies as quiet forest.
    assert "quiet" not in counts
    assert list(table["window_id"]) == [f"W{i + 1:04d}" for i in range(len(table))]
    seeded = table[table["stratum"] != "quiet"]
    assert (seeded["incl_prob"] == 1.0).all()  # every candidate was taken
    assert table.equals(draw_windows(run, [UNIT], design))
    fewer = draw_windows(
        run, [UNIT], SampleDesign(n_di=5, n_other=1, n_quiet=0, seed=1)
    )
    other = fewer[fewer["stratum"] == "other"]
    assert len(other) == 1 and other["incl_prob"].iloc[0] == pytest.approx(1.0)
    assert other["seed_cand"].iloc[0] in set(seeded["seed_cand"])


def test_reviewed_objects_get_their_own_windows(run: LabelRun) -> None:
    design = SampleDesign(n_di=0, n_other=0, n_quiet=0, seed=1)
    table = draw_windows(run, [UNIT], design, reviewed=[(UNIT, 1), (UNIT, 99)])
    assert list(table["stratum"]) == ["reviewed"]
    assert table["seed_cand"].iloc[0] == f"{UNIT}-di-2018-1"
    assert pd.isna(table["incl_prob"].iloc[0])
    # A reviewed object that is also drawn at random keeps one window.
    both = draw_windows(
        run,
        [UNIT],
        SampleDesign(n_di=5, n_other=0, n_quiet=0, seed=1),
        reviewed=[(UNIT, 1)],
    )
    assert list(both["stratum"]) == ["reviewed"]


def test_cli_steps(run: LabelRun, label_export: Path, tmp_path: Path) -> None:
    cli = CliRunner()
    where = ["--label-dir", str(run.root), "--export-dir", str(label_export)]
    result = cli.invoke(main, ["label", "candidates", *where])
    assert result.exit_code == 0, result.output
    assert (run.root / "candidates" / f"{UNIT}.gpkg").exists()
    assert "chm_n" in result.output

    result = cli.invoke(
        main,
        [
            "label",
            "sample",
            *where,
            "--dry-run",
            "--n-di",
            "5",
            "--n-other",
            "5",
            "--n-quiet",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "judged candidates" in result.output
    result = cli.invoke(main, ["label", "sample", *where])
    assert result.exit_code != 0 and "exists" in result.output

    data = ["--data-root", str(tmp_path / "data"), "--naip-source", "local"]
    result = cli.invoke(main, ["label", "render", *where, *data, "--check"])
    assert result.exit_code == 0 and "MISSING" not in result.output
    result = cli.invoke(main, ["label", "render", *where, *data, "--chips"])
    assert result.exit_code == 0, result.output
    assert "3 verdict requests" in result.output
    assert len(list((run.root / "chips").glob("*_naip.tif"))) == 4

    result = cli.invoke(main, ["label", "verdict", *where, "--dry-run", "--cap", "1"])
    assert result.exit_code == 0 and "6 new verdict calls" in result.output
    result = cli.invoke(main, ["label", "verdict", *where, "--dry-run"])
    assert result.exit_code != 0  # a cap is required

    _write_answers(run)
    result = cli.invoke(main, ["label", "build", *where, "--no-imagery"])
    assert result.exit_code == 0, result.output
    assert "1 events" in result.output
    assert len(list((run.root / "chips").glob("*_reason.tif"))) == 4
    import geopandas as gpd

    events = gpd.read_file(run.root / "labels" / "labels.gpkg", layer="events")
    assert (
        events.crs.to_epsg() == 5070 and events["severity"].iloc[0] == "stand_replacing"
    )
    assert events["last_pre_acq"].iloc[0] == "2015-08-01"
    reviewed = gpd.read_file(
        run.root / "labels" / "labels.gpkg", layer="reviewed_areas"
    )
    assert set(reviewed["finding"]) == {"complete"} and len(reviewed) == 2

    result = cli.invoke(main, ["label", "sheets", *where])
    assert result.exit_code == 0, result.output
    assert (run.root / "sheets" / "w1_overview.png").exists()

    requests = str(run.root / "requests")
    result = cli.invoke(main, ["label", "compare-renders", requests, requests])
    assert result.exit_code == 0 and "median 0.00" in result.output
