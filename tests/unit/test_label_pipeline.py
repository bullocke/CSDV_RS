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
    stack_complete,
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


def _blank(data_root: Path, year: int, keep: tuple[float, float, float, float]) -> None:
    """Leave imagery of one year only inside ``keep`` (x0, y0, x1, y1)."""
    path = next((data_root / "naip" / UNIT / str(year)).glob("*.tif"))
    with rasterio.open(path, "r+") as dst:
        data = dst.read()
        t = dst.transform
        xs = t.c + (np.arange(dst.width) + 0.5) * t.a
        ys = t.f + (np.arange(dst.height) + 0.5) * t.e
        inside = ((ys > keep[1]) & (ys < keep[3]))[:, None] & (
            (xs > keep[0]) & (xs < keep[2])
        )
        data[:, ~inside] = 0
        dst.write(data)


def _grid_bounds(grid) -> tuple[float, float, float, float]:
    t = grid.cell_transform
    rows, cols = grid.cell_shape
    return (t.c, t.f + rows * t.e, t.c + cols * t.a, t.f)


def test_a_window_past_the_unit_edge_is_kept(
    run: LabelRun, label_grid, tmp_path: Path
) -> None:
    """The mosaics of a unit end at its edge. A window there is still used."""
    data_root = tmp_path / "data"
    for year in (2015, 2018, 2021):
        _blank(data_root, year, _grid_bounds(label_grid))
    assert render_window(run, ROW, source="local", data_root=data_root) == 3
    meta = json.loads((run.root / "stacks" / "w1" / "meta.json").read_text())
    assert meta["failed"] == {}
    assert 0.9 < meta["inside_unit"] < 0.98
    assert all(0.9 < info["coverage"] < 0.98 for info in meta["years"].values())
    assert stack_complete(run, ROW)


def test_a_gap_inside_the_unit_skips_the_window(
    run: LabelRun, label_export: Path, label_grid, tmp_path: Path
) -> None:
    """A year without imagery inside the unit drops the window, not the run."""
    data_root = tmp_path / "data"
    x0, y0, x1, y1 = _grid_bounds(label_grid)
    full = next((data_root / "naip" / UNIT / "2018").glob("*.tif"))
    saved = full.read_bytes()
    _blank(data_root, 2018, (x0, y0 + 150.0, x1, y1))

    cli = CliRunner()
    args = ["label", "render", "--label-dir", str(run.root)]
    args += ["--export-dir", str(label_export), "--data-root", str(data_root)]
    args += ["--naip-source", "local", "--chips"]
    result = cli.invoke(main, args)
    assert result.exit_code == 0, result.output
    assert "0 verdict requests in 0 of 1 windows" in result.output
    assert "Skipped, imagery incomplete (1): w1" in result.output
    assert not list((run.root / "chips").glob("*_naip.tif"))
    meta = json.loads((run.root / "stacks" / "w1" / "meta.json").read_text())
    assert list(meta["failed"]) == ["2018"] and not stack_complete(run, ROW)

    # A failed year is read again once its mosaic is whole.
    full.write_bytes(saved)
    result = cli.invoke(main, args)
    assert result.exit_code == 0, result.output
    assert "3 verdict requests in 1 of 1 windows" in result.output
    assert len(list((run.root / "chips").glob("*_naip.tif"))) == 4


def test_compare_renders_skips_ground_without_imagery(tmp_path: Path) -> None:
    from PIL import Image

    rng = np.random.default_rng(0)
    image = rng.integers(60, 200, (64, 64, 3), dtype=np.uint8)
    cut = image.copy()
    cut[:, 40:] = 0
    for name, array in (("a", image), ("b", cut)):
        folder = tmp_path / name / "w1"
        folder.mkdir(parents=True)
        Image.fromarray(array).save(folder / "c1_v0.jpg", quality=95)
    result = CliRunner().invoke(
        main, ["label", "compare-renders", str(tmp_path / "a"), str(tmp_path / "b")]
    )
    assert result.exit_code == 0, result.output
    assert "1 images lack imagery" in result.output and "w1" in result.output
    median = float(result.output.split("median ")[1].split(",")[0])
    assert median < 12  # only the JPEG noise of the part both renders hold


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


def test_ground_beyond_the_unit_is_empty_for_every_source(
    run: LabelRun, label_grid, tmp_path: Path
) -> None:
    """The mosaics here reach past the unit. The stack must not show that ground."""
    from csdv_core.label.stacks import inside_unit, load_stack

    render_window(run, ROW, source="local", data_root=tmp_path / "data")
    folder = run.root / "stacks" / "w1"
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["clipped_to_unit"] and 0.9 < meta["inside_unit"] < 0.98
    stack = load_stack(folder)
    inside = inside_unit(run.unit(UNIT), stack.bounds)
    assert not stack.data[:, :, ~inside].any()
    assert stack.data.any(axis=1)[:, inside].all()
    for info in meta["years"].values():
        assert info["coverage"] == pytest.approx(meta["inside_unit"], abs=1e-3)
        assert info["date_rule"] == "mosaic_name"  # these mosaics have no sidecar


def test_cell_ndvi_follows_the_unit_cells(run: LabelRun, tmp_path: Path) -> None:
    from csdv_core.label.lattice import cells_to_pixels
    from csdv_core.label.stacks import load_cell_ndvi, load_stack, unit_cell_ndvi

    render_window(run, ROW, source="local", data_root=tmp_path / "data")
    folder = run.root / "stacks" / "w1"
    stored = load_cell_ndvi(folder)
    unit = run.unit(UNIT)
    stack = load_stack(folder)
    assert stored.years == unit.years
    fresh = unit_cell_ndvi(unit.grid, stack)
    assert (stored.row0, stored.col0) == (fresh.row0, fresh.col0)
    np.testing.assert_allclose(stored.values, fresh.values, atol=1e-4, equal_nan=True)

    # One cell, from the pixels that lie in it.
    row, col = stored.row0 + 5, stored.col0 + 7
    mark = np.zeros(unit.grid.cell_shape, bool)
    mark[row, col] = True
    pixels = cells_to_pixels(mark, unit.grid, stack.bounds)
    assert pixels.sum() == 32 * 32
    red = stack.data[1, 0][pixels].astype(float).sum()
    nir = stack.data[1, 3][pixels].astype(float).sum()
    assert stored.values[1, 5, 7] == pytest.approx((nir - red) / (nir + red), abs=1e-4)
    # A cell that the stack covers only in part has no value, and so has a
    # cell the stack does not reach.
    assert np.isnan(stored.on_unit(unit.grid.cell_shape)[:, 0, 0]).all() or (
        stored.row0 == 0 and stored.col0 == 0
    )
    assert load_cell_ndvi(tmp_path) is None


def _with_rule(run: LabelRun, value: float | None) -> None:
    rules = run.cfg.rules.model_copy(update={"never_green_ndvi": value})
    run.cfg = run.cfg.model_copy(update={"rules": rules})


def test_never_green_ground_leaves_the_forest_mask(
    run: LabelRun, tmp_path: Path
) -> None:
    """Random imagery has an NDVI near 0, so a high threshold flags every cell."""
    render_window(run, ROW, source="local", data_root=tmp_path / "data")
    _write_answers(run)
    records = J.read_records(run.answers)
    plain = build_window(run, ROW, records)
    assert plain.labels.never_green is None  # off by default
    assert (plain.labels.reason == Reason.STABLE_FOREST).any()
    assert (plain.labels.reason == Reason.SPARSE_WOODY).any()

    _with_rule(run, 0.5)
    built = build_window(run, ROW, records)
    labels = built.labels
    assert labels.never_green.all()
    assert not labels.forest.any()  # the stored forest band follows
    for gone in (Reason.STABLE_FOREST, Reason.SPARSE_WOODY, Reason.RAISED_PROB):
        assert not (labels.reason == gone).any()
    assert (labels.reason == Reason.NONFOREST).sum() > (
        plain.labels.reason == Reason.NONFOREST
    ).sum()
    # What the candidates paint is untouched: the accepted loss, the
    # rejection, the disagreement and their rings.
    for kept in (
        Reason.ACCEPTED_FULL,
        Reason.REJECTED_FLAGGED,
        Reason.DISAGREE,
        Reason.MARGIN,
    ):
        assert ((labels.reason == kept) == (plain.labels.reason == kept)).all()
    # The stored cell NDVI is enough, the imagery is not needed.
    blind = build_window(run, ROW, records, with_stack=False)
    assert (blind.labels.reason == labels.reason).all()

    # A threshold below every cell changes nothing.
    _with_rule(run, -0.9)
    same = build_window(run, ROW, records)
    assert (same.labels.reason == plain.labels.reason).all()
    assert not same.labels.never_green.any()


def test_never_green_rule_without_cell_ndvi_is_not_applied(
    run: LabelRun, caplog: pytest.LogCaptureFixture
) -> None:
    _with_rule(run, 0.5)
    with caplog.at_level("WARNING"):
        built = build_window(run, ROW, [], with_stack=False)
    assert built.labels.never_green is None
    assert "never-green rule is not applied" in caplog.text


def test_a_stale_stack_is_refused(run: LabelRun, tmp_path: Path) -> None:
    """Window IDs follow the sample's order. Other ground under an old ID fails."""
    render_window(run, ROW, source="local", data_root=tmp_path / "data")
    moved = ROW.copy()
    moved["ix0"] = ROW["ix0"] + 1
    with pytest.raises(ValueError, match="another window"):
        render_window(run, moved, source="local", data_root=tmp_path / "data")
    # The same window is still read from its stack.
    assert render_window(run, ROW, source="local", data_root=tmp_path / "data") == 3


def test_a_rendered_folder_is_not_sampled_again(
    run: LabelRun, label_export: Path, tmp_path: Path
) -> None:
    cli = CliRunner()
    where = ["--label-dir", str(run.root), "--export-dir", str(label_export)]
    render_window(run, ROW, source="local", data_root=tmp_path / "data")
    result = cli.invoke(main, ["label", "sample", *where, "--overwrite"])
    assert result.exit_code != 0 and "already holds" in result.output
    result = cli.invoke(main, ["label", "sample", *where, "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Stage 1:" in result.output


def test_verdict_reports_windows_without_requests_and_takes_a_stage(
    run: LabelRun, label_export: Path, tmp_path: Path
) -> None:
    cli = CliRunner()
    where = ["--label-dir", str(run.root), "--export-dir", str(label_export)]
    dry = ["label", "verdict", *where, "--dry-run", "--cap", "1"]
    result = cli.invoke(main, dry)
    assert result.exit_code == 0, result.output
    assert "0 new verdict calls" in result.output
    assert "1 of 1 selected windows have no requests.jsonl" in result.output
    assert "w1" in result.output

    # windows.csv without a stage column cannot be filtered by stage.
    result = cli.invoke(main, [*dry, "--stage", "1"])
    assert result.exit_code != 0 and "no stage column" in result.output

    render_window(run, ROW, source="local", data_root=tmp_path / "data")
    table = pd.read_csv(run.root / "windows.csv")
    table["stage"] = 2
    table.to_csv(run.root / "windows.csv", index=False)
    result = cli.invoke(main, [*dry, "--stage", "1"])
    assert result.exit_code == 0 and "0 new verdict calls" in result.output
    assert "have no requests" not in result.output
    result = cli.invoke(main, [*dry, "--stage", "2"])
    assert result.exit_code == 0 and "6 new verdict calls" in result.output


def test_a_chip_in_two_windows_is_written_once(
    run: LabelRun, label_export: Path
) -> None:
    second = ROW.copy()
    second["window_id"] = "w2"
    second["ix0"] = ROW["ix0"] + 1  # shares the eastern column of w1
    pd.DataFrame([ROW, second]).to_csv(run.root / "windows.csv", index=False)
    where = ["--label-dir", str(run.root), "--export-dir", str(label_export)]
    result = CliRunner().invoke(main, ["label", "build", *where, "--no-imagery"])
    assert result.exit_code == 0, result.output
    assert "2 chips lie in two windows" in result.output
    chips = pd.read_parquet(run.root / "chips" / "chips.parquet")
    assert len(chips) == 6 and chips["chip_id"].is_unique
    shared = chips[chips["chip_id"].isin([f"e26919_x978_y{y}" for y in (14646, 14647)])]
    assert set(shared["window_id"]) == {"w1"}


def test_render_reports_how_the_dates_were_found(
    run: LabelRun, label_export: Path, tmp_path: Path
) -> None:
    where = ["--label-dir", str(run.root), "--export-dir", str(label_export)]
    data = ["--data-root", str(tmp_path / "data"), "--naip-source", "local"]
    cli = CliRunner()
    result = cli.invoke(main, ["label", "render", *where, *data, "--check"])
    assert result.exit_code == 0 and "3 without a sidecar" in result.output
    result = cli.invoke(main, ["label", "render", *where, *data])
    assert result.exit_code == 0, result.output
    assert "3 by mosaic_name" in result.output and "median date" in result.output


def test_polygon_render_without_verdicts_is_refused(
    run: LabelRun, label_export: Path, tmp_path: Path
) -> None:
    where = ["--label-dir", str(run.root), "--export-dir", str(label_export)]
    data = ["--data-root", str(tmp_path / "data"), "--naip-source", "local"]
    polygon = ["label", "render", *where, *data, "--kind", "polygon"]
    cli = CliRunner()
    result = cli.invoke(main, polygon)
    assert result.exit_code != 0 and "no verdict" in result.output
    result = cli.invoke(main, ["label", "render", *where, *data])
    assert result.exit_code == 0, result.output
    _write_answers(run, fit="too_small")
    result = cli.invoke(main, polygon)
    assert result.exit_code == 0, result.output
    assert "verdicts of these windows" in result.output


def test_a_window_whose_stack_lacks_a_year_is_left_out_without_imagery(
    run: LabelRun, tmp_path: Path
) -> None:
    """Such a window is skipped at render. A build on another machine agrees."""
    render_window(run, ROW, source="local", data_root=tmp_path / "data")
    assert build_window(run, ROW, [], with_stack=False) is not None
    path = run.root / "stacks" / "w1" / "meta.json"
    meta = json.loads(path.read_text())
    meta["failed"]["2018"] = "covers 0.30 of the part inside the unit"
    del meta["years"]["2018"]
    path.write_text(json.dumps(meta))
    assert build_window(run, ROW, [], with_stack=False) is None
