"""Tests for the blind audit: the draw, the page build and the answer server."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin
from shapely.geometry import Point

from csdv_core.audit import score as S
from csdv_core.audit.build import (
    build_site,
    bundle_site,
    label_overlay_image,
    read_entries,
)
from csdv_core.audit.draw import (
    AuditDesign,
    draw_items,
    extend_items,
    label_images,
    read_items,
)
from csdv_core.audit.server import latest_answers, make_server, read_answers

DATES = ["2012-06-01", "2014-06-01", "2016-06-01", "2018-06-01"]
PX = 64
GSD = 0.6
STATUSES = [
    ("accepted", "intact|loss_full|open", "clearcut", "harvest"),
    ("accepted", "intact|intact|loss_partial", "wind", "wind"),
    ("accepted", "loss_full|open|open", "fire", "fire"),
    ("rejected", "intact|rejected_flagged|intact", None, None),
    ("rejected_weak", "intact|rejected_weak|intact", None, None),
    ("disagree", "intact|disagree|open", None, None),
    ("date_disputed", "intact|date_disputed|loss_partial", None, None),
]


def _view(k: int) -> list[float]:
    minx, miny = 1000.0 + 100.0 * k, 5000.0
    return [minx, miny, minx + PX * GSD, miny + PX * GSD, PX]


def _write_tif(path: Path, data: np.ndarray, view: list[float]) -> None:
    transform = from_origin(view[0], view[3], GSD, GSD)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype="uint8",
        crs="EPSG:26916",
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture()
def label_dir(tmp_path: Path) -> Path:
    """A small label run: 10 seeded windows, a reviewed one, 7 views each."""
    root = tmp_path / "label"
    (root / "labels").mkdir(parents=True)
    (root / "chips").mkdir()
    windows, cands, areas, chips = [], [], [], []
    strata = ["di"] * 6 + ["other"] * 2 + ["quiet"] * 2 + ["reviewed"]
    for k, stratum in enumerate(strata):
        window = f"W{k + 1:04d}"
        view = _view(k)
        windows.append(
            {
                "window_id": window,
                "unit": "q0000001",
                "stratum": stratum,
                "incl_prob": np.nan if stratum == "reviewed" else 0.5,
            }
        )
        finding = "partial" if window == "W0006" else "complete"
        areas.append({"window_id": window, "finding": finding})
        stack = root / "stacks" / window
        stack.mkdir(parents=True)
        years = {d[:4]: {"date": d} for d in DATES}
        (stack / "meta.json").write_text(json.dumps({"years": years}))
        if window == "W0001":
            block = np.full((4, PX, PX), 90, dtype=np.uint8)
            block[:, ::3, ::5] = 200
            for year in years:
                _write_tif(stack / f"{year}.tif", block, view)
        change = np.zeros((len(DATES) - 1, PX, PX), dtype=np.uint8)
        change[1, 10:30, 10:30] = 1
        change[1, 40:50, :] = 255
        chip = f"e26916_x{k}_y0"
        _write_tif(root / "chips" / f"{chip}_change.tif", change, view)
        chips.append(
            {
                "chip_id": chip,
                "window_id": window,
                "unit": "q0000001",
                "epsg": 26916,
                "bounds": json.dumps(view[:4]),
                "dates": json.dumps(DATES),
            }
        )
        if stratum == "quiet":
            continue
        folder = root / "requests" / window
        folder.mkdir(parents=True)
        lines = []
        for j, (status, outcomes, change_type, agent) in enumerate(STATUSES):
            cand = f"q0000001-di-2014-{k * 10 + j}"
            images = [f"{cand}_v{i}.jpg" for i in range(len(DATES))]
            for name in images:
                Image.new("RGB", (PX, PX), (40, 90, 40)).save(folder / name)
            lines.append(
                json.dumps(
                    {
                        "cand_id": cand,
                        "kind": "verdict",
                        "prompt": "of the same area in a test county. North is up.",
                        "images": images,
                        "dates": DATES,
                        "view": view,
                    }
                )
            )
            cands.append(
                {
                    "window_id": window,
                    "cand_id": cand,
                    "unit": "q0000001",
                    "judged": True,
                    "status": status,
                    "outcomes": outcomes,
                    "change_type": change_type,
                    "agent": agent,
                    "geometry": Point(view[0], view[1]),
                }
            )
        (folder / "requests.jsonl").write_text("\n".join(lines) + "\n")
    # One candidate is seen through two windows.
    twin = dict(cands[0], window_id="W0002")
    cands.append(twin)
    with (root / "requests" / "W0002" / "requests.jsonl").open("a") as handle:
        first = json.loads(
            (root / "requests" / "W0001" / "requests.jsonl").read_text().splitlines()[0]
        )
        for name in first["images"]:
            Image.new("RGB", (PX, PX)).save(root / "requests" / "W0002" / name)
        handle.write(json.dumps(first) + "\n")
    pd.DataFrame(windows).to_csv(root / "windows.csv", index=False)
    pd.DataFrame(chips).to_parquet(root / "chips" / "chips.parquet")
    gpkg = root / "labels" / "labels.gpkg"
    gpd.GeoDataFrame(cands, crs="EPSG:5070").to_file(gpkg, layer="candidates")
    gpd.GeoDataFrame(
        areas, geometry=[Point(0, 0)] * len(areas), crs="EPSG:5070"
    ).to_file(gpkg, layer="reviewed_areas")
    return root


DESIGN = AuditDesign(
    n_accepted=6, n_rejected=4, n_disagree=4, n_windows=4, min_per_agent=1, n_shared=2
)


def test_label_images_numbers_the_later_date():
    assert label_images("intact|loss_full|open") == "3"
    assert label_images("loss_partial|open|loss_disputed") == "2,4"
    assert label_images("intact|rejected_flagged|intact") == ""


def test_draw_is_seeded_and_leaves_out_seen_and_reviewed(label_dir):
    first = draw_items(label_dir, DESIGN, {"W0003"})
    again = draw_items(label_dir, DESIGN, {"W0003"})
    pd.testing.assert_frame_equal(first, again)
    assert not first["window_id"].isin(["W0003", "W0011"]).any()
    cands = first[first["kind"] == "cand"]
    assert not cands["cand_id"].duplicated().any()
    target = first[~first["reserve"]]
    assert target.groupby("stratum").size().to_dict() == {
        "accepted": 6,
        "background": 4,
        "disagree": 4,
        "rejected": 4,
    }
    # The window with a partly judged interval is not a complete window.
    assert "W0006" not in set(first.loc[first["kind"] == "window", "window_id"])


def test_target_items_come_first_and_names_are_opaque(label_dir):
    items = draw_items(label_dir, DESIGN)
    for _, block in items.groupby("block"):
        reserve = block.sort_values("order")["reserve"].tolist()
        assert reserve == sorted(reserve)
    assert items["item"].str.fullmatch(r"[CB]\d{3}").all()
    shared = items[items["shared"]]
    assert shared.groupby("stratum").size().to_dict() == {
        "accepted": 2,
        "background": 2,
    }
    # Strata are mixed in block 1, not served one after the other.
    order = items[(items["block"] == 1) & ~items["reserve"]]["stratum"]
    assert (order != order.shift()).sum() > 3


def test_sub_strata_record_their_population(label_dir):
    items = draw_items(label_dir, DESIGN)
    accepted = items[items["stratum"] == "accepted"]
    assert set(accepted["sub"]) == {
        "accepted_diffuse_wind",
        "accepted_other_fire",
        "accepted_other_harvest",
    }
    for _, part in items.groupby("sub"):
        assert part["n_population"].nunique() == 1
        assert part["rank"].max() <= part["n_population"].iloc[0]


def test_larger_draw_extends_without_renaming(label_dir, tmp_path):
    small = draw_items(label_dir, DESIGN)
    path = tmp_path / "items.csv"
    small.to_csv(path, index=False)
    larger = AuditDesign(
        n_accepted=9,
        n_rejected=6,
        n_disagree=6,
        n_windows=5,
        min_per_agent=1,
        n_shared=2,
    )
    grown = extend_items(read_items(path), draw_items(label_dir, larger))
    head = grown.iloc[: len(small)].reset_index(drop=True)
    assert head["item"].tolist() == small["item"].tolist()
    assert head["cand_id"].tolist() == small["cand_id"].tolist()
    assert len(grown) > len(small)
    assert not grown["item"].duplicated().any()


def _entries(site: Path) -> list[dict]:
    text = (site / "items.js").read_text()
    return json.loads(text.split("window.AUDIT_ITEMS = ", 1)[1].rstrip().rstrip(";"))


def test_build_places_images_and_keeps_the_key_out(label_dir, tmp_path):
    items = draw_items(label_dir, DESIGN)
    audit_dir = tmp_path / "audit"
    report = build_site(label_dir, audit_dir, items, name="test")
    site = audit_dir / "site"
    for name in ("index.html", "app.js", "style.css", "items.js"):
        assert (site / name).exists()
    entries = {e["id"]: e for e in _entries(site)}
    assert set(entries) == set(report.built)
    cands = items[items["kind"] == "cand"]
    assert set(cands["item"]) <= set(entries)
    for _, row in cands.iterrows():
        entry = entries[row["item"]]
        assert len(entry["imgs"]) == len(DATES)
        assert all((site / src).exists() for src in entry["imgs"])
        assert entry["place"] == "a test county"
        expected = [int(n) for n in row["label_images"].split(",") if n]
        # A chip covers every interval, so each labelled one has an overlay.
        assert [lab["image"] for lab in entry["labels"]] == expected
    text = (site / "items.js").read_text()
    names = " ".join(str(p.relative_to(site)) for p in site.rglob("*"))
    for secret in ("q0000001", "accepted", "rejected", "disagree", "gem37", "terra"):
        assert secret not in text
        assert secret not in names


def test_window_items_wait_for_their_stacks(label_dir, tmp_path):
    design = AuditDesign(
        n_accepted=2,
        n_rejected=2,
        n_disagree=2,
        n_windows=9,
        min_per_agent=1,
        n_shared=1,
    )
    items = draw_items(label_dir, design)
    audit_dir = tmp_path / "audit"
    report = build_site(label_dir, audit_dir, items)
    windows = items[items["kind"] == "window"].set_index("window_id")
    assert "W0001" in windows.index
    assert "W0001" not in report.missing_stacks
    missing = set(report.missing_stacks) | set(report.missing_reserve)
    assert missing == set(windows.index) - {"W0001"}
    entry = {e["id"]: e for e in _entries(audit_dir / "site")}[
        windows.loc["W0001", "item"]
    ]
    assert entry["kind"] == "window"
    assert len(entry["imgs"]) == len(DATES)
    assert [lab["image"] for lab in entry["labels"]] == [2, 3, 4]
    with Image.open(audit_dir / "site" / entry["imgs"][0]) as img:
        assert img.size == (PX, PX)


def test_overlay_marks_loss_and_ignore_only(label_dir):
    chips = pd.read_parquet(label_dir / "chips" / "chips.parquet")
    view = _view(0)
    overlay = label_overlay_image(chips, label_dir / "chips", view, DATES[1], DATES[2])
    alpha = np.asarray(overlay)[..., 3]
    assert overlay.size == (PX, PX)
    assert (alpha[12:28, 12:28] > 0).all()
    assert alpha[10, 10] == 255
    assert alpha[0:8, :].max() == 0
    assert alpha[40:50, :].max() > 0
    # Ground outside every chip is veiled, and loss gets no edge against it.
    wide = [view[0] - 12.0, view[1], view[2] - 12.0, view[3], PX]
    shifted = np.asarray(
        label_overlay_image(chips, label_dir / "chips", wide, DATES[1], DATES[2])
    )
    assert (shifted[:, :19, 3] == 150).all() and (shifted[:, :19, :3] == 0).all()
    clear = np.asarray(
        label_overlay_image(
            chips, label_dir / "chips", wide, DATES[1], DATES[2], veil_unlabelled=False
        )
    )
    assert clear[:, :19, 3].max() == 0
    # Another flight date of the same years is the same interval.
    near = label_overlay_image(
        chips, label_dir / "chips", view, "2014-09-20", "2016-05-02"
    )
    assert np.array_equal(np.asarray(near), np.asarray(overlay))
    # No chip holds an interval between dates that are not consecutive.
    assert (
        label_overlay_image(chips, label_dir / "chips", view, DATES[0], DATES[2])
        is None
    )
    far = [9e6, 9e6, 9e6 + 10, 9e6 + 10, PX]
    assert (
        label_overlay_image(chips, label_dir / "chips", far, DATES[1], DATES[2]) is None
    )


@pytest.fixture()
def served(tmp_path: Path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<p>audit</p>")
    answers = tmp_path / "answers" / "tester.jsonl"
    server = make_server(site, answers, "tester", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://localhost:{server.server_address[1]}", answers
    server.shutdown()
    server.server_close()


def _post(url: str, payload: bytes) -> dict:
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())


def test_server_appends_answers_and_resumes(served):
    base, answers = served
    with urllib.request.urlopen(f"{base}/index.html", timeout=5) as response:
        assert b"audit" in response.read()
    with urllib.request.urlopen(f"{base}/state", timeout=5) as response:
        assert json.loads(response.read()) == {"interpreter": "tester", "answers": []}
    for record in (
        {"item": "C001", "answer": "none"},
        {"item": "C002", "answer": "loss", "loss_at": 3},
        {"item": "C002", "void": True},
        {"item": "C002", "answer": "unsure"},
        {"item": "C001", "void": True},
    ):
        assert _post(f"{base}/answer", json.dumps(record).encode())["ok"]
    stored = read_answers(answers)
    assert len(stored) == 5
    assert all(r["interpreter"] == "tester" and r["saved"] for r in stored)
    latest = latest_answers(stored)
    assert set(latest) == {"C002"}
    assert latest["C002"]["answer"] == "unsure"
    with urllib.request.urlopen(f"{base}/state", timeout=5) as response:
        assert len(json.loads(response.read())["answers"]) == 5


def test_server_refuses_a_bad_answer(served):
    base, answers = served
    for payload in (b"not json", b'{"answer": "none"}', b'{"item": ""}'):
        with pytest.raises(urllib.error.HTTPError) as err:
            _post(f"{base}/answer", payload)
        assert err.value.code == 400
    assert not answers.exists()


def test_bundle_holds_the_shared_items_only(label_dir, tmp_path):
    items = draw_items(label_dir, DESIGN)
    audit_dir = tmp_path / "audit"
    build_site(label_dir, audit_dir, items, name="test")
    out = tmp_path / "bundle"
    missing = bundle_site(audit_dir, out, items)
    meta, entries = read_entries(out)
    shared = items[items["shared"]]
    built = set(shared.loc[shared["kind"] == "cand", "item"])
    assert meta["name"] == "test_shared"
    assert {e["id"] for e in entries} >= built
    assert {e["id"] for e in entries} | set(missing) == set(shared["item"])
    assert {p.name for p in (out / "img").iterdir()} == {e["id"] for e in entries}
    for name in ("index.html", "app.js", "style.css", "README.txt"):
        assert (out / name).exists()
    first = out / entries[0]["imgs"][0]
    assert first.exists() and first.stat().st_nlink == 1
    with pytest.raises(FileExistsError):
        bundle_site(audit_dir, out, items)


def _answers(tmp_path: Path, name: str, records: list[dict]) -> Path:
    path = tmp_path / f"{name}.jsonl"
    lines = [
        json.dumps({"interpreter": name, "saved": f"2026-01-01T00:00:{i:02d}", **r})
        for i, r in enumerate(records)
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_exact_interval_matches_known_values():
    low, high = S.exact_interval(0, 60)
    assert low == 0.0 and high == pytest.approx(0.0596, abs=1e-3)
    low, high = S.exact_interval(58, 60)
    assert low == pytest.approx(0.885, abs=2e-3) and high == pytest.approx(
        0.996, abs=2e-3
    )
    assert S.exact_interval(5, 5) == (pytest.approx(0.478, abs=2e-3), 1.0)
    assert all(np.isnan(v) for v in S.exact_interval(0, 0))


def test_rates_weigh_items_by_inclusion_probability(tmp_path):
    items = pd.DataFrame(
        {
            "item": ["C001", "C002", "C003", "C004", "C005"],
            "block": 1,
            "kind": "cand",
            "stratum": ["accepted"] * 3 + ["rejected"] * 2,
            "sub": ["accepted_other_fire"] * 2
            + ["accepted_diffuse_wind", "rejected", "rejected"],
            "agent": ["fire", "fire", "wind", "", ""],
            "label_images": ["3", "4", "2,4", "", ""],
            "window_incl_prob": [0.5, 0.5, 0.1, 0.5, 0.5],
            "n_population": [10, 10, 4, 20, 20],
            "shared": [True, True, False, False, False],
        }
    )
    eb = _answers(
        tmp_path,
        "eb",
        [
            {
                "item": "C001",
                "answer": "loss",
                "loss_at": 3,
                "more_at": [],
                "type": "fire",
                "outline": "good",
                "seconds": 20,
            },
            {"item": "C002", "answer": "none", "seconds": 10},
            {"item": "C002", "void": True},
            {
                "item": "C002",
                "answer": "loss",
                "loss_at": 3,
                "more_at": [],
                "type": "harvest",
                "outline": "large",
                "seconds": 30,
            },
            {
                "item": "C003",
                "answer": "loss",
                "loss_at": 3,
                "more_at": [4],
                "type": "wind",
                "outline": "good",
                "seconds": 40,
            },
            {"item": "C004", "answer": "none", "seconds": 5},
            {"item": "C005", "answer": "unsure", "seconds": 60},
            {"item": "X999", "answer": "none", "seconds": 1},
        ],
    )
    other = _answers(
        tmp_path,
        "other",
        [
            {
                "item": "C001",
                "answer": "loss",
                "loss_at": 3,
                "more_at": [],
                "type": "fire",
                "outline": "good",
                "seconds": 15,
            },
            {"item": "C002", "answer": "none", "seconds": 12},
        ],
    )
    judged = S.judged_table(items, S.load_answers([eb, other]))
    mine = judged[judged["interpreter"] == "eb"].set_index("item")
    assert len(mine) == 5
    assert mine.loc["C002", "answer"] == "loss"
    assert mine["same_interval"].tolist()[:3] == [True, False, True]
    assert mine["same_type"].tolist()[:3] == [True, False, True]
    # Two of ten judged in a sub-stratum whose windows had probability 0.5.
    assert mine.loc["C001", "weight"] == pytest.approx(1 / (0.5 * 2 / 10))
    assert mine.loc["C003", "weight"] == pytest.approx(1 / (0.1 * 1 / 4))
    table = S.rates(judged).set_index(["interpreter", "stratum", "measure"])
    seen = table.loc[("eb", "accepted", "loss seen")]
    assert (seen["n"], seen["k"]) == (3, 3)
    interval = table.loc[("eb", "accepted", "loss in labelled interval")]
    assert (interval["n"], interval["k"]) == (3, 2)
    assert interval["weighted_rate"] == pytest.approx((10 + 40) / (10 + 10 + 40))
    assert table.loc[("eb", "accepted", "label extent holds"), "k"] == 2
    rejected = table.loc[("eb", "rejected", "loss seen")]
    assert (rejected["n"], rejected["k"], rejected["unsure"]) == (1, 0, 1)
    both = S.agreement(judged).set_index("item")
    assert set(both.index) == {"C001", "C002"}
    assert both.loc["C001", "all_agree"] and not both.loc["C002", "all_agree"]


# --------------------------------------------------------------------------
# Fifth stratum: model-only patches (block 3)
# --------------------------------------------------------------------------

CELL = 9.6
MAP_DATES = ["2012-06-01_2014-06-01", "2014-06-01_2016-06-01"]


def _write_map(path: Path, data: np.ndarray, names: list[str] | None = None) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype="uint8",
        crs="EPSG:26916",
        transform=from_origin(0.0, 960.0, CELL, CELL),
        nodata=255,
    ) as dst:
        dst.write(data)
        for k, name in enumerate(names or []):
            dst.set_band_description(k + 1, name)


@pytest.fixture()
def pair_run(tmp_path: Path) -> Path:
    """A map of 100 by 100 cells with four flagged patches in band 1.

    Patch a: 6 x 6 cells on forest, far from any candidate (kept).
    Patch b: 6 x 6 cells on forest, beside a candidate (left out).
    Patch c: 6 x 6 cells, not forest at the earlier date (left out).
    Patch d: 2 x 2 cells, under 0.3 ha (left out).
    Band 2 holds one more kept patch.
    """
    run = tmp_path / "pairmodel" / "holdout_q0000001"
    (run / "map").mkdir(parents=True)
    (run / "compare" / "events").mkdir(parents=True)
    (run / "compare" / "compare.json").write_text(json.dumps({"threshold": 0.8}))
    prob = np.full((2, 100, 100), 10, dtype=np.uint8)
    for r, c in ((10, 10), (10, 50), (60, 10)):
        prob[0, r : r + 6, c : c + 6] = 95
    prob[0, 80:82, 80:82] = 95
    prob[1, 40:46, 40:46] = 90
    forest = prob.copy()
    forest[0, 60:66, 10:16] = 0
    agent = np.full((2, 100, 100), 1, dtype=np.uint8)
    agent[1, 40:46, 40:46] = 3
    _write_map(run / "map" / "prob_q0000001.tif", prob, MAP_DATES)
    _write_map(run / "map" / "prob_forest_q0000001.tif", forest, MAP_DATES)
    _write_map(run / "map" / "agent_q0000001.tif", agent, MAP_DATES)
    return run


def _candidates_beside_b() -> gpd.GeoDataFrame:
    # Patch b spans cells 50..55 by rows 10..15: x 480-537.6, y 806.4-864.
    from shapely.geometry import box

    return gpd.GeoDataFrame(
        {"cand_id": ["c1"], "source": ["weak"]},
        geometry=[box(540.0, 820.0, 560.0, 840.0)],
        crs="EPSG:26916",
    )


def test_model_only_population_keeps_untouched_forest_patches(pair_run):
    from csdv_core.audit import model_only as M

    pop = M.patch_population(pair_run, "q0000001", _candidates_beside_b())
    assert sorted(pop["patch"]) == ["q0000001_1-1", "q0000001_2-1"]
    row = pop.set_index("patch").loc["q0000001_2-1"]
    assert row["agent"] == "fire"
    assert row["interval"] == MAP_DATES[1]
    assert row["area_ha"] == pytest.approx(36 * CELL**2 / 10_000)
    assert pop.geometry.area.sum() == pytest.approx(72 * CELL**2)
    seen = M.patch_population(
        pair_run,
        "q0000001",
        _candidates_beside_b(),
        exclude={"q0000001_1-1"},
    )
    assert list(seen["patch"]) == ["q0000001_2-1"]


def test_shown_patches_reads_the_close_up_names(pair_run):
    from csdv_core.audit import model_only as M

    (pair_run / "compare" / "events" / "model_only_2_2-41.png").write_bytes(b"")
    (pair_run / "compare" / "events" / "object_3.png").write_bytes(b"")
    assert M.shown_patches(pair_run, "q0000001") == {"q0000001_2-41"}


def test_model_only_draw_is_seeded_per_unit():
    from csdv_core.audit import model_only as M

    frame = gpd.GeoDataFrame(
        {
            "patch": [f"u{u}_1-{i}" for u in (1, 2) for i in range(5)],
            "unit": [f"u{u}" for u in (1, 2) for _ in range(5)],
        },
        geometry=[Point(i, 0) for i in range(10)],
        crs="EPSG:5070",
    )
    design = M.ModelOnlyDesign(n_per_unit=2, reserve=1)
    drawn = M.draw_patches(frame, design)
    again = M.draw_patches(frame.sample(frac=1.0, random_state=3), design)
    assert sorted(drawn["patch"]) == sorted(again["patch"])
    assert drawn.groupby("unit").size().tolist() == [3, 3]
    assert drawn.groupby("unit")["reserve"].sum().tolist() == [1, 1]
    assert set(drawn["n_population"]) == {5}


def _stratum(tmp_path: Path, patches: list[tuple[str, str, bool]]) -> Path:
    """Request images of model-only patches: (patch, interval, reserve)."""
    root = tmp_path / "model_only"
    for patch, _, _ in patches:
        folder = root / "requests" / patch
        folder.mkdir(parents=True)
        images = [f"{patch}_v{i}.jpg" for i in range(len(DATES))]
        for name in images:
            Image.new("RGB", (PX, PX), (90, 40, 40)).save(folder / name)
        record = {
            "cand_id": patch,
            "kind": "verdict",
            "prompt": "of the same area in a test county. North is up.",
            "images": images,
            "dates": DATES,
            "view": _view(0),
        }
        (folder / "requests.jsonl").write_text(json.dumps(record) + "\n")
    return root


def _patch_table(patches: list[tuple[str, str, bool]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patch": [p for p, _, _ in patches],
            "unit": "q0000001",
            "interval": [i for _, i, _ in patches],
            "agent": "harvest",
            "n_population": 7,
            "rank": range(1, len(patches) + 1),
            "reserve": [r for _, _, r in patches],
        }
    )


def test_model_only_items_join_as_block_3_without_renaming(label_dir, tmp_path):
    from csdv_core.audit import model_only as M

    items = draw_items(label_dir, DESIGN)
    patches = [
        ("q0000001_1-1", "2012-06-01_2014-06-01", False),
        ("q0000001_2-1", "2014-06-01_2016-06-01", False),
        ("q0000001_1-9", "2014-06-01_2018-06-01", True),
    ]
    stratum = _stratum(tmp_path, patches)
    out = M.append_items(items, _patch_table(patches), stratum)
    assert out.head(len(items)).equals(items)
    block = out[out["block"] == 3].set_index("cand_id")
    assert sorted(block["item"]) == ["M001", "M002", "M003"]
    assert block.loc["q0000001_1-9", "reserve"]
    assert block.loc["q0000001_1-9", "order"] == 3
    # The later date of the interval, as an image number of the request.
    assert block.loc["q0000001_1-1", "label_images"] == "2"
    assert block.loc["q0000001_1-9", "label_images"] == "4"
    assert block["stratum"].eq("model_only").all()
    assert out["item"].is_unique
    # A second call adds nothing.
    assert len(M.append_items(out, _patch_table(patches), stratum)) == len(out)


def test_extend_names_a_new_block_by_its_letter(label_dir):
    items = draw_items(label_dir, DESIGN)
    extra = items[items["block"] == 1].head(2).copy()
    extra["block"] = 3
    extra["cand_id"] = ["new-a", "new-b"]
    out = extend_items(items, pd.concat([items, extra]))
    assert out[out["block"] == 3]["item"].tolist() == ["M001", "M002"]


def test_build_shows_patches_as_candidates_without_labels(label_dir, tmp_path):
    from csdv_core.audit import model_only as M
    from csdv_core.audit.build import MODEL_ONLY_DIR

    items = draw_items(label_dir, DESIGN)
    audit_dir = tmp_path / "audit"
    patches = [("q0000001_1-1", "2012-06-01_2014-06-01", False)]
    stratum = _stratum(audit_dir, patches)
    assert stratum == audit_dir / MODEL_ONLY_DIR
    items = M.append_items(items, _patch_table(patches), stratum)
    build_site(label_dir, audit_dir, items, name="test")
    entries = {e["id"]: e for e in _entries(audit_dir / "site")}
    entry = entries["M001"]
    assert entry["kind"] == "cand" and entry["block"] == 3
    assert entry["labels"] == []
    assert len(entry["imgs"]) == len(DATES)
    text = (audit_dir / "site" / "items.js").read_text()
    assert "q0000001" not in text and "model_only" not in text


def test_rates_report_the_model_only_stratum(tmp_path):
    items = pd.DataFrame(
        {
            "item": ["M001", "M002", "M003"],
            "block": 3,
            "kind": "patch",
            "stratum": "model_only",
            "sub": ["u1", "u1", "u2"],
            "agent": ["harvest", "fire", "harvest"],
            "label_images": ["3", "2", "4"],
            "window_incl_prob": 1.0,
            "n_population": [10, 10, 4],
            "shared": False,
        }
    )
    eb = _answers(
        tmp_path,
        "eb",
        [
            {
                "item": "M001",
                "answer": "loss",
                "loss_at": 3,
                "more_at": [],
                "type": "harvest",
                "seconds": 20,
            },
            {"item": "M002", "answer": "none", "seconds": 10},
            {
                "item": "M003",
                "answer": "loss",
                "loss_at": 2,
                "more_at": [],
                "type": "fire",
                "seconds": 30,
            },
        ],
    )
    table = S.rates(S.judged_table(items, S.load_answers([eb]))).set_index(
        ["stratum", "measure"]
    )
    seen = table.loc[("model_only", "loss seen")]
    assert (seen["n"], seen["k"]) == (3, 2)
    # Weights: 10/2 for each of u1, 4/1 for u2.
    assert seen["weighted_rate"] == pytest.approx((5 + 4) / (5 + 5 + 4))
    interval = table.loc[("model_only", "loss in model interval")]
    assert (interval["n"], interval["k"]) == (3, 1)
    cause = table.loc[("model_only", "cause equals model agent")]
    assert (cause["n"], cause["k"]) == (2, 1)
