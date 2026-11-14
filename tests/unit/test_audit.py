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

from csdv_core.audit.build import build_site, label_overlay_image
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
