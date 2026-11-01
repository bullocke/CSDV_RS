"""csdv_core.review.app: the Solara review app (``csdv review app``).

Two pages:

Review (``/``, ``/?id=R017``)
    One sample polygon at a time, with every detection around it, NAIP 0.6 m
    of every year, CHM overlays, a year strip and the interpretation form. It
    opens at the first polygon of the random sample without a label.
Explore (``/explore``)
    A whole unit with all its detections. Clicking an outline shows its
    attributes and links it to the review, or adds it as an opportunistic pick.

Keyboard shortcuts (outside text fields): 1 to 4 set the verdict, Enter saves
and opens the next unlabelled polygon, s saves and stays, b and a show the
years before and after the detected interval, the arrow keys step through the
years, w swipes between the previous and the selected year, h hides the
outlines, c shows the 19.2 m cells, n and p go to the next and previous ID.

The app runs in one process for one reviewer, so the store and the imagery
sources are shared module state.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs

import geopandas as gpd
import pandas as pd
import solara
from shapely.geometry.base import BaseGeometry

from csdv_core.review.config import review_config
from csdv_core.review.context import nearby_objects, onset_colours, year_strip
from csdv_core.review.imagery import ImagerySource, naip_sources
from csdv_core.review.mapview import ReviewMap
from csdv_core.review.sample import (
    STRATUM_DETECTED,
    ExportedRun,
    list_runs,
    opportunistic_row,
)
from csdv_core.review.store import (
    CHANGE_TYPES,
    CONTEXT_FLAGS,
    OUTLINE_CHOICES,
    UNSURE,
    VERDICTS,
    Interpretation,
    ReviewStore,
)

logger = logging.getLogger(__name__)

OVERLAYS = ["none", "CHM", "CHM change", "onset", "probability"]
#: Lets a row of toggle buttons wrap in the narrow form column.
WRAP = {"flex-wrap": "wrap", "height": "auto"}
HERE = Path(__file__).parent

# ---- shared state -------------------------------------------------------


@lru_cache(maxsize=1)
def _store() -> ReviewStore | None:
    path = review_config().store_path
    return ReviewStore(path) if path.exists() else None


@lru_cache(maxsize=1)
def _runs() -> dict[str, ExportedRun]:
    cfg = review_config()
    return {r.unit: r for r in list_runs(cfg.export_dir, cfg.run_tag)}


_sources: dict[str, dict[int, ImagerySource]] = {}
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def sources_for(unit: str) -> dict[int, ImagerySource]:
    """NAIP tile sources of a unit (cached for the life of the process)."""
    from csdv_core.io.paths import project_paths

    with _locks_guard:
        lock = _locks.setdefault(unit, threading.Lock())
    with lock:
        if unit not in _sources:
            run = _runs()[unit]
            try:
                record = json.loads((run.run_dir / "unit.json").read_text())
            except (OSError, ValueError):
                record = {}
            cache = project_paths().cache_root / "review" / "pc_items.json"
            _sources[unit] = naip_sources(unit, run.years, record, cache)
        return _sources[unit]


def _warm_sources() -> None:
    """Find the imagery of every unit in the background, so pages open fast.

    Units are prepared in the order of the next open sample polygons.
    """
    store = _store()
    order: list[str] = []
    if store is not None:
        done = store.done_ids()
        for sid in store.ordered_ids():
            unit = str(store.row(sid)["unit"])
            if sid not in done and unit not in order:
                order.append(unit)
    for unit in order + [u for u in _runs() if u not in order]:
        try:
            sources_for(unit)
        except Exception:
            logger.exception("%s: could not prepare imagery", unit)


_warm_started = threading.Event()


def start_warming() -> None:
    """Start :func:`_warm_sources` once, when the first page is shown."""
    if not _warm_started.is_set():
        _warm_started.set()
        threading.Thread(
            target=_warm_sources, daemon=True, name="review-imagery"
        ).start()


def overlay_url(kind: str, unit: str, year: int, prev: int | None) -> str | None:
    """Tile URL of an overlay, or None."""
    xyz = "{z}/{x}/{y}.png"
    if kind == "CHM":
        return f"/tiles/ql/{unit}/chm_{year}_3m/{xyz}"
    if kind == "CHM change" and prev is not None:
        return f"/tiles/dchm/{unit}/{prev}/{year}/{xyz}"
    if kind == "onset":
        return f"/tiles/run/{unit}/onset/{xyz}"
    if kind == "probability":
        return f"/tiles/run/{unit}/maxprob/{xyz}"
    return None


@dataclass
class Focus:
    """A sample polygon and what is drawn around it."""

    sid: str
    row: pd.Series
    run: ExportedRun
    geom_utm: BaseGeometry
    geom_4326: BaseGeometry
    focus_utm: gpd.GeoDataFrame
    outlines: gpd.GeoDataFrame
    cells: gpd.GeoDataFrame
    colours: dict[int, str]

    @property
    def year_before(self) -> int:
        return int(self.row["year_before"])

    @property
    def onset_year(self) -> int:
        return int(self.row["onset_year"])


def load_focus(store: ReviewStore, sid: str) -> Focus:
    row = store.row(sid)
    run = _runs()[row["unit"]]
    one = store.sample[store.sample["sample_id"] == sid]
    focus_utm = one.to_crs(run.crs)
    geom_utm = focus_utm.geometry.iloc[0]
    outlines, cells = nearby_objects(run, geom_utm)
    return Focus(
        sid=sid,
        row=row,
        run=run,
        geom_utm=geom_utm,
        geom_4326=one.to_crs(4326).geometry.iloc[0],
        focus_utm=focus_utm.assign(onset_year=row["onset_year"]),
        outlines=outlines,
        cells=cells,
        colours=onset_colours(run.years),
    )


def _prev_year(years: list[int], year: int) -> int | None:
    i = years.index(year)
    return years[i - 1] if i > 0 else None


# ---- components ---------------------------------------------------------


@solara.component_vue(str(HERE / "keys.vue"))
def KeyListener(event_key: Callable[[str], None]):
    """Calls ``event_key(key)`` for review shortcut keys."""


@solara.component
def Legend(colours: dict[int, str]):
    chips = " ".join(
        f'<span style="border:2px solid {c};padding:0 4px;margin-right:4px;'
        f'border-radius:3px">{y}</span>'
        for y, c in colours.items()
    )
    solara.HTML(
        tag="div",
        unsafe_innerHTML=f'<span style="font-size:12px">Onset year: {chips}</span>',
    )


@solara.component
def MapToolbar(
    years: list[int],
    year: int,
    on_year: Callable[[int], None],
    sources: dict[int, ImagerySource],
    swipe: bool,
    on_swipe: Callable[[bool], None],
    overlay: str,
    on_overlay: Callable[[str], None],
    opacity: float,
    on_opacity: Callable[[float], None],
    outlines_on: bool,
    on_outlines: Callable[[bool], None],
    cells_on: bool,
    on_cells: Callable[[bool], None],
    year_before: int | None = None,
    onset_year: int | None = None,
):
    with solara.Row(gap="8px", style={"align-items": "center", "flex-wrap": "wrap"}):
        solara.ToggleButtonsSingle(
            value=year, values=years, on_value=on_year, dense=True, style=WRAP
        )
        if year_before is not None and onset_year is not None:
            solara.Button(
                f"Before {year_before}",
                on_click=lambda: on_year(year_before),
                small=True,
            )
            solara.Button(
                f"After {onset_year}", on_click=lambda: on_year(onset_year), small=True
            )
        solara.Switch(label="Swipe prev | year", value=swipe, on_value=on_swipe)
    with solara.Row(gap="8px", style={"align-items": "center", "flex-wrap": "wrap"}):
        with solara.Column(style={"min-width": "170px", "max-width": "190px"}):
            solara.Select(
                "Overlay", values=OVERLAYS, value=overlay, on_value=on_overlay
            )
        with solara.Column(style={"min-width": "160px", "max-width": "200px"}):
            solara.SliderFloat(
                "Opacity",
                value=opacity,
                min=0.1,
                max=1.0,
                step=0.05,
                on_value=on_opacity,
            )
        solara.Switch(label="Outlines", value=outlines_on, on_value=on_outlines)
        solara.Switch(label="19.2 m cells", value=cells_on, on_value=on_cells)
        kinds = sorted({s.kind for s in sources.values()})
        note = ", ".join(
            f"{k}: {', '.join(str(y) for y, s in sources.items() if s.kind == k)}"
            for k in kinds
        )
        solara.Text(f"NAIP {note}", style={"font-size": "12px", "color": "#666"})


@solara.component
def MapPanel(
    ctrl: ReviewMap,
    unit: str,
    years: list[int],
    initial_year: int,
    reset_key: str,
    year_before: int | None = None,
    onset_year: int | None = None,
    register_keys: Callable[[Callable[[str], bool]], None] | None = None,
):
    """Toolbar and map; keeps the imagery and overlays in step with the controls."""
    year, set_year = solara.use_state(initial_year)
    swipe, set_swipe = solara.use_state(False)
    overlay, set_overlay = solara.use_state("none")
    opacity, set_opacity = solara.use_state(0.6)
    outlines_on, set_outlines_on = solara.use_state(True)
    cells_on, set_cells_on = solara.use_state(False)

    def reset() -> None:
        set_year(initial_year)

    solara.use_effect(reset, [reset_key])
    year_now = year if year in years else initial_year
    prev = _prev_year(years, year_now)
    sources = solara.use_memo(lambda: sources_for(unit), [unit])

    def apply_imagery() -> None:
        if swipe and prev is not None:
            ctrl.set_split(sources[prev], sources[year_now])
        else:
            ctrl.set_imagery(sources[year_now])

    def apply_overlay() -> None:
        ctrl.set_overlay(overlay_url(overlay, unit, year_now, prev), opacity)

    def apply_outlines() -> None:
        ctrl.show_outlines(outlines_on)

    def apply_cells() -> None:
        ctrl.show_cells(cells_on)

    solara.use_effect(apply_imagery, [unit, year_now, swipe])
    solara.use_effect(apply_overlay, [unit, year_now, overlay, opacity])
    solara.use_effect(apply_outlines, [outlines_on])
    solara.use_effect(apply_cells, [cells_on])

    def step(delta: int) -> None:
        i = min(max(years.index(year_now) + delta, 0), len(years) - 1)
        set_year(years[i])

    def handle_key(key: str) -> bool:
        if key == "b" and year_before is not None:
            set_year(year_before)
        elif key == "a" and onset_year is not None:
            set_year(onset_year)
        elif key == "ArrowLeft":
            step(-1)
        elif key == "ArrowRight":
            step(1)
        elif key == "w":
            set_swipe(not swipe)
        elif key == "h":
            set_outlines_on(not outlines_on)
        elif key == "c":
            set_cells_on(not cells_on)
        else:
            return False
        return True

    if register_keys is not None:
        register_keys(handle_key)

    MapToolbar(
        years=years,
        year=year_now,
        on_year=set_year,
        sources=sources,
        swipe=swipe,
        on_swipe=set_swipe,
        overlay=overlay,
        on_overlay=set_overlay,
        opacity=opacity,
        on_opacity=set_opacity,
        outlines_on=outlines_on,
        on_outlines=set_outlines_on,
        cells_on=cells_on,
        on_cells=set_cells_on,
        year_before=year_before,
        onset_year=onset_year,
    )
    solara.display(ctrl.map)


@solara.component
def InfoCard(focus: Focus, saved: Interpretation | None, clicked: dict | None):
    r = focus.row
    flags = []
    if bool(r["growth"]):
        flags.append("growth object")
    if not bool(r["follow_up"]):
        flags.append("no follow-up image")
    if r["stratum"] != STRATUM_DETECTED:
        flags.append(str(r["stratum"]))
    status = f"**saved: {saved.verdict}**" if saved else "not interpreted"
    lines = [
        f"### {focus.sid} · {r['site_label'] or r['unit']}",
        f"{r['unit']} · object {int(r['object_id'])} · {status}",
        f"Detected **{r['interval']}** · outline {r['assessed_ha']:.2f} ha "
        f"({r['geom_src']}) · object {r['area_ha']:.2f} ha · "
        f"mean P {r['mean_prob']:.2f}",
    ]
    if flags:
        lines.append("Flags: " + ", ".join(flags))
    if clicked:
        lines.append(
            f"Clicked: object {clicked.get('object_id')} · "
            f"{clicked.get('interval')} · {float(clicked.get('area_ha', 0)):.2f} ha"
        )
    solara.Markdown("\n\n".join(lines))


@solara.component
def FormPanel(
    sid: str,
    focus: Focus,
    ctrl: ReviewMap,
    on_saved: Callable[[bool], None],
    on_key: Callable[[str], None],
):
    """The interpretation form of one polygon. Keyed by sample ID."""
    store = _store()
    assert store is not None
    saved = store.get(sid)
    started = solara.use_memo(time.time, [])
    verdict, set_verdict = solara.use_state(saved.verdict if saved else None)
    change_type, set_change_type = solara.use_state(
        (saved.change_type or None) if saved else None
    )
    ref_interval, set_ref_interval = solara.use_state(
        (saved.ref_interval or focus.row["interval"])
        if saved
        else focus.row["interval"]
    )
    outline, set_outline = solara.use_state((saved.outline or None) if saved else None)
    flags, set_flags = solara.use_state(
        [f for f in saved.context.split(",") if f] if saved else []
    )
    confidence, set_confidence = solara.use_state(
        (saved.confidence or 2) if saved else 2
    )
    notes, set_notes = solara.use_state(saved.notes if saved else "")
    message, set_message = solara.use_state("")

    def save(advance: bool) -> None:
        if verdict is None:
            set_message("Choose a verdict first (keys 1 to 4).")
            return
        geometry = ctrl.drawn_geometry()
        rec = Interpretation(
            sample_id=sid,
            verdict=verdict,
            change_type=change_type or "",
            ref_interval="" if verdict == "rejected" else (ref_interval or ""),
            outline=outline or "",
            context=",".join(flags),
            confidence=confidence,
            notes=notes,
            interpreter=review_config().interpreter,
            seconds_spent=round(time.time() - started, 1),
        )
        try:
            store.save(rec, geometry=geometry, clear_geometry=geometry is None)
        except Exception as exc:
            logger.exception("save failed")
            set_message(f"Save failed: {exc}")
            return
        set_message("")
        on_saved(advance)

    def key(k: str) -> None:
        if k in "1234" and len(k) == 1:
            set_verdict(VERDICTS[int(k) - 1])
        elif k == "Enter":
            save(True)
        elif k == "s":
            save(False)
        else:
            on_key(k)

    KeyListener(event_key=key)
    intervals = focus.run.intervals + [UNSURE]
    solara.Text("Verdict (1 to 4)", style={"font-weight": "bold"})
    solara.ToggleButtonsSingle(
        value=verdict, values=list(VERDICTS), on_value=set_verdict, style=WRAP
    )
    solara.Select(
        "Change type",
        values=list(CHANGE_TYPES),
        value=change_type,
        on_value=set_change_type,
    )
    solara.Select(
        "Interval of the disturbance",
        values=intervals,
        value=ref_interval if ref_interval in intervals else None,
        on_value=set_ref_interval,
    )
    solara.Text("Outline", style={"font-weight": "bold"})
    solara.ToggleButtonsSingle(
        value=outline, values=list(OUTLINE_CHOICES), on_value=set_outline, style=WRAP
    )
    with solara.Row(gap="6px"):
        solara.Button(
            "Edit detected outline",
            on_click=lambda: ctrl.load_drawing(focus.geom_4326),
            small=True,
            outlined=True,
        )
        solara.Button(
            "Clear drawing",
            on_click=lambda: ctrl.load_drawing(None),
            small=True,
            outlined=True,
        )
    solara.Text(
        "Draw or edit with the map toolbar. A drawing is saved as the redrawn outline.",
        style={"font-size": "12px", "color": "#666"},
    )
    solara.Text("Context", style={"font-weight": "bold"})
    solara.ToggleButtonsMultiple(
        value=flags, values=list(CONTEXT_FLAGS), on_value=set_flags, style=WRAP
    )
    with solara.Row(gap="8px", style={"align-items": "center"}):
        solara.Text("Confidence")
        solara.ToggleButtonsSingle(
            value=confidence, values=[1, 2, 3], on_value=set_confidence
        )
    solara.InputTextArea("Notes", value=notes, on_value=set_notes, rows=2)
    with solara.Row(gap="8px"):
        solara.Button("Save (s)", on_click=lambda: save(False), outlined=True)
        solara.Button(
            "Save + next open (Enter)", on_click=lambda: save(True), color="primary"
        )
    if message:
        solara.Warning(message, dense=True)


@solara.component
def ProgressPanel(store: ReviewStore, revision: int):
    done, total = store.progress()
    solara.ProgressLinear(value=100 * done / max(total, 1))
    solara.Text(f"{done} of {total} sample polygons interpreted")
    with solara.Details(summary="By site"):
        table = store.progress_by_unit()
        lines = ["| unit | site | done |", "|---|---|---|"] + [
            f"| {r.unit} | {r.label} | {r.done}/{r.total} |" for r in table.itertuples()
        ]
        solara.Markdown("\n".join(lines))


@solara.component
def ReviewPage():
    cfg = review_config()
    store = _store()
    start_warming()
    if store is None:
        solara.Error(
            f"No review sample at {cfg.store_path}. Draw one with "
            f"`csdv review sample --run-tag {cfg.run_tag}` and restart the app."
        )
    elif not store.ordered_ids():
        solara.Error("The review sample is empty.")
    else:
        ReviewView(store)


@solara.component
def ReviewView(store: ReviewStore):
    cfg = review_config()
    router = solara.use_router()
    revision, set_revision = solara.use_state(0)
    clicked, set_clicked = solara.use_state(None)
    key_handler = solara.use_ref(None)
    goto_text, set_goto_text = solara.use_state("")

    ids = store.ordered_ids()
    requested = parse_qs(router.search or "").get("id", [None])[0]
    sid = requested if requested in ids else (store.first_open() or ids[0])
    focus = solara.use_memo(lambda: load_focus(store, sid), [sid])
    ctrl = solara.use_memo(ReviewMap, [])

    def on_focus() -> None:
        ctrl.set_context(focus.outlines, focus.cells, focus.focus_utm, focus.colours)
        ctrl.zoom_to(focus.geom_4326)
        ctrl.load_drawing(store.redrawn_geometry(sid))
        ctrl.on_outline_click(set_clicked)
        set_clicked(None)

    solara.use_effect(on_focus, [sid])

    def make_strip() -> bytes:
        quicklooks = cfg.quicklook_root / focus.run.unit
        return year_strip(
            focus.run,
            quicklooks,
            focus.geom_utm,
            focus.outlines,
            focus.year_before,
            focus.onset_year,
        )

    strip = solara.use_thread(make_strip, dependencies=[sid])

    def go(target: str | None) -> None:
        if target:
            router.push(f"/?id={target}")

    i = ids.index(sid)
    prev_id = ids[i - 1] if i > 0 else None
    next_id = ids[i + 1] if i + 1 < len(ids) else None

    def on_saved(advance: bool) -> None:
        set_revision(revision + 1)
        if advance:
            nxt = store.next_open(sid)
            if nxt is None:
                logger.info("Every sample polygon is interpreted")
            go(nxt)

    def on_key(k: str) -> None:
        if k == "n":
            go(next_id)
        elif k == "p":
            go(prev_id)
        elif key_handler.current is not None:
            key_handler.current(k)

    def register_keys(handler: Callable[[str], bool]) -> None:
        key_handler.current = handler

    def goto(text: str | None) -> None:
        set_goto_text("")
        text = (text or "").strip().upper()
        if text.isdigit():
            text = f"R{int(text):03d}"
        if text in ids:
            go(text)

    with solara.Columns([8, 4], gutters_dense=True):
        with solara.Column(gap="6px"):
            MapPanel(
                ctrl=ctrl,
                unit=focus.run.unit,
                years=focus.run.years,
                initial_year=focus.onset_year,
                reset_key=sid,
                year_before=focus.year_before,
                onset_year=focus.onset_year,
                register_keys=register_keys,
            )
            Legend(focus.colours)
            if strip.state == solara.ResultState.FINISHED and strip.value:
                solara.Image(strip.value, width="100%")
            elif strip.state == solara.ResultState.ERROR:
                solara.Warning(f"Year strip failed: {strip.error}", dense=True)
            else:
                solara.Text("Rendering the year strip...")
        with solara.Column(gap="6px"):
            with solara.Row(gap="6px", style={"align-items": "center"}):
                solara.Button(
                    "◀ p",
                    on_click=lambda: go(prev_id),
                    disabled=prev_id is None,
                    small=True,
                )
                solara.Button(
                    "n ▶",
                    on_click=lambda: go(next_id),
                    disabled=next_id is None,
                    small=True,
                )
                solara.Button(
                    "Next open",
                    on_click=lambda: go(store.next_open(sid)),
                    small=True,
                    color="primary",
                    outlined=True,
                )
                with solara.Column(style={"min-width": "110px", "max-width": "130px"}):
                    solara.InputText(
                        "Go to ID",
                        value=goto_text,
                        on_value=goto,
                        continuous_update=False,
                    )
            InfoCard(focus, store.get(sid), clicked)
            FormPanel(
                sid=sid, focus=focus, ctrl=ctrl, on_saved=on_saved, on_key=on_key
            ).key(f"{sid}-{revision}")
            ProgressPanel(store, revision)


@solara.component
def ExplorePage():
    cfg = review_config()
    start_warming()
    if not _runs():
        solara.Error(f"No exported runs under {cfg.export_dir}/runs/{cfg.run_tag}")
    else:
        ExploreView()


@solara.component
def ExploreView():
    store = _store()
    runs = _runs()
    router = solara.use_router()
    units = list(runs)
    labels = {u: f"{u} · {r.label}" for u, r in runs.items()}
    unit, set_unit = solara.use_state(units[0])
    selected, set_selected = solara.use_state(None)
    message, set_message = solara.use_state("")
    ctrl = solara.use_memo(lambda: ReviewMap(height="72vh"), [])
    run = runs[unit]
    colours = solara.use_memo(lambda: onset_colours(run.years), [unit])

    def load_unit() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        outlines, cells = nearby_objects(run, _unit_box(run), radius_m=0)
        return outlines, cells

    layers = solara.use_memo(load_unit, [unit])

    def on_unit() -> None:
        outlines, cells = layers
        empty = gpd.GeoDataFrame(geometry=[], crs=run.crs)
        ctrl.set_context(outlines, cells, empty, colours, simplify_m=1.0)
        ctrl.zoom_to(_unit_box(run, crs=4326), pad=1.1)
        ctrl.on_outline_click(set_selected)
        set_selected(None)
        set_message("")

    solara.use_effect(on_unit, [unit])

    def in_sample(object_id: int) -> str | None:
        if store is None:
            return None
        s = store.sample
        hit = s[(s["unit"] == unit) & (s["object_id"] == object_id)]
        return None if hit.empty else str(hit.sort_values("order")["sample_id"].iloc[0])

    def add_pick(object_id: int) -> None:
        if store is None:
            set_message("No review sample yet (csdv review sample).")
            return
        rows = opportunistic_row(store.sample, run, object_id)
        store.add_sample_rows(rows)
        router.push(f"/?id={rows['sample_id'].iloc[0]}")

    with solara.Columns([9, 3], gutters_dense=True):
        with solara.Column(gap="6px"):
            MapPanel(
                ctrl=ctrl,
                unit=unit,
                years=run.years,
                initial_year=run.years[-1],
                reset_key=unit,
            )
            Legend(colours)
        with solara.Column(gap="8px"):
            solara.Select(
                "Unit",
                values=units,
                value=unit,
                on_value=set_unit,
            )
            solara.Text(labels[unit])
            outlines, _ = layers
            solara.Text(f"{len(outlines)} detected objects, years {run.years}")
            if selected:
                oid = int(selected.get("object_id"))
                solara.Markdown(
                    f"**Object {oid}** · {selected.get('interval')} · "
                    f"{float(selected.get('area_ha', 0)):.2f} ha · "
                    f"{selected.get('geom_src')}"
                    + (" · growth" if str(selected.get("growth")) == "True" else "")
                )
                sid = in_sample(oid)
                if sid:
                    solara.Button(
                        f"Review {sid}",
                        on_click=lambda: router.push(f"/?id={sid}"),
                        color="primary",
                    )
                else:
                    solara.Button(
                        "Add to review (opportunistic)",
                        on_click=lambda: add_pick(oid),
                        outlined=True,
                    )
            else:
                solara.Text("Click an outline to see it.")
            if message:
                solara.Warning(message, dense=True)


def _unit_box(run: ExportedRun, crs: str | None = None) -> BaseGeometry:
    """The unit's extent (from ``unit.json``), in the run CRS or ``crs``."""
    from shapely.geometry import box

    try:
        record = json.loads((run.run_dir / "unit.json").read_text())
        bbox = record["bbox_4326"]
        geom = gpd.GeoSeries([box(*bbox)], crs=4326)
    except (OSError, ValueError, KeyError):
        cells = gpd.read_file(run.gpkg, layer="objects")
        geom = gpd.GeoSeries([box(*cells.total_bounds)], crs=cells.crs)
    return geom.to_crs(crs or run.crs).iloc[0]


@solara.component
def Layout(children=[]):  # noqa: B006 (Solara layout signature)
    """App bar with the two pages, instead of tabs named after the paths."""
    router = solara.use_router()
    cfg = review_config()
    with solara.AppLayout(title="", navigation=False, children=[]):
        with solara.AppBar():
            solara.Button("Review", on_click=lambda: router.push("/"), text=True)
            solara.Button(
                "Explore", on_click=lambda: router.push("/explore"), text=True
            )
            solara.Text(
                f"run tag {cfg.run_tag} · {cfg.store_path.name}",
                style={"font-size": "13px", "margin-left": "16px", "opacity": "0.8"},
            )
        solara.Column(children=children, style={"padding": "6px"})


routes = [
    solara.Route(path="/", component=ReviewPage, label="Review", layout=Layout),
    solara.Route(path="explore", component=ExplorePage, label="Explore"),
]
