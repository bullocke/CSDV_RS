"""csdv_core.label.cli: ``csdv label``, the labelling pipeline step by step."""

from __future__ import annotations

import logging
from pathlib import Path

import click

__all__ = ["label"]

#: What one verdict call cost in the example run, US dollars (two models, mixed
#: image sizes). Used only to forecast a run before it starts.
CALL_COST = 0.0081


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s"
    )
    for name in ("httpx", "rasterio", "csdv_core.download"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _common(func):
    """Options that say where the label run and the disturbance runs are."""
    options = [
        click.option(
            "--label-dir",
            required=True,
            type=click.Path(file_okay=False, path_type=Path),
            help="Folder of the label run.",
        ),
        click.option(
            "--export-dir",
            default=None,
            type=click.Path(exists=True, file_okay=False, path_type=Path),
            help="Export folder with runs/<tag>/<unit>/. Default: "
            "$CSDV_EXPORT_DIR, else results/chpc_exports.",
        ),
        click.option("--run-tag", default="default", show_default=True),
        click.option(
            "--runs-dir",
            default=None,
            type=click.Path(exists=True, file_okay=False, path_type=Path),
            help="Folder with one run folder per unit, instead of an export. "
            "On CHPC: $CSDV_RESULTS_ROOT/disturbance, with --run-subdir.",
        ),
        click.option(
            "--run-subdir",
            default="",
            help="Sub-folder of each unit folder that holds the run (the run "
            "tag on CHPC).",
        ),
        click.option(
            "--layers-dir",
            default=None,
            type=click.Path(exists=True, file_okay=False, path_type=Path),
            help="Folder with pulled cell layers (<unit>/<tag>/). Default: "
            "<export-dir>/run_layers.",
        ),
        click.option(
            "--config",
            "config_path",
            default=None,
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            help="YAML with values that replace those of the packaged "
            "label.yaml (only the keys it names).",
        ),
    ]
    for option in reversed(options):
        func = option(func)
    return func


def _config(config_path: Path | None):
    """The packaged thresholds, with the values of ``config_path`` on top."""
    import yaml

    from csdv_core.config import load_label, load_yaml
    from csdv_core.config._models import LabelConfig

    if config_path is None:
        return load_label()
    merged = load_yaml("label")
    extra = yaml.safe_load(config_path.read_text()) or {}
    for section, values in extra.items():
        if isinstance(values, dict) and isinstance(merged.get(section), dict):
            merged[section] = {**merged[section], **values}
        else:
            merged[section] = values
    return LabelConfig.model_validate(merged)


def _run(
    label_dir, export_dir, run_tag, runs_dir, run_subdir, layers_dir, config_path=None
):
    from csdv_core.label.pipeline import LabelRun
    from csdv_core.review.config import default_export_dir

    if runs_dir is None:
        runs_dir = (export_dir or default_export_dir()) / "runs" / run_tag
    label_dir.mkdir(parents=True, exist_ok=True)
    return LabelRun(
        root=label_dir,
        runs_dir=runs_dir,
        run_tag=run_tag,
        run_subdir=run_subdir,
        layers_dir=layers_dir,
        cfg=_config(config_path),
    )


def _units(run, units: str | None) -> list[str]:
    if units:
        return [u.strip() for u in units.split(",") if u.strip()]
    found = []
    for folder in sorted(p for p in run.runs_dir.glob("*") if p.is_dir()):
        target = folder / run.run_subdir if run.run_subdir else folder
        if (target / "run_params.json").is_file():
            found.append(folder.name)
    return found


def _selected(run, windows: str | None, unit: str | None, stage: int | None = None):
    table = run.windows()
    if unit:
        table = table[table["unit"] == unit]
    if windows:
        table = table[table["window_id"].isin(windows.split(","))]
    if stage is not None:
        if "stage" not in table:
            raise click.ClickException(
                "windows.csv has no stage column. It was drawn before stages "
                "existed. Select windows with --windows instead."
            )
        table = table[table["stage"].astype(int) == stage]
    return table


def _stage_option(func):
    return click.option(
        "--stage",
        type=int,
        default=None,
        help="Only the windows of this stage (column `stage` of windows.csv). "
        "Stage 1 is a small random part of the sample with the reviewed "
        "windows, to be judged and looked at before the rest.",
    )(func)


#: What a label folder holds once it has been rendered or judged.
_RENDERED = ("stacks", "requests", "render_done", "answers.jsonl")


@click.group()
def label() -> None:
    """Build training labels for a NAIP disturbance model.

    See docs/guides/disturbance_labels.md.
    """


@label.command("candidates")
@_common
@click.option("--units", default=None, help="Comma-separated unit IDs. Default: all.")
def candidates_cmd(units: str | None, **where) -> None:
    """Build the candidate set of every unit and report its size."""
    import pandas as pd

    _setup_logging()
    run = _run(**where)
    rows = []
    for unit_id in _units(run, units):
        frame = run.candidates(unit_id)
        big = frame[frame["area_ha"] >= run.cfg.candidates.min_ha]
        row = {"unit": unit_id, "below_min": int(len(frame) - len(big))}
        for source in ("di", "weak", "chm"):
            part = big[big["source"] == source]
            row[f"{source}_n"] = len(part)
            row[f"{source}_ha"] = round(float(part["area_ha"].sum()), 1)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(run.root / "candidates" / "yield.csv", index=False)
    click.echo(table.to_string(index=False))


@label.command("sample")
@_common
@click.option("--units", default=None, help="Comma-separated unit IDs. Default: all.")
@click.option("--n-di", type=int, default=650, show_default=True)
@click.option("--n-other", type=int, default=150, show_default=True)
@click.option("--n-quiet", type=int, default=150, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--quiet-max-per-unit",
    type=int,
    default=25,
    show_default=True,
    help="Most quiet windows in one unit. Units with few quiet forest chips "
    "pass their share to the others, up to this number.",
)
@click.option(
    "--stage1-di",
    type=int,
    default=4,
    show_default=True,
    help="Windows per unit, seeded by a detected object, in stage 1.",
)
@click.option(
    "--stage1-other",
    type=int,
    default=2,
    show_default=True,
    help="Windows per unit, seeded by another object, in stage 1.",
)
@click.option(
    "--reviewed",
    "review_store",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Review GeoPackage of `csdv review`. Every interpreted polygon gets "
    "a window of its own, as a fixed reference inside the run.",
)
@click.option("--dry-run", is_flag=True, help="Report the sample, write nothing.")
@click.option("--overwrite", is_flag=True, help="Replace an existing windows.csv.")
def sample_cmd(
    units,
    n_di,
    n_other,
    n_quiet,
    seed,
    quiet_max_per_unit,
    stage1_di,
    stage1_other,
    review_store,
    dry_run,
    overwrite,
    **where,
) -> None:
    """Draw the sample of windows and forecast the cost of judging it.

    Window IDs follow the order of the sample, so a sample is drawn once per
    label folder. A folder that has been rendered or judged is not drawn
    again, with or without --overwrite.
    """
    from csdv_core.label.pipeline import window_parts
    from csdv_core.label.sample import SampleDesign, draw_windows

    _setup_logging()
    run = _run(**where)
    path = run.root / "windows.csv"
    if path.exists() and not (dry_run or overwrite):
        raise click.ClickException(f"{path} exists. Pass --overwrite to replace it.")
    used = [name for name in _RENDERED if (run.root / name).exists()]
    if used and not dry_run:
        raise click.ClickException(
            f"{run.root} already holds {', '.join(used)}. A new sample would "
            "give other windows the same IDs. Draw it into a new label folder."
        )
    design = SampleDesign(
        n_di=n_di,
        n_other=n_other,
        n_quiet=n_quiet,
        seed=seed,
        quiet_max_per_unit=quiet_max_per_unit,
        stage1_di=stage1_di,
        stage1_other=stage1_other,
    )
    reviewed = None
    if review_store is not None:
        import geopandas as gpd

        done = set(gpd.read_file(review_store, layer="interpretations")["sample_id"])
        sample = gpd.read_file(review_store, layer="sample", ignore_geometry=True)
        sample = sample[sample["sample_id"].isin(done)]
        reviewed = [
            (str(u), int(o))
            for u, o in zip(sample["unit"], sample["object_id"], strict=True)
        ]
    table = draw_windows(run, _units(run, units), design, reviewed)
    judged = []
    for row in table.itertuples():
        parts = window_parts(run, row, with_stack=False)
        judged.append(int(parts.candidates["judged"].sum()))
    table["judged"] = judged
    table["chips"] = table["nx"] * table["ny"]
    click.echo(
        table.groupby("stratum")[["chips", "judged"]]
        .agg(["count", "sum", "mean"])
        .round(2)
        .to_string()
    )
    click.echo(
        table.pivot_table(
            index="unit",
            columns="stratum",
            values="chips",
            aggfunc="count",
            fill_value=0,
        ).to_string()
    )
    calls = 2 * int(table["judged"].sum())
    click.echo(
        f"{len(table)} windows, {int(table['chips'].sum())} chips, "
        f"{int(table['judged'].sum())} judged candidates "
        f"({table['judged'].mean():.2f} per window). Two verdicts each: {calls} "
        f"calls, about ${calls * CALL_COST:.0f} at list price."
    )
    for stage, part in table.groupby("stage"):
        n = int(part["judged"].sum())
        by = part["stratum"].value_counts().to_dict()
        click.echo(
            f"Stage {stage}: {len(part)} windows {by}, {n} judged candidates, "
            f"about ${2 * n * CALL_COST:.2f}."
        )
    if not dry_run:
        table.to_csv(path, index=False)
        _write_units(run)
        click.echo(f"Wrote {path}")


def _write_units(run) -> int:
    """Write units.csv, the list of units that have a window."""
    units = sorted(set(run.windows()["unit"]))
    (run.root / "units.csv").write_text("unit\n" + "".join(f"{u}\n" for u in units))
    return len(units)


@label.command("units")
@_common
def units_cmd(**where) -> None:
    """Write units.csv from windows.csv (after editing the windows by hand)."""
    run = _run(**where)
    click.echo(f"{_write_units(run)} units in {run.root / 'units.csv'}")


@label.command("render")
@_common
@click.option("--unit", default=None, help="Render the windows of one unit.")
@click.option("--windows", default=None, help="Comma-separated window IDs.")
@click.option(
    "--kind",
    type=click.Choice(["verdict", "polygon"]),
    default="verdict",
    show_default=True,
)
@click.option(
    "--naip-source",
    type=click.Choice(["local", "remote"]),
    default="remote",
    show_default=True,
    help="local: the unit's mosaics under the data root (CHPC). "
    "remote: Planetary Computer and the USDA service.",
)
@click.option(
    "--data-root",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Root of the local mosaics. Default: $CSDV_DATA_ROOT.",
)
@click.option(
    "--chips/--no-chips",
    "with_chips",
    default=False,
    help="Also write the four-band chips of every window.",
)
@click.option(
    "--check",
    is_flag=True,
    help="List the mosaics each window would read, render nothing.",
)
def render_cmd(unit, windows, kind, naip_source, data_root, with_chips, check, **where):
    """Fetch the imagery of each window and write the request images."""
    import json

    from csdv_core.io.paths import project_paths
    from csdv_core.label.chips import write_naip_chips
    from csdv_core.label.naip import local_mosaics
    from csdv_core.label.pipeline import render_window, stack_complete
    from csdv_core.label.stacks import load_stack

    _setup_logging()
    run = _run(**where)
    table = _selected(run, windows, unit)
    source = "local" if naip_source == "local" else None
    root = data_root or project_paths().data_root
    if check:
        missing, undated = 0, 0
        for unit_id in sorted(set(table["unit"])):
            for year in run.unit(unit_id).years:
                found = local_mosaics(root, unit_id, year)
                missing += not found
                note = ""
                if found and not found[0].with_suffix(".provenance.json").exists():
                    undated += 1
                    note = "  (no .provenance.json: windows take the unit's date)"
                click.echo(
                    f"{unit_id} {year}: {found[0] if found else 'MISSING'}{note}"
                )
        click.echo(
            f"{len(table)} windows, {missing} unit-years without a mosaic, "
            f"{undated} without a sidecar"
        )
        if missing:
            raise click.ClickException("Some mosaics are missing.")
        return
    if kind == "polygon":
        # Polygon requests follow the verdicts. Without them every window
        # would get an empty request list, and nothing would say why.
        from csdv_core.label import judge

        wanted = set(table["window_id"].astype(str))
        verdicts = sum(
            rec["kind"] == "verdict" and rec["window_id"] in wanted
            for rec in judge.read_records(run.answers)
        )
        if not verdicts:
            raise click.ClickException(
                f"{run.answers} holds no verdict for these windows. Polygon "
                "images are rendered from the verdicts: copy answers.jsonl "
                "into the label folder first (scripts/chpc/push_label_run.sh)."
            )
        click.echo(f"{verdicts} verdicts of these windows in {run.answers}")
    total = 0
    skipped: list[str] = []
    chips_written: set[str] = set()
    for n, row in enumerate(table.itertuples(), start=1):
        count = render_window(run, row, kind=kind, source=source, data_root=root)
        total += count
        if not stack_complete(run, row):
            # render_window has logged which years lack imagery.
            skipped.append(str(row.window_id))
            continue
        if with_chips:
            # A chip in two windows is written from the first, as `build` does.
            window = run.window(row)
            mine = {c.chip_id for c in window.chips()} - chips_written
            write_naip_chips(
                window,
                load_stack(run.root / "stacks" / window.window_id),
                run.root / "chips",
                only=mine,
            )
            chips_written |= mine
        logging.getLogger(__name__).info(
            "%d/%d %s: %d %s requests", n, len(table), row.window_id, count, kind
        )
    click.echo(
        f"{total} {kind} requests in {len(table) - len(skipped)} of "
        f"{len(table)} windows"
    )
    rules: dict[str, int] = {}
    mixed = 0
    for row in table.itertuples():
        meta = run.root / "stacks" / str(row.window_id) / "meta.json"
        if not meta.exists():
            continue
        for info in json.loads(meta.read_text())["years"].values():
            rule = info.get("date_rule") or "unknown"
            rules[rule] = rules.get(rule, 0) + 1
            mixed += bool(info.get("mixed_dates"))
    click.echo(
        "Dates of the window-years: "
        + ", ".join(f"{n} by {rule}" for rule, n in sorted(rules.items()))
        + f". {mixed} hold two flight dates."
    )
    if rules.get("mosaic_name"):
        click.echo(
            "Dates by mosaic_name are the unit's median date, which can be "
            "months from the window's own. Their mosaics lack a readable "
            ".provenance.json."
        )
    if skipped:
        click.echo(
            f"Skipped, imagery incomplete ({len(skipped)}): {', '.join(skipped)}. "
            "The reason for each is in stacks/<window>/meta.json under 'failed'."
        )


@label.command("verdict")
@_common
@click.option("--windows", default=None, help="Comma-separated window IDs.")
@click.option(
    "--kind",
    type=click.Choice(["verdict", "polygon"]),
    default="verdict",
    show_default=True,
)
@click.option(
    "--cap",
    type=float,
    required=True,
    help="Spending cap of this label run, US dollars. No default.",
)
@click.option("--workers", type=int, default=6, show_default=True)
@click.option("--dry-run", is_flag=True, help="Count the calls, send nothing.")
@_stage_option
def verdict_cmd(windows, kind, cap, workers, dry_run, stage, **where) -> None:
    """Send the rendered requests to the models (needs the API keys)."""
    from csdv_core.label import judge as J
    from csdv_core.label.clients import estimate_cost
    from csdv_core.label.ledger import Ledger
    from csdv_core.label.pipeline import load_requests

    _setup_logging()
    run = _run(**where)
    ledger = Ledger(run.root, cap=cap)
    models = run.models if kind == "verdict" else (run.models[1],)
    name = "requests.jsonl" if kind == "verdict" else "polygons.jsonl"
    done = {J.record_key(r) for r in J.read_records(run.answers) if not r["error"]}
    calls, upper = 0, 0.0
    table = _selected(run, windows, None, stage)
    unrendered = [
        str(w)
        for w in table["window_id"]
        if not (run.root / "requests" / str(w) / name).exists()
    ]
    for row in table.itertuples():
        requests = load_requests(run.root / "requests" / row.window_id, name)
        for model in models:
            todo = [
                q
                for q in requests
                if J.record_key(
                    {
                        "window_id": row.window_id,
                        "cand_id": q.cand_id,
                        "kind": kind,
                        "model": model,
                        "extra": q.extra,
                    }
                )
                not in done
            ]
            calls += len(todo)
            upper += sum(estimate_cost(model, len(q.images)) for q in todo)
            if dry_run or not todo:
                continue
            J.append_records(
                run.answers,
                J.send(
                    ledger,
                    todo,
                    model,
                    window_id=row.window_id,
                    stage=kind,
                    workers=workers,
                ),
            )
    click.echo(
        f"{calls} new {kind} calls, at most ${upper:.2f}, about "
        f"${calls * CALL_COST:.2f} expected. Spent so far: ${ledger.spent():.2f} "
        f"of ${cap:.2f}."
    )
    if unrendered:
        click.echo(
            f"{len(unrendered)} of {len(table)} selected windows have no {name} "
            f"(not rendered, or not copied here) and are not counted: "
            f"{', '.join(unrendered[:10])}{' ...' if len(unrendered) > 10 else ''}"
        )


@label.command("build")
@_common
@click.option("--windows", default=None, help="Comma-separated window IDs.")
@click.option(
    "--imagery/--no-imagery",
    "with_imagery",
    default=True,
    help="Use the window stacks (for missing-imagery pixels and the "
    "four-band chips). Without them only the label rasters are written.",
)
@_stage_option
def build_cmd(windows, with_imagery, stage, **where) -> None:
    """Route the answers and write the label rasters and the vector labels.

    The tables and labels.gpkg are written anew for the selected windows, so
    the last build of a run should select all of them.
    """
    import pandas as pd

    from csdv_core.label import diagnostics as D
    from csdv_core.label import export as E
    from csdv_core.label import judge as J
    from csdv_core.label.chips import write_label_chips, write_naip_chips
    from csdv_core.label.pipeline import build_window, window_dates

    _setup_logging()
    run = _run(**where)
    records = J.read_records(run.answers)
    chips: list[dict] = []
    cands: list[dict] = []
    events: list[dict] = []
    reviewed: list[dict] = []
    acquired: list[dict] = []
    intervals: list[dict] = []
    kept: list[dict] = []
    suspects: list[dict] = []
    written: set[str] = set()
    shared = 0
    for row in _selected(run, windows, None, stage).itertuples():
        built = build_window(run, row, records, with_stack=with_imagery)
        if built is None:
            continue
        window, stack = built.parts.window, built.parts.stack
        dates = (
            stack.dates
            if stack
            else (
                window_dates(run, window.window_id)
                or [built.parts.unit.dates[y] for y in built.parts.unit.years]
            )
        )
        # Two windows can hold the same chip of the lattice. The first window
        # in the order of windows.csv keeps it, so a chip has one set of
        # label rasters and one row.
        rows = write_label_chips(
            window,
            built.labels,
            run.root / "chips",
            dates=dates,
            native_m=stack.native_m if stack else None,
            skip=written,
        )
        shared += len(window.chips()) - len(rows)
        written.update(r["chip_id"] for r in rows)
        chips += rows
        if stack is not None:
            write_naip_chips(
                window, stack, run.root / "chips", only={r["chip_id"] for r in rows}
            )
            intervals += D.interval_rows(stack, window, built.labels, run.cfg)
            suspects += D.suspect_rows(stack, window, built.labels)
        kept += D.event_rows(window, built.labels, built.judged, run.cfg)
        cands += E.candidate_rows(built)
        events += E.event_rows(built, run)
        reviewed += E.reviewed_rows(built)
        acquired += E.acquisition_rows(run, window.window_id, window.unit)
    out = run.root / "labels"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(chips).to_parquet(run.root / "chips" / "chips.parquet", index=False)
    pd.DataFrame(intervals).to_csv(out / "intervals.csv", index=False)
    pd.DataFrame(kept).to_csv(out / "events_kept.csv", index=False)
    pd.DataFrame(suspects).to_csv(out / "suspects.csv", index=False)
    E.write_layers(
        out / "labels.gpkg",
        candidates=cands,
        events=events,
        reviewed=reviewed,
        acquisitions=acquired,
    )
    status = pd.Series([c["status"] for c in cands]).value_counts().to_dict()
    click.echo(f"{len(chips)} chips, {len(events)} events. Candidates: {status}")
    if shared:
        click.echo(
            f"{shared} chips lie in two windows and keep the labels of the first."
        )


@label.command("sheets")
@_common
@click.option("--windows", default=None, help="Comma-separated window IDs.")
@click.option(
    "--max-pairs",
    type=int,
    default=4,
    show_default=True,
    help="Pair sheets per window.",
)
@_stage_option
def sheets_cmd(windows, max_pairs, stage, **where) -> None:
    """Draw the overview and pair sheets of windows (needs their stacks)."""
    from csdv_core.label import judge as J
    from csdv_core.label import sheets as S
    from csdv_core.label.pipeline import build_window
    from csdv_core.label.requests import LOCATIONS

    _setup_logging()
    run = _run(**where)
    records = J.read_records(run.answers)
    out = run.root / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    for row in _selected(run, windows, None, stage).itertuples():
        built = build_window(run, row, records)
        if built is None:
            continue
        unit, window, stack = built.parts.unit, built.parts.window, built.parts.stack
        note = getattr(row, "note", "") or getattr(row, "stratum", "")
        title = (
            f"{window.window_id}   {unit.unit} {unit.label}   "
            f"{LOCATIONS.get(unit.unit, '')}   [{note}]"
        )
        view = S.WindowView(
            window.window_id,
            title,
            window.bounds,
            stack,
            built.labels,
            built.judged,
            single_loss=2 not in run.cfg.policy.values(),
        )
        S.overview_sheet(view).save(out / f"{window.window_id}_overview.png")
        for chip, k, loss_px, ignore in S.choose_pairs(built.labels, window, max_pairs):
            notes = [S.pair_note(stack, built.labels, k, loss_px, ignore)]
            name = (
                f"{window.window_id}_pair_{chip.chip_id}_{stack.dates[k + 1][:4]}.png"
            )
            S.pair_sheet(
                view, chip.bounds, chip.chip_id, k, run.cfg.pooling, notes
            ).save(out / name)
    click.echo(f"Sheets in {out}")


#: A pixel darker than this in every band of a request image has no imagery.
_EMPTY_LEVEL = 12


@label.command("compare-renders")
@click.argument("first", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("second", type=click.Path(exists=True, file_okay=False, path_type=Path))
def compare_renders(first: Path, second: Path) -> None:
    """Compare the request images of two renders of the same windows.

    FIRST and SECOND are ``requests`` folders, for example one rendered from
    Planetary Computer and one from the mosaics on CHPC. Images are matched
    by window folder and file name.
    """
    import numpy as np
    from PIL import Image

    rows = []
    only_first = 0
    for path in sorted(first.rglob("*.jpg")):
        other = second / path.relative_to(first)
        if not other.exists():
            only_first += 1
            continue
        name = str(path.relative_to(first))
        a = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
        b = np.asarray(Image.open(other).convert("RGB"), dtype=np.int16)
        if a.shape != b.shape:
            rows.append((name, float("nan"), float("nan")))
            continue
        # A window on the edge of its unit is black beyond the unit's mosaic
        # in a local render and filled from the next quad in a remote one.
        # The two are compared where both have imagery.
        empty = (a.max(axis=2) < _EMPTY_LEVEL) | (b.max(axis=2) < _EMPTY_LEVEL)
        diff = np.abs(a - b).mean(axis=2)
        both = float(diff[~empty].mean()) if (~empty).any() else float("nan")
        rows.append((name, both, float(empty.mean())))
    if not rows:
        raise click.ClickException("No request image is in both folders.")
    diffs = np.array([d for _, d, _ in rows])
    gaps = np.array([g for _, _, g in rows])
    worst = sorted(rows, key=lambda r: -np.nan_to_num(r[1], nan=1e9))[:5]
    click.echo(
        f"{len(rows)} images in both, {only_first} only in the first. Mean "
        f"absolute difference per image where both have imagery: median "
        f"{np.nanmedian(diffs):.2f}, 90th percentile "
        f"{np.nanpercentile(diffs, 90):.2f} (grey levels of 255). Different "
        f"size: {int(np.isnan(gaps).sum())}."
    )
    for name, diff, _ in worst:
        click.echo(f"  {diff:6.2f}  {name}")
    partial = sorted((r for r in rows if r[2] > 0.05), key=lambda r: -r[2])
    if partial:
        windows = sorted({Path(name).parts[0] for name, _, _ in partial})
        click.echo(
            f"{len(partial)} images lack imagery over more than 5% of the view in "
            f"one of the renders (windows: {', '.join(windows)}). Largest share:"
        )
        for name, _, gap in partial[:5]:
            click.echo(f"  {gap:6.0%}  {name}")
