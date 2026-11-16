"""csdv_core.audit.cli: ``csdv audit``: draw, build, app, bundle, score."""

from __future__ import annotations

import getpass
import logging
from pathlib import Path

import click

__all__ = ["audit"]

log = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s"
    )


_label_opt = click.option(
    "--label-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Folder of the label run the audit checks.",
)
_audit_opt = click.option(
    "--audit-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Folder of the audit (items.csv, site/, answers/).",
)


@click.group()
def audit() -> None:
    """Blind audit of a label run by human interpreters.

    See docs/guides/label_audit.md.
    """


@audit.command("draw")
@_label_opt
@_audit_opt
@click.option(
    "--exclude-windows",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="File of window IDs, one per line, that the interpreter has already "
    "seen with the model answers.",
)
@click.option("--n-accepted", type=int, default=60, show_default=True)
@click.option("--n-rejected", type=int, default=40, show_default=True)
@click.option("--n-disagree", type=int, default=20, show_default=True)
@click.option("--n-windows", type=int, default=60, show_default=True)
@click.option("--reserve", type=float, default=0.5, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--extend",
    is_flag=True,
    help="Add the items of a larger draw to an existing items.csv. Use the "
    "same seed and the same excluded windows.",
)
def draw_cmd(
    label_dir: Path,
    audit_dir: Path,
    exclude_windows: Path | None,
    n_accepted: int,
    n_rejected: int,
    n_disagree: int,
    n_windows: int,
    reserve: float,
    seed: int,
    extend: bool,
) -> None:
    """Draw the audit items once and write items.csv."""
    from csdv_core.audit.draw import AuditDesign, draw_items, extend_items, read_items

    _setup_logging()
    path = audit_dir / "items.csv"
    if path.exists() and not extend:
        raise click.UsageError(
            f"{path} exists. A sample is drawn once. Use --extend to add items."
        )
    exclude = set()
    if exclude_windows is not None:
        exclude = set(exclude_windows.read_text().split())
    design = AuditDesign(
        n_accepted=n_accepted,
        n_rejected=n_rejected,
        n_disagree=n_disagree,
        n_windows=n_windows,
        reserve=reserve,
        seed=seed,
    )
    items = draw_items(label_dir, design, exclude)
    if extend and path.exists():
        items = extend_items(read_items(path), items)
    audit_dir.mkdir(parents=True, exist_ok=True)
    items.to_csv(path, index=False)
    table = items.groupby(["block", "stratum", "reserve"]).size()
    click.echo(table.to_string())
    click.echo(f"{len(items)} items written to {path}")


@audit.command("build")
@_label_opt
@_audit_opt
@click.option("--name", default="audit", show_default=True)
@click.option(
    "--copy",
    "copy",
    is_flag=True,
    help="Copy the request images instead of hard-linking them.",
)
@click.option("--refresh-labels", is_flag=True, help="Draw the label overlays again.")
def build_cmd(
    label_dir: Path, audit_dir: Path, name: str, copy: bool, refresh_labels: bool
) -> None:
    """Write the page and the images of every item that can be built.

    Window items need the stacks of their windows. The windows still missing
    are listed in stack_windows.txt (target items) and
    stack_windows_reserve.txt, ready for
    ``scripts/chpc/pull_label_run.sh NAME --no-requests --stacks-from``.
    """
    from csdv_core.audit.build import build_site
    from csdv_core.audit.draw import read_items

    _setup_logging()
    items = read_items(audit_dir / "items.csv")
    report = build_site(
        label_dir,
        audit_dir,
        items,
        name=name,
        link=not copy,
        refresh_labels=refresh_labels,
    )
    click.echo(f"{len(report.built)} items built in {audit_dir / 'site'}")
    for name, windows in (
        ("stack_windows.txt", report.missing_stacks),
        ("stack_windows_reserve.txt", report.missing_reserve),
    ):
        path = audit_dir / name
        path.write_text("".join(f"{w}\n" for w in windows))
        if windows:
            click.echo(f"{len(windows)} windows have no imagery yet: {path}")
    for item, reason in report.skipped.items():
        click.echo(f"left out {item}: {reason}")


@audit.command("app")
@_audit_opt
@click.option(
    "--interpreter",
    default=None,
    help="Name stored with every answer. Default: the login name.",
)
@click.option("--host", default="localhost", show_default=True)
@click.option("--port", type=int, default=8766, show_default=True)
def app_cmd(audit_dir: Path, interpreter: str | None, host: str, port: int) -> None:
    """Serve the audit page and store the answers of one interpreter."""
    from csdv_core.audit.server import make_server

    _setup_logging()
    site = audit_dir / "site"
    if not (site / "items.js").exists():
        raise click.UsageError(f"{site} is not built. Run `csdv audit build` first.")
    name = (interpreter or getpass.getuser()).strip()
    answers = audit_dir / "answers" / f"{name}.jsonl"
    server = make_server(site, answers, name, host, port)
    click.echo(f"Audit page: http://{host}:{server.server_address[1]}/")
    click.echo(f"Answers of {name}: {answers}")
    click.echo("Ctrl+C stops the server. Answers are stored as you go.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("Stopped.")
    finally:
        server.server_close()


@audit.command("bundle")
@_audit_opt
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Folder to write and share. Opened from disk, no install needed.",
)
@click.option(
    "--all-items",
    is_flag=True,
    help="Every built item. Default: the items every interpreter judges.",
)
def bundle_cmd(audit_dir: Path, out_dir: Path, all_items: bool) -> None:
    """Copy the page and the shared items into a folder for other interpreters.

    They open index.html in a browser, judge with the keyboard, press
    "Download answers" and send the file back. Put it in answers/ and run
    ``csdv audit score``.
    """
    from csdv_core.audit.build import bundle_site
    from csdv_core.audit.draw import read_items

    _setup_logging()
    items = read_items(audit_dir / "items.csv")
    try:
        missing = bundle_site(audit_dir, out_dir, items, shared_only=not all_items)
    except FileExistsError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Bundle written to {out_dir}")
    if missing:
        click.echo(
            f"{len(missing)} items are not built yet and are not in the bundle "
            f"({missing[0]} to {missing[-1]}). Pull their stacks, build, and "
            "bundle into a new folder."
        )


@audit.command("score")
@_audit_opt
@click.option(
    "--answers",
    "answer_paths",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Answers files. Default: every answers/*.jsonl of the audit.",
)
def score_cmd(audit_dir: Path, answer_paths: tuple[Path, ...]) -> None:
    """Rates per stratum and agreement between interpreters.

    Writes score/judged.csv (every answer beside its key), score/rates.csv
    and score/agreement.csv. Do not open them before your own audit is done:
    they hold the key.
    """
    from csdv_core.audit import score as S
    from csdv_core.audit.draw import read_items

    _setup_logging()
    paths = list(answer_paths) or sorted((audit_dir / "answers").glob("*.jsonl"))
    if not paths:
        raise click.ClickException(f"No answers in {audit_dir / 'answers'}")
    items = read_items(audit_dir / "items.csv")
    judged = S.judged_table(items, S.load_answers(paths))
    if judged.empty:
        raise click.ClickException("No answer matches an item of this audit.")
    out = audit_dir / "score"
    out.mkdir(exist_ok=True)
    rates = S.rates(judged)
    agreement = S.agreement(judged)
    judged.to_csv(out / "judged.csv", index=False)
    rates.to_csv(out / "rates.csv", index=False)
    agreement.to_csv(out / "agreement.csv", index=False)
    click.echo(rates.round(3).to_string(index=False))
    if not agreement.empty:
        agreed = int(agreement["all_agree"].sum())
        click.echo(
            f"{len(agreement)} items judged by more than one interpreter, "
            f"{agreed} with the same answer from all."
        )
    click.echo(f"Tables: {out}")


# --------------------------------------------------------------------------
# Fifth stratum: patches only the model flags (block 3)
# --------------------------------------------------------------------------

_stratum_opt = click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Folder of the stratum (patches.gpkg, requests/). On CHPC: "
    "$CSDV_EXPORT_DIR/audit/model_only.",
)


def _batch_units() -> list[str]:
    """Units with ``batch`` set to yes in config/pair/sites.csv."""
    import pandas as pd

    from csdv_core.io.paths import _repo_root

    path = _repo_root() / "config" / "pair" / "sites.csv"
    sites = pd.read_csv(path, comment="#")
    return sites.loc[sites["batch"].astype(str).eq("yes"), "unit"].astype(str).tolist()


@audit.group("model-only")
def model_only() -> None:
    """The fifth stratum: patches the model flags that no candidate touches.

    Drawn from maps of units the model never trained on (the holdout runs),
    rendered like block 1, and added to the audit as block 3. See
    docs/guides/label_audit.md.
    """


@model_only.command("draw")
@_label_opt
@_stratum_opt
@click.option(
    "--pair-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Folder of the pair-model runs. Default: $CSDV_EXPORT_DIR/pairmodel.",
)
@click.option(
    "--units",
    default=None,
    help="Comma-separated units. Default: " "the batch units of config/pair/sites.csv.",
)
@click.option(
    "--run-pattern",
    default="holdout_{unit}",
    show_default=True,
    help="Run folder of each unit.",
)
@click.option(
    "--fallback-run",
    default="first_look_blocks2",
    show_default=True,
    help="Run used for a unit without its own run folder, if it mapped the "
    "unit (the run that held Oregon out).",
)
@click.option(
    "--exclude-shown/--keep-shown",
    default=True,
    show_default=True,
    help="Leave out patches that have a close-up in the run's compare/events/.",
)
@click.option("--n-per-unit", type=int, default=2, show_default=True)
@click.option("--reserve", type=int, default=1, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--allow-missing",
    is_flag=True,
    help="Draw although some units have no map. The stratum is drawn once, "
    "so those units can never join it.",
)
def model_only_draw(
    label_dir: Path,
    out_dir: Path,
    pair_dir: Path | None,
    units: str | None,
    run_pattern: str,
    fallback_run: str,
    exclude_shown: bool,
    n_per_unit: int,
    reserve: int,
    seed: int,
    allow_missing: bool,
) -> None:
    """Draw the model-only patches once and write patches.gpkg."""
    import geopandas as gpd
    import pandas as pd

    from csdv_core.audit import model_only as M
    from csdv_core.review.config import default_export_dir

    _setup_logging()
    path = out_dir / "patches.gpkg"
    if path.exists():
        raise click.UsageError(f"{path} exists. The stratum is drawn once.")
    pair_dir = pair_dir or default_export_dir() / "pairmodel"
    wanted = [u.strip() for u in units.split(",")] if units else _batch_units()
    design = M.ModelOnlyDesign(n_per_unit=n_per_unit, reserve=reserve, seed=seed)
    frames, used, missing = [], {}, []
    for unit in wanted:
        run_dir = pair_dir / run_pattern.format(unit=unit)
        if not (run_dir / "map" / f"prob_forest_{unit}.tif").exists():
            run_dir = pair_dir / fallback_run
        if not (run_dir / "map" / f"prob_forest_{unit}.tif").exists():
            missing.append(unit)
            continue
        cand_path = label_dir / "candidates" / f"{unit}.gpkg"
        if not cand_path.exists():
            raise click.ClickException(f"{cand_path} is missing.")
        shown = M.shown_patches(run_dir, unit) if exclude_shown else set()
        pop = M.patch_population(
            run_dir, unit, gpd.read_file(cand_path), design, exclude=shown
        )
        pop["run"] = run_dir.name
        used[unit] = run_dir.name
        if not pop.empty:
            frames.append(M.to_sample_crs(pop))
    if missing:
        where = ", ".join(missing)
        if not allow_missing:
            raise click.ClickException(
                f"No map for {where} in {run_pattern} or {fallback_run}. Wait "
                "for their holdout runs, or pass --allow-missing."
            )
        click.echo(f"Left out, no map: {where}")
    if not frames:
        raise click.ClickException("No patch in any unit.")
    population = gpd.GeoDataFrame(pd.concat(frames), crs=M.SAMPLE_EPSG)
    drawn = M.draw_patches(population, design)
    out_dir.mkdir(parents=True, exist_ok=True)
    drawn.drop(columns=["key"]).to_file(path, layer="patches")
    population.drop(columns="geometry").to_csv(out_dir / "population.csv", index=False)
    (out_dir / "patches.txt").write_text("".join(f"{p}\n" for p in drawn["patch"]))
    counts = population.groupby("unit").size()
    for unit, run in used.items():
        click.echo(f"{unit}: {int(counts.get(unit, 0))} patches in {run}")
    click.echo(
        f"{int((~drawn['reserve']).sum())} target and {int(drawn['reserve'].sum())} "
        f"reserve patches written to {path}"
    )


@model_only.command("render")
@_label_opt
@_stratum_opt
@click.option(
    "--export-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Export folder with runs/<tag>/<unit>/. Default: $CSDV_EXPORT_DIR.",
)
@click.option("--run-tag", default="default", show_default=True)
@click.option(
    "--runs-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Folder with one disturbance run folder per unit. On CHPC: "
    "$CSDV_RESULTS_ROOT/disturbance, with --run-subdir.",
)
@click.option("--run-subdir", default="", help="Sub-folder of each unit folder.")
@click.option(
    "--naip-source",
    type=click.Choice(["local", "remote"]),
    default="local",
    show_default=True,
)
@click.option(
    "--data-root",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Root of the local mosaics. Default: $CSDV_DATA_ROOT.",
)
@click.option("--limit", type=int, default=None, help="Render the first N only.")
@click.option("--keep-stacks", is_flag=True, help="Keep the yearly GeoTIFFs.")
def model_only_render(
    label_dir: Path,
    out_dir: Path,
    export_dir: Path | None,
    run_tag: str,
    runs_dir: Path | None,
    run_subdir: str,
    naip_source: str,
    data_root: Path | None,
    limit: int | None,
    keep_stacks: bool,
) -> None:
    """Render every date of each drawn patch, outlined as in block 1."""
    import geopandas as gpd

    from csdv_core.audit.model_only import render_patches
    from csdv_core.config import load_label
    from csdv_core.io.paths import project_paths
    from csdv_core.label.pipeline import LabelRun
    from csdv_core.review.config import default_export_dir

    _setup_logging()
    patches = gpd.read_file(out_dir / "patches.gpkg", layer="patches")
    if limit:
        patches = patches.head(limit)
    if runs_dir is None:
        runs_dir = (export_dir or default_export_dir()) / "runs" / run_tag
    run = LabelRun(
        root=label_dir,
        runs_dir=runs_dir,
        run_tag=run_tag,
        run_subdir=run_subdir,
        cfg=load_label(),
    )
    failed = render_patches(
        patches,
        out_dir,
        run,
        source="local" if naip_source == "local" else None,
        data_root=data_root or project_paths().data_root,
        keep_stacks=keep_stacks,
    )
    done = sum(
        (out_dir / "requests" / p / "requests.jsonl").exists() for p in patches["patch"]
    )
    click.echo(f"{done} of {len(patches)} patches rendered in {out_dir / 'requests'}")
    if failed:
        click.echo(f"Imagery incomplete, not rendered: {', '.join(failed)}")


@model_only.command("add")
@_audit_opt
@click.option(
    "--model-only",
    "stratum_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Pulled folder of the stratum (patches.gpkg, requests/).",
)
def model_only_add(audit_dir: Path, stratum_dir: Path) -> None:
    """Add the rendered patches to the audit as block 3.

    Existing items keep their names and order, and items.csv is backed up
    first. The request images are copied into the audit's model_only/
    folder. Then run `csdv audit build`.
    """
    import shutil

    import geopandas as gpd

    from csdv_core.audit.build import MODEL_ONLY_DIR
    from csdv_core.audit.draw import read_items
    from csdv_core.audit.model_only import BLOCK, append_items

    _setup_logging()
    path = audit_dir / "items.csv"
    items = read_items(path)
    patches = gpd.read_file(stratum_dir / "patches.gpkg", layer="patches")
    target = audit_dir / MODEL_ONLY_DIR
    shutil.copytree(stratum_dir / "requests", target / "requests", dirs_exist_ok=True)
    shutil.copy2(stratum_dir / "patches.gpkg", target / "patches.gpkg")
    result = append_items(items, patches, target)
    added = len(result) - len(items)
    if not added:
        click.echo("No new patch to add.")
        return
    backup = path.with_name("items_before_block3.csv")
    if not backup.exists():
        shutil.copy2(path, backup)
    result.to_csv(path, index=False)
    block = result[result["block"] == BLOCK]
    click.echo(
        f"{added} items added as block {BLOCK} "
        f"({int((~block['reserve']).sum())} target, {int(block['reserve'].sum())} "
        f"reserve). Earlier items unchanged: {len(items)}. Backup: {backup.name}"
    )
