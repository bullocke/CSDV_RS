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
