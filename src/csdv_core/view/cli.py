"""csdv_core.view.cli: the ``csdv view`` command."""

from __future__ import annotations

import logging
from pathlib import Path

import click

__all__ = ["view"]


@click.command("view")
@click.option(
    "--export-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Folder of the CHPC exports. Default: $CSDV_EXPORT_DIR or "
    "<results>/chpc_exports.",
)
@click.option(
    "--notes",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="GeoJSON file of the comment points. Default: "
    "<data>/reference/map_notes/pair_maps.geojson.",
)
@click.option("--host", default="localhost", show_default=True)
@click.option("--port", type=int, default=8767, show_default=True)
@click.option(
    "--open/--no-open",
    "open_browser",
    default=True,
    show_default=True,
    help="Open the page in the default browser.",
)
@click.option("-v", "--verbose", is_flag=True, help="Log at DEBUG level.")
def view(
    export_dir: Path | None,
    notes: Path | None,
    host: str,
    port: int,
    open_browser: bool,
    verbose: bool,
) -> None:
    """Browse the pair-model disturbance maps over NAIP in a browser.

    Shows the date and type of loss of every exported run, two runs side by
    side, and saves the comment points dropped on the map.
    """
    from csdv_core.io.paths import _repo_root
    from csdv_core.review.config import default_export_dir
    from csdv_core.view.notes import default_notes_path
    from csdv_core.view.server import ViewConfig, serve

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    exports = (export_dir or default_export_dir()).resolve()
    if not (exports / "pairmodel").is_dir():
        raise click.ClickException(f"{exports} has no pairmodel folder.")
    cfg = ViewConfig(
        export_dir=exports,
        notes_path=(notes or default_notes_path()).resolve(),
        sites_csv=_repo_root() / "config" / "pair" / "sites.csv",
    )
    url = f"http://{host}:{port}"
    click.echo(f"Map viewer: {url}  (Ctrl+C to stop)")
    click.echo(f"Notes: {cfg.notes_path}")
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    serve(cfg, host=host, port=port)
