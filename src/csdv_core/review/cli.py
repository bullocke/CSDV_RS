"""csdv_core.review.cli — ``csdv review``: sample, extend, app, summary."""

from __future__ import annotations

import logging
from pathlib import Path

import click

__all__ = ["review"]


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s"
    )


def _export_dir(export_dir: Path | None) -> Path:
    from csdv_core.review.config import default_export_dir

    return export_dir or default_export_dir()


def _store_path(store: Path | None, run_tag: str) -> Path:
    from csdv_core.review.store import default_store_path

    return store or default_store_path(run_tag)


_export_opt = click.option(
    "--export-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Export folder. Default: $CSDV_EXPORT_DIR, else results/chpc_exports.",
)
_tag_opt = click.option("--run-tag", default="default", show_default=True)
_store_opt = click.option(
    "--store",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Review GeoPackage. Default: data/reference/grid_pilot/review_<tag>.gpkg.",
)


@click.group()
def review() -> None:
    """Interpret a sample of detected disturbances in a browser app.

    See docs/guides/disturbance_review.md.
    """


@review.command("sample")
@_export_opt
@_tag_opt
@_store_opt
@click.option("--n-per-site", type=int, default=6, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--exclude-growth",
    is_flag=True,
    help="Leave out growth objects (below the MMU, kept because they touch an "
    "earlier detection).",
)
def sample(
    export_dir: Path | None,
    run_tag: str,
    store: Path | None,
    n_per_site: int,
    seed: int,
    exclude_growth: bool,
) -> None:
    """Draw a new review sample (refuses to overwrite one)."""
    from csdv_core.review.sample import draw_sample
    from csdv_core.review.store import ReviewStore

    _setup_logging()
    path = _store_path(store, run_tag)
    if path.exists():
        raise click.ClickException(f"{path} exists. Use `csdv review extend`.")
    frame = draw_sample(
        _export_dir(export_dir),
        run_tag,
        n_per_site,
        seed=seed,
        exclude_growth=exclude_growth,
    )
    ReviewStore.create(path, frame)
    click.echo(f"{len(frame)} sample polygons from {frame['unit'].nunique()} units")
    click.echo(f"Wrote {path}")


@review.command("extend")
@_export_opt
@_tag_opt
@_store_opt
@click.option(
    "--n-per-site",
    type=int,
    required=True,
    help="Objects per unit after extending (not the number to add).",
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--exclude-growth", is_flag=True)
def extend(
    export_dir: Path | None,
    run_tag: str,
    store: Path | None,
    n_per_site: int,
    seed: int,
    exclude_growth: bool,
) -> None:
    """Grow the sample to N objects per unit, keeping every earlier ID.

    Give the same --seed and --exclude-growth as the first draw, so the
    extended sample is the one a single draw of N per unit would give. Units
    new to the export (for example re-run download failures) join with N
    objects each.
    """
    from csdv_core.review.sample import extend_sample
    from csdv_core.review.store import ReviewStore

    _setup_logging()
    path = _store_path(store, run_tag)
    if not path.exists():
        raise click.ClickException(f"{path} does not exist. Use `csdv review sample`.")
    st = ReviewStore(path)
    before = len(st.sample)
    frame = extend_sample(
        st.sample,
        _export_dir(export_dir),
        run_tag,
        n_per_site,
        seed=seed,
        exclude_growth=exclude_growth,
    )
    if len(frame) == before:
        click.echo("Nothing to add.")
        return
    st.set_sample(frame)
    click.echo(f"Added {len(frame) - before} sample polygons ({len(frame)} in all)")


@review.command("app")
@_export_opt
@_tag_opt
@_store_opt
@click.option("--host", default="localhost", show_default=True)
@click.option("--port", type=int, default=8765, show_default=True)
@click.option("--interpreter", default=None, help="Name saved with each label.")
def app(
    export_dir: Path | None,
    run_tag: str,
    store: Path | None,
    host: str,
    port: int,
    interpreter: str | None,
) -> None:
    """Start the review app; open the printed address in a browser."""
    import getpass

    try:
        import solara  # noqa: F401
    except ImportError as exc:
        raise click.ClickException(
            "The review app needs solara: pip install solara (see environment.yml)"
        ) from exc
    from csdv_core.review.config import ReviewConfig
    from csdv_core.review.server import serve

    _setup_logging()
    cfg = ReviewConfig(
        export_dir=_export_dir(export_dir).resolve(),
        run_tag=run_tag,
        store_path=_store_path(store, run_tag).resolve(),
        interpreter=interpreter or getpass.getuser(),
    )
    click.echo(f"Review app: http://{host}:{port}  (Ctrl+C to stop)")
    serve(cfg, host=host, port=port)


@review.command("summary")
@_tag_opt
@_store_opt
def summary(run_tag: str, store: Path | None) -> None:
    """Score the labelled sample per unit and pooled; write CSV and Markdown."""
    from csdv_core.review.store import ReviewStore
    from csdv_core.review.summary import summarize

    _setup_logging()
    path = _store_path(store, run_tag)
    if not path.exists():
        raise click.ClickException(f"{path} does not exist.")
    try:
        result = summarize(ReviewStore(path))
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    for written in result.write(path.parent, path.stem):
        click.echo(f"Wrote {written}")
    click.echo(result.to_markdown())
