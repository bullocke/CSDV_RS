"""csdv_core.pairmodel.cli: ``csdv pair``, train a pair model and map a unit."""

from __future__ import annotations

import logging
from pathlib import Path

import click

__all__ = ["pair"]


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s"
    )
    for name in ("rasterio", "matplotlib", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _units(text: str | None) -> tuple[str, ...]:
    return tuple(u.strip() for u in (text or "").split(",") if u.strip())


@click.group()
def pair() -> None:
    """Siamese pair model for canopy loss between two NAIP dates."""


@pair.command("train")
@click.option(
    "--chips-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Chip folder of a label run (chips.parquet and the chip rasters).",
)
@click.option(
    "--out-dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Run folder for checkpoints and tables. Not needed with --dry-run.",
)
@click.option("--holdout-units", default="", help="Test units, comma-separated.")
@click.option("--val-units", default="", help="Validation units, comma-separated.")
@click.option("--fold", type=int, default=None, help="Validate on this grouped fold.")
@click.option("--n-folds", type=int, default=5, show_default=True)
@click.option(
    "--split",
    type=click.Choice(["units", "blocks"]),
    default="units",
    show_default=True,
    help="units: validate on whole units. blocks: validate on blocks of ground "
    "inside every unit that is not held out, so every agent is on both sides.",
)
@click.option(
    "--block-chips",
    type=int,
    default=8,
    show_default=True,
    help="Side of a block of the blocks split, in chips.",
)
@click.option(
    "--val-share",
    type=float,
    default=0.2,
    show_default=True,
    help="Share of the blocks split aimed at for validation.",
)
@click.option(
    "--cache-dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Folder for decoded chips, built at the start and read by the "
    "loaders (about 12 MB per chip). Cuts the wait for data.",
)
@click.option(
    "--encoder",
    type=click.Choice(["resnet34", "resnet18"]),
    default="resnet34",
    show_default=True,
)
@click.option(
    "--encoder-weights",
    default=None,
    help="torchvision state dict of the encoder (.pth), or 'none' for a "
    "random start. Compute nodes have no internet, so fetch it beforehand.",
)
@click.option(
    "--cell-m",
    type=click.Choice(["9.6", "4.8", "2.4"]),
    default="9.6",
    show_default=True,
    help="Side of the output cells, metres.",
)
@click.option("--crop-px", type=int, default=384, show_default=True)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--epochs", type=int, default=40, show_default=True)
@click.option("--samples-per-epoch", type=int, default=8000, show_default=True)
@click.option("--lr", type=float, default=2e-4, show_default=True)
@click.option("--workers", type=int, default=7, show_default=True)
@click.option("--reverse-prob", type=float, default=0.2, show_default=True)
@click.option(
    "--agents/--no-agents",
    default=True,
    show_default=True,
    help="Train a second head that names the agent of a loss.",
)
@click.option("--agent-weight", type=float, default=0.5, show_default=True)
@click.option("--patience", type=int, default=8, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--max-val-samples", type=int, default=None)
@click.option("--device", default="auto", show_default=True)
@click.option("--no-resume", is_flag=True, help="Ignore last.pt in the run folder.")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show the split and check the chip files, train nothing.",
)
def train_cmd(
    chips_dir,
    out_dir,
    holdout_units,
    val_units,
    fold,
    n_folds,
    split,
    block_chips,
    val_share,
    cache_dir,
    encoder,
    encoder_weights,
    cell_m,
    crop_px,
    batch_size,
    epochs,
    samples_per_epoch,
    lr,
    workers,
    reverse_prob,
    agents,
    agent_weight,
    patience,
    seed,
    max_val_samples,
    device,
    no_resume,
    dry_run,
) -> None:
    """Train on the chips of a label run, with units held out."""
    from csdv_core.pairmodel import data as D
    from csdv_core.pairmodel import train as T

    _setup_logging()
    if split == "blocks" and (val_units or fold is not None):
        raise click.UsageError(
            "--split blocks chooses its own validation ground: "
            "leave out --val-units and --fold."
        )
    if not dry_run and out_dir is None:
        raise click.UsageError("--out-dir is required to train.")
    if not dry_run and encoder_weights is None:
        raise click.UsageError(
            "--encoder-weights is required: a .pth file, or 'none' for a random start."
        )
    weights = None
    if encoder_weights and encoder_weights.lower() != "none":
        weights = Path(encoder_weights)
    cfg = T.TrainConfig(
        chips_dir=chips_dir,
        out_dir=out_dir or Path("."),
        holdout_units=_units(holdout_units),
        val_units=_units(val_units),
        fold=fold,
        n_folds=n_folds,
        split=split,
        block_chips=block_chips,
        val_share=val_share,
        cache_dir=cache_dir,
        encoder=encoder,
        encoder_weights=weights,
        cell_px=round(float(cell_m) / D.GSD),
        crop_px=crop_px,
        batch_size=batch_size,
        epochs=epochs,
        samples_per_epoch=samples_per_epoch,
        lr=lr,
        workers=workers,
        reverse_prob=reverse_prob,
        agents=agents,
        agent_weight=agent_weight,
        patience=patience,
        seed=seed,
        max_val_samples=max_val_samples,
        device=device,
    )
    if dry_run:
        try:
            planned = T.plan(cfg)
            click.echo(T.describe(planned))
        except (FileNotFoundError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        missing = D.missing_files(chips_dir, sorted(planned.index["chip_id"].unique()))
        if missing:
            raise click.ClickException(
                f"{len(missing)} chip files are missing, for example {missing[0]}. "
                "Build the labels beside the four-band chips first "
                "(scripts/slurm/run_label_build.sbatch)."
            )
        if weights is not None and not weights.is_file():
            raise click.ClickException(f"No encoder weights at {weights}")
        click.echo(T.describe_agents(planned, chips_dir, cfg.cell_px))
        click.echo("Every chip file is there.")
        return
    try:
        result = T.train(cfg, resume=not no_resume)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"{result.epochs_run} epochs. Best epoch {result.best_epoch}: mean average "
        f"precision of the validation units {result.best_score:.3f}, best "
        f"threshold {result.best_threshold:.2f}. Run folder: {result.out_dir}"
    )


def _run_dir(runs_dir: Path, unit: str, run_subdir: str) -> Path:
    path = runs_dir / unit / run_subdir if run_subdir else runs_dir / unit
    if not (path / "run_params.json").is_file():
        raise click.ClickException(f"No disturbance run at {path}")
    return path


_where = [
    click.option("--unit", required=True, help="Grid unit, e.g. q4412328."),
    click.option(
        "--runs-dir",
        required=True,
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        help="Folder with one run folder per unit. On CHPC: "
        "$CSDV_RESULTS_ROOT/disturbance with --run-subdir default. On a laptop: "
        "results/chpc_exports/runs/default.",
    ),
    click.option("--run-subdir", default="", help="Sub-folder that holds the run."),
]


def _with(options):
    def wrap(func):
        for option in reversed(options):
            func = option(func)
        return func

    return wrap


@pair.command("predict")
@click.option(
    "--checkpoint",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@_with(_where)
@click.option(
    "--data-root",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Root that holds naip/<unit>/<year>/. Default: the project's data root.",
)
@click.option(
    "--out-dir", required=True, type=click.Path(file_okay=False, path_type=Path)
)
@click.option("--block-cells", type=int, default=128, show_default=True)
@click.option("--halo-cells", type=int, default=16, show_default=True)
@click.option("--workers", type=int, default=4, show_default=True)
@click.option("--limit-blocks", type=int, default=None, help="Quick run on N blocks.")
@click.option("--device", default="auto", show_default=True)
@click.option(
    "--threshold",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Loss probability from which a cell gets its agent in the type rasters. "
    "Default: the checkpoint's best threshold on validation.",
)
def predict_cmd(
    checkpoint,
    unit,
    runs_dir,
    run_subdir,
    data_root,
    out_dir,
    block_cells,
    halo_cells,
    workers,
    limit_blocks,
    device,
    threshold,
) -> None:
    """Map loss probability for every pair of consecutive dates of a unit."""
    from csdv_core.pairmodel.predict import predict_unit

    _setup_logging()
    if data_root is None:
        from csdv_core.io.paths import project_paths

        data_root = project_paths().data_root
    try:
        result = predict_unit(
            checkpoint,
            _run_dir(runs_dir, unit, run_subdir),
            data_root,
            out_dir,
            block_cells=block_cells,
            halo_cells=halo_cells,
            workers=workers,
            limit_blocks=limit_blocks,
            device=device,
            threshold=threshold,
        )
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"{result.unit}: {len(result.intervals)} intervals, {result.blocks} blocks, "
        f"{result.seconds:.0f} s. {result.prob_path}"
    )
    if result.type_paths:
        click.echo(
            f"Agent of loss at or above {result.threshold:.2f}: "
            f"{len(result.type_paths)} type rasters beside it."
        )
    for note in result.notes:
        click.echo(note)


@pair.command("compare")
@click.option(
    "--prob",
    "prob_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="prob_<unit>.tif of `csdv pair predict`.",
)
@click.option(
    "--chips-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@_with(_where)
@click.option(
    "--checkpoint",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Take the threshold from this checkpoint (best on validation).",
)
@click.option("--threshold", type=float, default=None, help="Default: 0.5.")
@click.option(
    "--data-root",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Root that holds naip/<unit>/<year>/, for the close-ups.",
)
@click.option(
    "--out-dir", required=True, type=click.Path(file_okay=False, path_type=Path)
)
@click.option("--panels", type=int, default=12, show_default=True)
def compare_cmd(
    prob_path,
    chips_dir,
    unit,
    runs_dir,
    run_subdir,
    checkpoint,
    threshold,
    data_root,
    out_dir,
    panels,
) -> None:
    """Score a unit's map against its label chips and set it beside the DI."""
    from csdv_core.pairmodel.compare import compare_unit

    _setup_logging()
    if threshold is None and checkpoint is not None:
        import rasterio  # noqa: F401  (GDAL has to load before torch)
        import torch

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        threshold = float(ckpt.get("best_threshold", 0.5))
    try:
        result = compare_unit(
            prob_path,
            chips_dir,
            _run_dir(runs_dir, unit, run_subdir),
            out_dir,
            threshold=threshold if threshold is not None else 0.5,
            data_root=data_root,
            n_objects=panels,
            n_model_only=max(panels // 2, 1),
            n_di_only=max(panels // 2, 1),
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    model, di = result.summary["model"], result.summary["di"]
    click.echo(
        f"{result.summary['chips']} chips, {result.summary['cells_scored']} cells, "
        f"{result.summary['loss_cells']} of them loss.\n"
        f"Model: average precision {model['average_precision']:.3f}, precision "
        f"{model['precision']:.3f}, recall {model['recall']:.3f} at "
        f"{model['threshold']:.2f}.\n"
        f"DI:    average precision {di['average_precision']:.3f}, precision "
        f"{di['precision']:.3f}, recall {di['recall']:.3f} at 0.50.\n"
        f"{result.summary['model_only_patches']} model-only patches "
        f"({result.summary['model_only_ha']:.1f} ha). {result.panels} close-ups. "
        f"{result.out_dir}"
    )
    for note in result.notes:
        click.echo(note)


@pair.command("crops")
@click.option(
    "--boxes",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="CSV with unit, box, epsg, minx, miny, maxx, maxy, years "
    "(space-separated).",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Folder of the crops. On CHPC: $CSDV_EXPORT_DIR/pair_crops/<name>.",
)
@click.option(
    "--data-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Root of the local mosaics. Default: $CSDV_DATA_ROOT.",
)
def crops_cmd(boxes: Path, out_dir: Path, data_root: Path | None) -> None:
    """0.6 m NAIP crops of chosen boxes, one GeoTIFF per box and year.

    For close-up figures on a machine without the mosaics. Pull the folder
    with scripts/chpc/pull_exports.sh pair_crops/<name>.
    """
    from csdv_core.io.paths import project_paths
    from csdv_core.pairmodel.crops import read_boxes, write_crops

    _setup_logging()
    table = read_boxes(boxes)
    report = write_crops(table, data_root or project_paths().data_root, out_dir)
    click.echo(f"{len(report.written)} crops in {out_dir}")
    if report.missing:
        click.echo(f"No mosaic for: {', '.join(sorted(set(report.missing)))}")
