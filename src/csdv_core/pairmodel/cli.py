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


def _run_draws(items: tuple[str, ...]) -> dict[str, float]:
    """``("stable_forest=0.15",)`` as ``{"stable_forest": 0.15}``.

    A share of 0 never draws the run's samples without loss, and draws every
    other sample as if the run were not in the chip folder: the control of a
    run with a share, on the same chips, split and band statistics.
    """
    out: dict[str, float] = {}
    for item in items:
        name, sep, share = item.partition("=")
        try:
            value = float(share)
        except ValueError:
            value = -1.0
        if not sep or not name.strip() or not 0 <= value < 1:
            raise click.BadParameter(
                f"{item!r}: expected RUN=SHARE with 0 <= SHARE < 1"
            )
        out[name.strip()] = value
    return out


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
@click.option(
    "--dates",
    "n_dates",
    type=click.Choice(["2", "3"]),
    default="2",
    show_default=True,
    help="3 also feeds the image after each pair, where a chip has one. The "
    "target stays the loss between the first two.",
)
@click.option(
    "--third-drop",
    type=click.FloatRange(0.0, 1.0),
    default=0.3,
    show_default=True,
    help="With --dates 3: share of training samples whose third image is "
    "withheld, so that the model also maps an interval with no later image.",
)
@click.option(
    "--run-draws",
    multiple=True,
    metavar="RUN=SHARE",
    help="Share of draws for the samples without loss of a label run of a merged "
    "chip folder (`csdv pair merge-chips`), e.g. stable_forest=0.15. Repeatable.",
)
@click.option(
    "--exclude-runs",
    multiple=True,
    metavar="RUN",
    help="A label run of a merged chip folder that is never drawn in training. "
    "The split, the validation samples and the band statistics stay those of "
    "the whole folder: the control of a model trained with the run. Repeatable.",
)
@click.option(
    "--hold-role",
    "hold_roles",
    multiple=True,
    metavar="ROLE",
    help="Also hold out the units with this role in the chip folder's "
    "sites.csv, e.g. test. Units that the table ties to a held-out unit "
    "(holdout_with) are always held out. Repeatable.",
)
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
    n_dates,
    third_drop,
    run_draws,
    exclude_runs,
    hold_roles,
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
        n_dates=int(n_dates),
        third_drop=third_drop,
        run_draws=_run_draws(run_draws),
        exclude_runs=tuple(r.strip() for r in exclude_runs if r.strip()),
        hold_roles=tuple(r.strip() for r in hold_roles if r.strip()),
        patience=patience,
        seed=seed,
        max_val_samples=max_val_samples,
        device=device,
    )
    if dry_run:
        try:
            planned = T.plan(cfg)
            click.echo(T.describe(planned))
            draws = T.describe_draws(planned, cfg)
        except (FileNotFoundError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        if draws:
            click.echo(draws)
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
@click.option(
    "--skip-pairs",
    is_flag=True,
    help="Also score every pair of dates that skips one (date k against date "
    "k + 2) and write prob_skip_<unit>.tif. A loss that the skipping pair "
    "shows too has lasted to the next image.",
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
    skip_pairs,
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
            skip_pairs=skip_pairs,
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
    if result.pair_path:
        click.echo(f"From two dates alone: {result.pair_path}")
    if result.skip_path:
        click.echo(f"Pairs that skip a date: {result.skip_path}")
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


@pair.command("score")
@click.option(
    "--checkpoint",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--chips-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Chip folder that holds the chips of the units to score.",
)
@click.option(
    "--out-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option("--units", default="", help="Units to score, comma-separated.")
@click.option(
    "--role",
    "roles",
    multiple=True,
    metavar="ROLE",
    help="Score the units with this role in the chip folder's sites.csv, "
    "e.g. test. Repeatable, and added to --units.",
)
@click.option("--batch-size", type=int, default=16, show_default=True)
@click.option("--workers", type=int, default=4, show_default=True)
@click.option("--device", default="auto", show_default=True)
def score_cmd(
    checkpoint, chips_dir, out_dir, units, roles, batch_size, workers, device
):
    """Score a checkpoint on the label chips of units it did not train on.

    Writes one row per chip and interval with the predicted probabilities
    of its loss and background cells (samples.parquet), scores per unit
    (units.csv) and the pooled scores (score.json). Nothing checks that the
    model did not train on the units: hold them out when training
    (--holdout-units, --hold-role).
    """
    from csdv_core.pairmodel import data as D
    from csdv_core.pairmodel.score import score

    _setup_logging()
    wanted = set(_units(units))
    if roles:
        sites = D.load_sites(chips_dir)
        if sites is None:
            raise click.ClickException(f"No {D.SITES_FILE} in {chips_dir}")
        wanted |= set(D.role_units(sites, roles))
    if not wanted:
        raise click.UsageError("Name the units to score with --units or --role.")
    try:
        result = score(
            checkpoint,
            chips_dir,
            out_dir,
            sorted(wanted),
            batch_size=batch_size,
            workers=workers,
            device=device,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    pooled = result.pooled
    click.echo(
        f"{result.samples} samples of {len(result.units)} units. Average precision "
        f"{pooled['average_precision']:.3f}, recall {pooled['recall']:.3f} and "
        f"precision {pooled['precision']:.3f} at {result.threshold:.2f}, "
        f"{100 * pooled['background_flagged']:.2f}% of background cells flagged. "
        f"Tables in {result.out_dir}"
    )


@pair.command("merge-chips")
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="New label folder for the merged chips (pass it as the label folder of "
    "training). On CHPC: $CSDV_EXPORT_DIR/label/<name>.",
)
@click.option(
    "--run",
    "runs",
    multiple=True,
    required=True,
    metavar="NAME=LABEL_DIR",
    help="A label run and its folder (the one that holds chips/), in order of "
    "precedence. Repeatable.",
)
def merge_chips_cmd(out_dir: Path, runs: tuple[str, ...]) -> None:
    """Merge the chips of several label runs into one folder for training."""
    from csdv_core.pairmodel.merge import merge_chips

    _setup_logging()
    named: dict[str, Path] = {}
    for item in runs:
        name, sep, folder = item.partition("=")
        if not sep or not name.strip() or not folder.strip():
            raise click.BadParameter(f"{item!r}: expected NAME=LABEL_DIR")
        named[name.strip()] = Path(folder.strip())
    try:
        result = merge_chips(out_dir, named)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    for name, n in result.chips.items():
        click.echo(f"{name}: {n} chips ({result.dropped[name]} left out as duplicates)")
    if result.sites:
        click.echo(
            f"sites.csv: {result.sites} units with a role, a target or a tie "
            "to another unit"
        )
    click.echo(
        f"{result.files} files linked ({result.hard_links} hard links) in "
        f"{result.chips_dir}"
    )


@pair.command("patches")
@click.option(
    "--batch",
    "batches",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Batch file of a block. Give it once per file.",
)
@click.option(
    "--unit",
    "unit_ids",
    multiple=True,
    help="A unit, in place of or beside the batch files. Repeatable.",
)
@click.option("--out", "out_dir", required=True, type=click.Path(path_type=Path))
@click.option(
    "--export-dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Exports folder (run_layers). Default: $CSDV_EXPORT_DIR.",
)
@click.option("--run-tag", default="default", show_default=True)
@click.option(
    "--threshold",
    type=click.IntRange(1, 100),
    default=90,
    show_default=True,
    help="Percent from which a cell is the core of a patch.",
)
@click.option(
    "--grow",
    type=click.IntRange(1, 100),
    default=50,
    show_default=True,
    help="Percent down to which a patch grows from its core. Give the "
    "threshold for a plain cut.",
)
@click.option("--min-ha", type=float, default=0.3, show_default=True)
@click.option(
    "--model-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Folder of the map, in place of the export's. For one unit.",
)
@click.option(
    "--layers-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Folder of chm_cover.tif and chm_height.tif, in place of the "
    "export's. For one unit.",
)
def patches_cmd(
    batches: tuple[Path, ...],
    unit_ids: tuple[str, ...],
    out_dir: Path,
    export_dir: Path | None,
    run_tag: str,
    threshold: int,
    grow: int,
    min_ha: float,
    model_dir: Path | None,
    layers_dir: Path | None,
) -> None:
    """The maps of a block of units as one layer of patches.

    Writes patches.gpkg and patches.csv (one row per patch and interval:
    dates, area, probability, cause, change in canopy cover and height),
    summary.json and seams.csv (the maps of adjoining units compared where
    they overlap).
    """
    from csdv_core.grid.units import Unit, grid_paths, read_batch
    from csdv_core.pairmodel.patches import PatchSettings, write_block

    _setup_logging()
    units: dict[str, tuple[float, float, float, float] | None] = {}
    for batch in batches:
        for unit in read_batch(batch):
            # Only a quad has neighbours whose maps reach into it.
            units[unit.unit_id] = unit.bbox_4326 if unit.kind == "cell" else None
    for unit_id in unit_ids:
        units.setdefault(
            unit_id,
            Unit.cell(unit_id).bbox_4326 if unit_id.startswith("q") else None,
        )
    if not units:
        raise click.UsageError("Name the units with --batch or --unit.")
    try:
        result = write_block(
            list(units.items()),
            export_dir or grid_paths().export_dir,
            out_dir,
            PatchSettings(threshold=threshold, grow=grow, min_ha=min_ha),
            run_tag=run_tag,
            model_dir=model_dir,
            layers_dir=layers_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"{result.patches} patches, {result.area_ha:.0f} ha, in {len(result.units)} "
        f"units ({result.joined} joined across a quad edge). {result.seams} rows "
        f"in seams.csv. Files in {result.out_dir}"
    )
    if result.without_map:
        click.echo(
            f"{len(result.without_map)} units have no model map: "
            f"{', '.join(result.without_map[:8])}"
        )
