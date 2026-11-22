"""csdv_core.pairmodel.train: train the pair model on labelled chips.

Inputs: a chip folder of a label run and a :class:`TrainConfig`.
Outputs, in the run folder: ``best.pt`` and ``last.pt`` (checkpoints),
``config.json``, ``metrics.csv`` (one row per epoch), ``pr_val.csv`` (the
precision-recall curve of the best epoch on the validation units) and
``curves.png``.

The checkpoints are written to the run folder itself after every epoch, so
a job that is stopped can go on from ``last.pt``.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import logging
import math
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio  # noqa: F401  (GDAL has to load before torch, see __init__)

from csdv_core.pairmodel import data as D
from csdv_core.pairmodel.metrics import (
    AgentCounts,
    CellCounts,
    agent_loss,
    masked_bce_dice,
    to_percent,
)

logger = logging.getLogger(__name__)

__all__ = [
    "TrainConfig",
    "TrainResult",
    "describe",
    "describe_agents",
    "load_checkpoint",
    "plan",
    "split_chips",
    "train",
]


@dataclass
class TrainConfig:
    """Settings of one training run.

    Attributes:
        chips_dir: Chip folder of the label run.
        out_dir: Run folder.
        holdout_units: Test units, kept out of training and validation.
            Units tied to one of them in the chip folder's sites table are
            held out as well (:func:`csdv_core.pairmodel.data.tied_units`).
        hold_roles: Roles of the sites table whose units are held out too,
            for example ``("test",)``.
        val_units: Validation units (ignored when ``fold`` is set).
        fold: Use this grouped fold as validation.
        n_folds: Number of grouped folds.
        split: ``units`` holds whole units out for validation. ``blocks``
            validates on blocks of ground inside every unit that is not
            held out (:func:`csdv_core.pairmodel.data.assign_blocks`), so
            that every agent is trained on and validated on.
        block_chips: Side of a block of the ``blocks`` split, in chips.
        val_share: Share of the ``blocks`` split aimed at for validation.
        cache_dir: Folder for decoded chips
            (:mod:`csdv_core.pairmodel.cache`). None reads the chip rasters.
        encoder: ``resnet34`` or ``resnet18``.
        encoder_weights: torchvision state dict of the encoder, or None for
            a random start.
        cell_px: Pixels per output cell side (16 is 9.6 m).
        crop_px: Side of the training crop.
        batch_size: Samples per step.
        epochs: Largest number of epochs.
        samples_per_epoch: Draws per epoch (with replacement, weighted).
        lr: Peak learning rate of AdamW.
        weight_decay: Weight decay of AdamW.
        workers: Loader processes.
        reverse_prob: Share of training pairs fed in reverse order.
        n_dates: 2 trains a pair model. 3 also feeds the image after the
            pair where a chip has one (see ``SiamesePairNet``).
        third_drop: With three dates: share of training samples whose third
            image is withheld.
        loss_draws: Share of training draws that hold loss.
        run_draws: Share of draws for the samples without loss of named label
            runs of a merged chip folder (see ``sample_weights``).
        exclude_runs: Label runs of a merged chip folder that are never
            drawn in training. The split, the validation samples and the
            band statistics stay those of the whole folder, so the run is
            the control of one trained with them.
        agents: Train a second head that names the agent of a loss.
        agent_weight: Weight of the agent head's loss beside the loss head's.
        patience: Epochs without a better validation score before stopping.
        amp: Mixed precision on a GPU.
        seed: Seed of the split, the statistics and the loaders.
        max_val_samples: Cap on validation samples (for quick tests).
        device: ``cuda``, ``cpu`` or ``auto``.
    """

    chips_dir: Path
    out_dir: Path
    holdout_units: tuple[str, ...] = ()
    hold_roles: tuple[str, ...] = ()
    val_units: tuple[str, ...] = ()
    fold: int | None = None
    n_folds: int = 5
    split: str = "units"
    block_chips: int = 8
    val_share: float = 0.2
    cache_dir: Path | None = None
    encoder: str = "resnet34"
    encoder_weights: Path | None = None
    cell_px: int = 16
    crop_px: int = 384
    batch_size: int = 32
    epochs: int = 40
    samples_per_epoch: int = 8000
    lr: float = 2e-4
    weight_decay: float = 1e-4
    workers: int = 7
    reverse_prob: float = 0.2
    n_dates: int = 2
    third_drop: float = 0.3
    loss_draws: float = 0.5
    run_draws: dict[str, float] = field(default_factory=dict)
    exclude_runs: tuple[str, ...] = ()
    agents: bool = True
    agent_weight: float = 0.5
    patience: int = 8
    amp: bool = True
    seed: int = 0
    max_val_samples: int | None = None
    device: str = "auto"

    def to_json(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        for key, value in out.items():
            if isinstance(value, Path):
                out[key] = str(value)
        return out


@dataclass
class TrainResult:
    """What a training run produced."""

    out_dir: Path
    epochs_run: int
    best_epoch: int
    best_score: float
    best_threshold: float
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _Plan:
    """A split of the index.

    Attributes:
        index: Every sample of the chip folder.
        split: Units of each part. With the ``blocks`` split the training
            and validation units are the same.
        train_index: Training samples.
        val_index: Validation samples.
        sides: With the ``blocks`` split, ``train``, ``val`` or ``dropped``
            per chip outside the held-out units. None otherwise.
        agent_cells: Loss cells per agent and chip, when they were counted.
    """

    index: Any
    split: D.Split
    train_index: Any
    val_index: Any
    sides: Any = None
    agent_cells: Any = None
    held_with: tuple[str, ...] = ()


def held_out(cfg: TrainConfig, index: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Every unit a run holds out, and those among them it did not name.

    The named units, the units of ``hold_roles``, and the units tied to any
    of these by the sites table of the chip folder. Units of the table that
    have no chip are left out.
    """
    sites = D.load_sites(cfg.chips_dir)
    known = set(index["unit"])
    held = set(cfg.holdout_units) | (set(D.role_units(sites, cfg.hold_roles)) & known)
    if cfg.hold_roles and sites is None:
        raise ValueError(
            f"No {D.SITES_FILE} in {cfg.chips_dir}: roles cannot be held out"
        )
    held |= set(D.tied_units(sites, sorted(held))) & known
    added = tuple(sorted(held - set(cfg.holdout_units)))
    return tuple(sorted(held)), added


def _plan_blocks(cfg: TrainConfig, index: Any) -> _Plan:
    """Split by blocks of ground inside every unit that is not held out."""
    if cfg.val_units or cfg.fold is not None:
        raise ValueError(
            "The blocks split chooses its own validation ground. "
            "Leave out the validation units and the fold."
        )
    known = sorted(index["unit"].unique())
    unknown = sorted(set(cfg.holdout_units) - set(known))
    if unknown:
        raise ValueError(f"Units not in the chips: {', '.join(unknown)}")
    rest = tuple(u for u in known if u not in set(cfg.holdout_units))
    chips = index[index["unit"].isin(rest)].drop_duplicates("chip_id")
    cells = D.agent_cells(cfg.chips_dir, chips["chip_id"].tolist(), cfg.cell_px)
    sides = D.assign_blocks(
        chips,
        cells,
        block_chips=cfg.block_chips,
        val_share=cfg.val_share,
        seed=cfg.seed,
    )
    side = index["chip_id"].map(sides)
    train_index = index[side == "train"].reset_index(drop=True)
    val_index = index[side == "val"].reset_index(drop=True)
    if train_index.empty or val_index.empty:
        raise ValueError("The blocks split left training or validation empty")
    split = D.Split(rest, rest, tuple(sorted(cfg.holdout_units)))
    return _Plan(index, split, train_index, val_index, sides, cells)


def plan(cfg: TrainConfig) -> _Plan:
    """Read the index and split it, without touching the imagery.

    Raises:
        ValueError: If the split is unknown or cannot be made.
    """
    index = D.load_index(cfg.chips_dir)
    holdout, added = held_out(cfg, index)
    cfg = dataclasses.replace(cfg, holdout_units=holdout)
    if cfg.split == "blocks":
        planned = _plan_blocks(cfg, index)
    elif cfg.split == "units":
        split = D.make_split(
            sorted(index["unit"].unique()),
            holdout=cfg.holdout_units,
            val_units=cfg.val_units,
            fold=cfg.fold,
            n_folds=cfg.n_folds,
            seed=cfg.seed,
        )
        train_index = index[index["unit"].isin(split.train)].reset_index(drop=True)
        val_index = index[index["unit"].isin(split.val)].reset_index(drop=True)
        planned = _Plan(index, split, train_index, val_index)
    else:
        raise ValueError(f"Unknown split {cfg.split!r}: use units or blocks")
    planned.held_with = added
    if cfg.max_val_samples and len(planned.val_index) > cfg.max_val_samples:
        planned.val_index = planned.val_index.sample(
            cfg.max_val_samples, random_state=cfg.seed
        ).reset_index(drop=True)
    return planned


def split_chips(planned: _Plan) -> dict[str, list[str]] | None:
    """Chips of each side of a ``blocks`` split, or None for a unit split."""
    if planned.sides is None:
        return None
    return {
        side: sorted(planned.sides.index[planned.sides == side])
        for side in ("train", "val", "dropped")
    }


def describe(planned: _Plan) -> str:
    """A few lines on the size and balance of a planned run."""
    lines = []
    for name, units, part in (
        ("train", planned.split.train, planned.train_index),
        ("val", planned.split.val, planned.val_index),
    ):
        with_loss = int((part["loss_share"] >= 0.01).sum())
        lines.append(
            f"{name}: {part['unit'].nunique()} units, "
            f"{part['chip_id'].nunique()} chips, "
            f"{len(part)} samples, {with_loss} with loss ({', '.join(units)})"
        )
    if planned.sides is not None:
        dropped = int((planned.sides == "dropped").sum())
        lines.append(
            f"blocks: training and validation share the units. {dropped} "
            "training chips beside validation ground are left out"
        )
    held = planned.index[planned.index["unit"].isin(planned.split.test)]
    lines.append(
        f"test: {len(planned.split.test)} units, {held['chip_id'].nunique()} chips, "
        f"{len(held)} samples ({', '.join(planned.split.test)})"
    )
    if planned.held_with:
        lines.append(
            f"held out by role or because the sites table ties them to a "
            f"held-out unit: {', '.join(planned.held_with)}"
        )
    return "\n".join(lines)


def describe_draws(planned: _Plan, cfg: TrainConfig) -> str:
    """Share of the training draws per label run, with loss and without.

    A merged chip folder holds runs of very different sizes, and a unit's
    weight grows with the root of its sample count, so many small units can
    take most of the draws. Empty for a folder of one run.
    """
    part = planned.train_index
    if "run" not in part or part["run"].nunique() < 2:
        return ""
    weights = D.sample_weights(
        part,
        loss_draws=cfg.loss_draws,
        run_draws=cfg.run_draws,
        exclude_runs=cfg.exclude_runs,
    )
    loss = (part["loss_share"] >= 0.01).to_numpy()
    table = pd.DataFrame(
        {
            "units": part.groupby("run")["unit"].nunique(),
            "samples": part.groupby("run").size(),
            "draws with loss": pd.Series(weights * loss).groupby(part["run"]).sum(),
            "draws without": pd.Series(weights * ~loss).groupby(part["run"]).sum(),
        }
    )
    total = weights.sum()
    table[["draws with loss", "draws without"]] = (
        table[["draws with loss", "draws without"]] / total
    ).round(3)
    return "Training draws per label run:\n" + table.to_string()


def describe_agents(planned: _Plan, chips_dir: Path, cell_px: int) -> str:
    """Loss cells per agent on each side of a planned run, as a table.

    Counts the agent rasters of every chip when the plan has not done so,
    which takes about half a minute for a full label run.
    """
    chip_ids = sorted(planned.index["chip_id"].unique())
    cells = planned.agent_cells
    if cells is None or not set(chip_ids) <= set(cells.index):
        cells = D.agent_cells(chips_dir, chip_ids, cell_px)
    parts = {
        "train": planned.train_index,
        "val": planned.val_index,
        "test": planned.index[planned.index["unit"].isin(planned.split.test)],
    }
    table = pd.DataFrame(
        {
            name: cells.loc[sorted(part["chip_id"].unique())].sum()
            for name, part in parts.items()
        }
    )
    both = (table["train"] + table["val"]).replace(0, np.nan)
    table["val share"] = (table["val"] / both).round(2)
    table[["train", "val", "test"]] = table[["train", "val", "test"]].round(0)
    return "Loss cells per agent:\n" + table.to_string()


def _git_commit() -> str:
    try:
        repo = Path(__file__).resolve().parents[3]
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return head + ("+changes" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _device(name: str) -> Any:
    import torch

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def load_checkpoint(path: Path, device: Any = "cpu") -> tuple[Any, dict[str, Any]]:
    """The model of a checkpoint in evaluation mode, and the checkpoint itself."""
    import torch

    from csdv_core.pairmodel.model import SiamesePairNet

    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = SiamesePairNet(
        encoder=ckpt["encoder"],
        cell_px=ckpt["cell_px"],
        weights=None,
        n_agents=int(ckpt.get("n_agents", 0)),
        n_dates=int(ckpt.get("n_dates", 2)),
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def _validate(
    model: Any, loader: Any, device: Any, units: list[str], amp: bool
) -> tuple[float, dict[str, CellCounts], AgentCounts]:
    import torch

    counts = {unit: CellCounts() for unit in units}
    agents = AgentCounts(D.AGENT_NAMES)
    total, batches = 0.0, 0
    model.eval()
    with torch.no_grad():
        for a, b, target, agent, unit_ids, *third in loader:
            a, b = a.to(device, non_blocking=True), b.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            c, has_c = _third(third, device)
            with torch.autocast(device.type, enabled=amp):
                logits, agent_logits = model.forward_agents(a, b, c, has_c)
            total += float(masked_bce_dice(logits, target))
            batches += 1
            percent = to_percent(torch.sigmoid(logits.float()).squeeze(1).cpu().numpy())
            labels = target.cpu().numpy()
            for i, unit_id in enumerate(unit_ids.tolist()):
                counts[units[unit_id]].update(percent[i], labels[i])
            if agent_logits is not None:
                agents.update(agent_logits.argmax(dim=1).cpu().numpy(), agent.numpy())
    return total / max(batches, 1), counts, agents


def _third(third: list[Any], device: Any) -> tuple[Any, Any]:
    """The third image and its flag of a batch, or two Nones for a pair model."""
    if not third:
        return None, None
    c, has_c = third
    return c.to(device, non_blocking=True), has_c.to(device, non_blocking=True)


def _macro_ap(counts: dict[str, CellCounts]) -> float:
    """Mean average precision over the units that hold loss cells."""
    values = [c.average_precision() for c in counts.values() if c.n_pos]
    return float(np.mean(values)) if values else float("nan")


def _write_curves(history: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="training")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=False)
    axes[1].plot(epochs, [row["val_ap_macro"] for row in history], label="AP, mean")
    axes[1].plot(epochs, [row["val_f1"] for row in history], label="F1 at 0.5")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation score")
    axes[1].set_ylim(0, 1)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def train(cfg: TrainConfig, *, resume: bool = True) -> TrainResult:
    """Train, validate after every epoch, keep the best and the last state.

    Args:
        cfg: The run's settings.
        resume: Go on from ``last.pt`` in the run folder when it exists.

    Raises:
        FileNotFoundError: If chip files or the encoder weights are missing.
    """
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from csdv_core.config import load_label
    from csdv_core.pairmodel.model import SiamesePairNet, count_parameters

    planned = plan(cfg)
    logger.info("\n%s", describe(planned))
    used = sorted(
        set(planned.train_index["chip_id"]) | set(planned.val_index["chip_id"])
    )
    missing = D.missing_files(cfg.chips_dir, used)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} chip files are missing under {cfg.chips_dir}, "
            f"for example {missing[0]}"
        )
    if cfg.encoder_weights is not None and not Path(cfg.encoder_weights).is_file():
        raise FileNotFoundError(f"No encoder weights at {cfg.encoder_weights}")
    if planned.sides is not None:
        logger.info("\n%s", describe_agents(planned, Path(cfg.chips_dir), cfg.cell_px))
    if cfg.cache_dir is not None:
        # Before the device is set up: the build starts processes.
        from csdv_core.pairmodel.cache import build_cache

        # Chips of an excluded run are never drawn, so they are not decoded.
        drawn = planned.train_index
        if cfg.exclude_runs and "run" in drawn:
            drawn = drawn[~drawn["run"].isin(set(cfg.exclude_runs))]
        build_cache(
            cfg.chips_dir,
            cfg.cache_dir,
            pd.concat([drawn, planned.val_index])[["chip_id", "n_dates"]],
            workers=max(cfg.workers, 1),
        )
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = _device(cfg.device)
    amp = cfg.amp and device.type == "cuda"
    pooling = load_label().pooling
    torch.manual_seed(cfg.seed)

    last_path = out / "last.pt"
    state = None
    if resume and last_path.is_file():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        logger.info("Resuming after epoch %d from %s", state["epoch"], last_path)
    if state is not None and int(state.get("n_dates", 2)) != cfg.n_dates:
        raise ValueError(
            f"{last_path} holds a model with {state.get('n_dates', 2)} dates. "
            "Train into a new run folder, or pass --no-resume."
        )
    if state is not None:
        mean, std = np.asarray(state["mean"]), np.asarray(state["std"])
    else:
        mean, std = D.band_stats(
            cfg.chips_dir, sorted(set(planned.train_index["chip_id"])), seed=cfg.seed
        )
    logger.info("Band mean %s, std %s", np.round(mean, 3), np.round(std, 3))

    val_units = list(planned.split.val)
    aug = D.Augment(
        crop_px=cfg.crop_px, reverse_prob=cfg.reverse_prob, third_drop=cfg.third_drop
    )
    common = {
        "n_dates": cfg.n_dates,
        "cell_px": cfg.cell_px,
        "pooling": pooling,
        "mean": mean,
        "std": std,
        "cache_dir": cfg.cache_dir,
    }
    train_set = D.PairDataset(
        planned.train_index, cfg.chips_dir, aug=aug, seed=cfg.seed, **common
    )
    val_set = D.PairDataset(planned.val_index, cfg.chips_dir, units=val_units, **common)
    weights = D.sample_weights(
        planned.train_index,
        loss_draws=cfg.loss_draws,
        run_draws=cfg.run_draws,
        exclude_runs=cfg.exclude_runs,
    )
    draws = describe_draws(planned, cfg)
    if draws:
        logger.info("\n%s", draws)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=cfg.samples_per_epoch,
        replacement=True,
    )
    loader_args = {
        "num_workers": cfg.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": cfg.workers > 0,
    }
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        sampler=sampler,
        drop_last=True,
        **loader_args,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False, **loader_args
    )

    model = SiamesePairNet(
        encoder=cfg.encoder,
        cell_px=cfg.cell_px,
        weights=cfg.encoder_weights if state is None else None,
        n_agents=len(D.AGENT_NAMES) if cfg.agents else 0,
        n_dates=cfg.n_dates,
    ).to(device)
    logger.info(
        "%s, %d dates, %.1f M parameters",
        cfg.encoder,
        cfg.n_dates,
        count_parameters(model) / 1e6,
    )
    if cfg.n_dates == 3:
        later = planned.train_index["k"] + 2 < planned.train_index["n_dates"]
        logger.info(
            "%d of %d training samples have a third date, withheld for %.0f%% of "
            "the draws and for reversed pairs",
            int(later.sum()),
            len(later),
            100 * cfg.third_drop,
        )
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = cfg.epochs * steps_per_epoch

    def rate(step: int) -> float:
        if step < steps_per_epoch:
            return (step + 1) / steps_per_epoch
        done = (step - steps_per_epoch) / max(total_steps - steps_per_epoch, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(done, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, rate)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    history: list[dict[str, Any]] = []
    best_score, best_epoch, best_threshold, start = -1.0, 0, 0.5, 1
    if state is not None:
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history = list(state["history"])
        best_score = state["best_score"]
        best_epoch = state["best_epoch"]
        best_threshold = state["best_threshold"]
        start = state["epoch"] + 1

    record = {
        "config": cfg.to_json(),
        "encoder": cfg.encoder,
        "cell_px": cfg.cell_px,
        "n_agents": len(D.AGENT_NAMES) if cfg.agents else 0,
        "n_dates": cfg.n_dates,
        "agent_names": list(D.AGENT_NAMES),
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "split": dataclasses.asdict(planned.split),
        "split_mode": cfg.split,
        "split_chips": split_chips(planned),
        "pooling": pooling.model_dump(),
        "chips_sha1": _sha1(Path(cfg.chips_dir) / "chips.parquet"),
        "git_commit": _git_commit(),
        "torch": torch.__version__,
    }
    if state is not None and state.get("split_chips") != record["split_chips"]:
        raise ValueError(
            f"The split differs from the one stored in {last_path}. "
            "Train into a new run folder, or pass --no-resume."
        )
    (out / "config.json").write_text(json.dumps(record, indent=1), encoding="utf-8")

    epoch = start - 1
    for epoch in range(start, cfg.epochs + 1):
        model.train()
        clock = time.perf_counter()
        waited, running, steps = 0.0, 0.0, 0
        mark = time.perf_counter()
        for a, b, target, agent, _, *third in train_loader:
            waited += time.perf_counter() - mark
            a, b = a.to(device, non_blocking=True), b.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            c, has_c = _third(third, device)
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                logits, agent_logits = model.forward_agents(a, b, c, has_c)
            loss = masked_bce_dice(logits, target)
            if agent_logits is not None:
                agent = agent.to(device, non_blocking=True)
                loss = loss + cfg.agent_weight * agent_loss(agent_logits, agent)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            running += float(loss.detach())
            steps += 1
            mark = time.perf_counter()
        train_seconds = time.perf_counter() - clock
        clock = time.perf_counter()
        val_loss, counts, agent_counts = _validate(
            model, val_loader, device, val_units, amp
        )
        val_seconds = time.perf_counter() - clock
        overall = CellCounts()
        for unit_counts in counts.values():
            overall = overall + unit_counts
        at_half = overall.at(0.5)
        threshold, _ = overall.best_f1()
        score = _macro_ap(counts)
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": running / max(steps, 1),
            "val_loss": val_loss,
            "val_ap_macro": score,
            "val_ap": overall.average_precision(),
            "val_precision": at_half["precision"],
            "val_recall": at_half["recall"],
            "val_f1": at_half["f1"],
            "val_iou": at_half["iou"],
            "val_best_threshold": threshold,
            "val_agent_accuracy": agent_counts.accuracy(),
            "val_agent_balanced_accuracy": agent_counts.balanced_accuracy(),
            "val_agent_cells": agent_counts.n,
            "lr": scheduler.get_last_lr()[0],
            "train_seconds": round(train_seconds, 1),
            "loader_wait_seconds": round(waited, 1),
            "val_seconds": round(val_seconds, 1),
        }
        for unit, unit_counts in counts.items():
            row[f"ap_{unit}"] = unit_counts.average_precision()
            row[f"f1_{unit}"] = unit_counts.at(0.5)["f1"]
        history.append(row)
        logger.info(
            "epoch %d: train %.4f, val %.4f, AP %.3f (mean of units %.3f), "
            "F1 %.3f, agent accuracy %.3f on %d cells, %.0f s (%.0f s waiting "
            "for data), validation %.0f s",
            epoch,
            row["train_loss"],
            val_loss,
            row["val_ap"],
            score,
            row["val_f1"],
            row["val_agent_accuracy"],
            agent_counts.n,
            train_seconds,
            waited,
            val_seconds,
        )
        better = np.isfinite(score) and score > best_score
        if better:
            best_score, best_epoch, best_threshold = float(score), epoch, threshold
        checkpoint = {
            **record,
            "model": model.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "best_threshold": best_threshold,
            "val": row,
            "history": history,
        }
        if better:
            torch.save(checkpoint, out / "best.pt")
            curve = [{"unit": "all", **point} for point in overall.curve()]
            for unit, unit_counts in counts.items():
                curve += [{"unit": unit, **point} for point in unit_counts.curve()]
            _write_csv(out / "pr_val.csv", curve)
            if cfg.agents:
                _write_csv(out / "agent_confusion_val.csv", agent_counts.rows())
        torch.save(
            {
                **checkpoint,
                "optimiser": optimiser.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
            },
            out / "last.pt.tmp",
        )
        (out / "last.pt.tmp").replace(last_path)
        _write_csv(out / "metrics.csv", history)
        if epoch - best_epoch >= cfg.patience:
            logger.info("No better epoch for %d epochs, stopping", cfg.patience)
            break
    if history:
        _write_curves(history, out / "curves.png")
    if not (out / "best.pt").is_file() and last_path.is_file():
        # No epoch had a finite score (no loss cell in validation).
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        for key in ("optimiser", "scheduler", "scaler"):
            state.pop(key, None)
        torch.save(state, out / "best.pt")
    return TrainResult(out, epoch, best_epoch, best_score, best_threshold, history)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    names: list[str] = []
    for row in rows:
        names += [key for key in row if key not in names]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
