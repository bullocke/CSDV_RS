"""csdv_core.pairmodel.merge: one chip folder from the chips of several label runs.

Training reads a single chip folder (``<label dir>/chips``). To train on two
label runs, for example the scale test and a run of stable-forest windows,
this module writes a new label folder whose ``chips/`` holds

- ``chips.parquet``: the rows of every run, with a column ``run`` (the run's
  name) and window ids prefixed with it (``stable_forest:W0001``), so that
  windows of different runs never share an id;
- the raster files of every chip, as hard links where the file system allows
  them and symbolic links otherwise.

A chip id that an earlier run already holds is left out of a later run, and
the number left out is logged. The source folders are not changed.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["MergeResult", "merge_chips"]


@dataclass
class MergeResult:
    """What a merge wrote."""

    chips_dir: Path
    chips: dict[str, int]
    dropped: dict[str, int]
    files: int
    hard_links: int


def _link(source: Path, target: Path) -> bool:
    """Link ``target`` to ``source``. True for a hard link, False for a symlink."""
    if target.is_symlink() or target.exists():
        target.unlink()
    try:
        os.link(source, target)
        return True
    except OSError:
        target.symlink_to(source.resolve())
        return False


def merge_chips(out_dir: Path, runs: Mapping[str, Path]) -> MergeResult:
    """Merge the chip folders of several label runs into ``out_dir/chips``.

    Args:
        out_dir: New label folder. Its ``chips/`` must not hold a
            ``chips.parquet`` yet.
        runs: Run name to label folder (the folder that holds ``chips/``),
            in order of precedence.

    Raises:
        FileExistsError: If ``out_dir/chips/chips.parquet`` exists.
        FileNotFoundError: If a run has no ``chips/chips.parquet``.
        ValueError: If no run is given or a name holds a colon.
    """
    if not runs:
        raise ValueError("No label run to merge")
    chips_out = Path(out_dir) / "chips"
    if (chips_out / "chips.parquet").exists():
        raise FileExistsError(
            f"{chips_out}/chips.parquet exists. Merge into a new folder"
        )
    tables = []
    seen: set[str] = set()
    counts: dict[str, int] = {}
    dropped: dict[str, int] = {}
    sources: list[tuple[Path, str]] = []
    for name, label_dir in runs.items():
        if ":" in name:
            raise ValueError(f"A run name cannot hold a colon: {name!r}")
        parquet = Path(label_dir) / "chips" / "chips.parquet"
        if not parquet.exists():
            raise FileNotFoundError(f"No chips.parquet in {parquet.parent}")
        table = pd.read_parquet(parquet)
        dup = table["chip_id"].isin(seen)
        dropped[name] = int(dup.sum())
        table = table[~dup].copy()
        table["window_id"] = name + ":" + table["window_id"].astype(str)
        table["run"] = name
        seen.update(table["chip_id"])
        counts[name] = len(table)
        tables.append(table)
        sources.extend((parquet.parent, c) for c in table["chip_id"])
        logger.info(
            "%s: %d chips (%d already in an earlier run, left out)",
            name,
            len(table),
            dropped[name],
        )
    chips_out.mkdir(parents=True, exist_ok=True)
    files = hard = 0
    listings: dict[Path, dict[str, list[Path]]] = {}
    for folder, chip_id in sources:
        if folder not in listings:
            by_chip: dict[str, list[Path]] = {}
            for path in folder.glob("*.tif"):
                by_chip.setdefault(path.name.rsplit("_", 1)[0], []).append(path)
            listings[folder] = by_chip
        for path in listings[folder].get(chip_id, []):
            hard += _link(path, chips_out / path.name)
            files += 1
    merged = pd.concat(tables, ignore_index=True)
    merged.to_parquet(chips_out / "chips.parquet", index=False)
    logger.info(
        "%d chips, %d files linked (%d hard links) in %s",
        len(merged),
        files,
        hard,
        chips_out,
    )
    return MergeResult(chips_out, counts, dropped, files, hard)
