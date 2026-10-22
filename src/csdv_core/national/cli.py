"""csdv_core.national.cli — ``csdv benchmark-crowns`` for one national tile.

Runs :func:`csdv_core.national.benchmark.benchmark_tile` over every block of a
tile (or a sample of blocks) and writes three files named after the tile:
per-block timings, output bytes per crown by format, and a tile summary with
totals per engine. One Slurm array task runs one tile.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.command("benchmark-crowns")
@click.option(
    "--tile",
    "tile_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="NAIP-CHM tile COG.",
)
@click.option("--tile-id", default=None, help="Identifier. Defaults to the file stem.")
@click.option(
    "--out-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for <tile_id>_blocks.csv, _sizes.csv and _summary.json.",
)
@click.option("--engines", default="watershed,lidr", show_default=True)
@click.option("--block-px", type=int, default=2048, show_default=True)
@click.option(
    "--n-blocks",
    type=int,
    default=0,
    show_default=True,
    help="Random sample of blocks to run. 0 runs every block of the tile.",
)
@click.option("--scale-factor", type=float, default=0.01, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
def cli(
    tile_path: Path,
    tile_id: str | None,
    out_dir: Path,
    engines: str,
    block_px: int,
    n_blocks: int,
    scale_factor: float,
    seed: int,
) -> None:
    """Time crown engines over one national NAIP-CHM tile."""
    import numpy as np
    import pandas as pd
    import rasterio

    from csdv_core.national.benchmark import (
        benchmark_tile,
        cpu_model,
        iter_blocks,
        measure_output_sizes,
        tile_canopy_stats,
    )
    from csdv_core.segmentation.params import DEFAULT_PARAMS

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s"
    )
    logging.getLogger("csdv_core.segmentation").setLevel(logging.WARNING)
    tile_id = tile_id or tile_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    engine_list = tuple(e.strip() for e in engines.split(",") if e.strip())

    with rasterio.open(tile_path) as src:
        all_blocks = iter_blocks(src.height, src.width, block_px)
    blocks = all_blocks
    if 0 < n_blocks < len(all_blocks):
        rng = np.random.default_rng(seed)
        pick = sorted(rng.choice(len(all_blocks), size=n_blocks, replace=False))
        blocks = [all_blocks[i] for i in pick]

    t0 = time.perf_counter()
    results, crowns = benchmark_tile(
        tile_path,
        tile_id,
        engines=engine_list,
        params=DEFAULT_PARAMS,
        block_px=block_px,
        blocks=blocks,
        scale_factor=scale_factor,
        keep_crowns=True,
    )
    elapsed = time.perf_counter() - t0
    frame = pd.DataFrame([r.as_row() for r in results])
    frame["block_px"] = block_px
    frame["cpu"] = cpu_model()
    frame.to_csv(out_dir / f"{tile_id}_blocks.csv", index=False)

    summary: dict[str, object] = {
        "tile_id": tile_id,
        "cpu": cpu_model(),
        "params_key": DEFAULT_PARAMS.key,
        "block_px": block_px,
        "n_blocks_run": len(blocks),
        "n_blocks_tile": len(all_blocks),
        "elapsed_s": elapsed,
        **tile_canopy_stats(tile_path, scale_factor=scale_factor),
    }
    for engine in engine_list:
        sub = frame[frame["engine"] == engine]
        summary[f"{engine}_compute_s"] = float(sub["compute_s"].sum())
        summary[f"{engine}_wall_s"] = float(sub["wall_s"].sum())
        summary[f"{engine}_n_crowns"] = int(sub["n_crowns"].sum())
        summary[f"{engine}_peak_rss_mb"] = float(sub["peak_rss_mb"].max())
        summary[f"{engine}_n_errors"] = int((sub["error"].fillna("") != "").sum())

    ws = crowns.get("watershed")
    if ws is not None and not ws.empty:
        sizes = measure_output_sizes(ws, out_dir / f"tmp_{tile_id}")
        pd.DataFrame(
            {
                "format": list(sizes.bytes_per_crown),
                "bytes_per_crown": list(sizes.bytes_per_crown.values()),
                "write_s": [sizes.write_s[k] for k in sizes.bytes_per_crown],
                "n_crowns": sizes.n_crowns,
            }
        ).to_csv(out_dir / f"{tile_id}_sizes.csv", index=False)
        (out_dir / f"tmp_{tile_id}").rmdir()
    (out_dir / f"{tile_id}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    logger.info("Tile %s done in %.0f s: %s", tile_id, elapsed, summary)


__all__ = ["cli"]
