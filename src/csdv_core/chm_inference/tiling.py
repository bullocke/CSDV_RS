"""csdv_core.chm_inference.tiling — NAIP-CHM chips on a fixed map lattice.

The upstream ``naip-chm`` inference lays its chips out from the top-left pixel
of each input mosaic, and the model reads its conditioning features
(elevation, climate, soil, NLCD, ecoregion) at the centre of each chip. The
same forest therefore gets a different CHM when it falls in another
processing unit: on the grid cell q4407152 (2018), shifting the mosaic by
101 x 77 px changed the CHM by 0.81 m RMSE per pixel and 0.63 m per 19.2 m
cell, while rerunning the unchanged mosaic changed it by 3 mm.

This module removes the dependence on the unit. Upstream resamples every
input to 0.6 m and keeps its origin, and every NAIP mosaic written by
``csdv download`` is snapped to 3 m in UTM, so every mosaic shares one global
0.6 m pixel lattice. :func:`lattice_chip_grid` starts the chips where the
global pixel index is a multiple of the chip stride, so a location away from
the mosaic edges is predicted from the same chips, with the same conditioning
samples, whichever unit it belongs to. Chips are still added at the mosaic
edges where the lattice does not reach them, so pixels within one chip
(259 m) of a mosaic edge still depend on the unit. The 150 m download pad
keeps most of that band outside the unit itself.

Run as ``python -m csdv_core.chm_inference.tiling INFERENCE_SCRIPT [ARGS...]``
from the upstream repository: the runner replaces
``src.inference_utils.create_chip_grid`` with the lattice grid and then runs
the upstream script unchanged. :func:`csdv_core.chm_inference.infer.predict_chm`
does this when ``chip_lattice`` is on.
"""

from __future__ import annotations

import importlib
import logging
import math
import runpy
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Pixel size (m) upstream resamples every input to before chipping.
TARGET_RES_M = 0.6

__all__ = [
    "TARGET_RES_M",
    "global_pixel_origin",
    "lattice_chip_grid",
    "lattice_starts",
    "main",
]


def global_pixel_origin(transform: Any, res_m: float = TARGET_RES_M) -> tuple[int, int]:
    """Global (column, row) index of a raster's top-left pixel at ``res_m``.

    Columns count east from x = 0 and rows count south from y = 0, so two
    rasters on the same pixel lattice get indices that differ by their pixel
    offset. A raster whose origin is not on the lattice is logged and its
    index rounded, which puts its chips within half a pixel of the lattice.
    """
    col = transform.c / res_m
    row = -transform.f / res_m
    for name, value in (("x", col), ("y", row)):
        if not math.isclose(value, round(value), abs_tol=1e-6):
            logger.warning(
                "Origin %s = %.4f m is not on the %.1f m lattice; chips are "
                "rounded to the nearest pixel",
                name,
                value * res_m,
                res_m,
            )
    return int(round(col)), int(round(row))


def lattice_starts(length: int, chip: int, stride: int, origin: int) -> list[int]:
    """Chip start positions along one axis of ``length`` pixels.

    Interior chips start where ``origin + start`` is a multiple of ``stride``.
    A chip at 0 and a chip ending at ``length`` are added when the lattice
    does not reach the edges. Neighbouring chips overlap by at least
    ``chip - stride`` pixels.
    """
    if length <= chip:
        return [0]
    first = (-origin) % stride
    starts = list(range(first, length - chip + 1, stride))
    if not starts or starts[0] > 0:
        starts.insert(0, 0)
    if starts[-1] + chip < length:
        starts.append(length - chip)
    return starts


def lattice_chip_grid(
    width: int,
    height: int,
    chip_size: int,
    overlap_fraction: float,
    col0: int,
    row0: int,
) -> list[tuple[int, int, int, int]]:
    """Chips as ``(row_start, col_start, row_end, col_end)``, on the lattice.

    Takes the arguments of upstream ``create_chip_grid`` plus the global
    index of the image's top-left pixel (:func:`global_pixel_origin`).

    Raises:
        ValueError: If ``chip_size`` is not a multiple of 16 (an upstream
            model requirement).
    """
    if chip_size % 16 != 0:
        raise ValueError(f"Chip size must be multiple of 16, got {chip_size}")
    stride = chip_size - int(chip_size * overlap_fraction)
    rows = lattice_starts(height, chip_size, stride, row0)
    cols = lattice_starts(width, chip_size, stride, col0)
    return [
        (r, c, min(r + chip_size, height), min(c + chip_size, width))
        for r in rows
        for c in cols
    ]


def _option(args: list[str], name: str) -> str:
    for i, arg in enumerate(args):
        if arg == name and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    raise SystemExit(f"tiling runner: upstream arguments lack {name}")


def main(argv: list[str] | None = None) -> None:
    """Run the upstream inference script with the lattice chip grid."""
    import rasterio

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("usage: python -m csdv_core.chm_inference.tiling SCRIPT ARGS")
    script = Path(argv[0]).resolve()
    args = argv[1:]
    with rasterio.open(_option(args, "--naip-quad")) as src:
        col0, row0 = global_pixel_origin(src.transform)

    repo = script.parent.parent
    sys.path.insert(0, str(repo))
    utils = importlib.import_module("src.inference_utils")

    def create_chip_grid(
        width: int, height: int, chip_size: int, overlap_fraction: float
    ) -> list[tuple[int, int, int, int]]:
        chips = lattice_chip_grid(
            width, height, chip_size, overlap_fraction, col0, row0
        )
        logger.info(
            "Lattice chip grid: %d chips of %d px, overlap %.0f%%, image origin "
            "at global pixel (%d, %d) of the %.1f m lattice",
            len(chips),
            chip_size,
            100 * overlap_fraction,
            col0,
            row0,
            TARGET_RES_M,
        )
        return chips

    utils.create_chip_grid = create_chip_grid
    sys.argv = [str(script), *args]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
