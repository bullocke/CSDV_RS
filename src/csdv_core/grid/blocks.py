"""csdv_core.grid.blocks: the quads of a block of adjoining grid cells.

A block is every 7.5 minute cell of a bounding box that lies inside one
polygon, as a rule one state. NAIP states are flown in different years, and a
cell on a state line loses the years of its neighbour (``plan_years``), so a
block keeps to one state.

Inputs: a box in degrees and a polygon in EPSG:4326 or 4269. Outputs: units,
north to south and west to east, and batch files of ``csdv grid queue``.
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from pathlib import Path
from typing import Any

import numpy as np

from csdv_core.grid.units import Unit

logger = logging.getLogger(__name__)

__all__ = ["block_units", "write_batches"]

#: Side of a grid cell, degrees.
CELL_DEG = 0.125
KM_PER_DEG = 111.0


def block_units(
    bbox: tuple[float, float, float, float],
    polygon: Any | None = None,
    *,
    margin_km: float = 0.0,
    skip: Collection[str] = (),
    prefix: str = "",
) -> list[Unit]:
    """Cells of a box that lie wholly inside a polygon.

    Args:
        bbox: ``(west, south, east, north)`` in degrees. Edges are snapped
            outward to the grid.
        polygon: A shapely geometry in degrees, or None for every cell.
        margin_km: A cell must also keep this distance from the polygon's
            edge, for outlines that are drawn coarsely.
        skip: Unit IDs to leave out (cells that are processed already).
        prefix: Start of each unit's label, which ends in the cell number.

    Returns:
        Units from north to south, and from west to east within a row.
    """
    from shapely.geometry import box

    west, south, east, north = bbox
    lons = np.arange(
        np.floor(west / CELL_DEG) * CELL_DEG, east - 1e-9, CELL_DEG
    ).tolist()
    lats = np.arange(
        np.floor(south / CELL_DEG) * CELL_DEG, north - 1e-9, CELL_DEG
    ).tolist()
    inner = polygon
    if polygon is not None and margin_km > 0:
        inner = polygon.buffer(-margin_km / KM_PER_DEG)
    leave = set(skip)
    units: list[Unit] = []
    for lat in reversed(lats):
        for lon in lons:
            cell = box(lon, lat, lon + CELL_DEG, lat + CELL_DEG)
            if inner is not None and not inner.contains(cell):
                continue
            unit = Unit.cell_at(lon + CELL_DEG / 2, lat + CELL_DEG / 2)
            if unit.unit_id in leave:
                continue
            unit.label = f"{prefix}{unit.unit_id[1:]}"
            units.append(unit)
    return units


def write_batches(
    units: list[Unit], folder: Path, name: str, *, part_size: int = 0, note: str = ""
) -> list[Path]:
    """Write a block as one batch file, or as numbered parts.

    Args:
        units: Units of :func:`block_units`.
        folder: ``config/grid``.
        name: Name of the block. Parts are ``<name>_1.csv``, ``<name>_2.csv``.
        part_size: Largest number of units in a part. 0 writes one file,
            ``<name>.csv``.
        note: Comment lines for the head of every file.

    Returns:
        The files written.
    """
    if part_size <= 0 or len(units) <= part_size:
        groups = [units]
        paths = [folder / f"{name}.csv"]
    else:
        n_parts = int(np.ceil(len(units) / part_size))
        size = int(np.ceil(len(units) / n_parts))
        groups = [units[i : i + size] for i in range(0, len(units), size)]
        paths = [folder / f"{name}_{i + 1}.csv" for i in range(len(groups))]
    head = "".join(f"# {line}\n".replace("#  \n", "#\n") for line in note.splitlines())
    for path, group in zip(paths, groups, strict=True):
        rows = "".join(f"{u.label},,,,{u.unit_id},\n" for u in group)
        path.write_text(
            f"{head}label,lon,lat,buffer_km,cell,notes\n{rows}", encoding="utf-8"
        )
        logger.info("Wrote %s (%d quads)", path, len(group))
    return paths
