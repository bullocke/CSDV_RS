"""csdv_core.grid.compare: the canopy heights of one unit against another's.

A unit can be a whole quad or a small box inside one. Both get their canopy
heights from the same model, which reads a mosaic in chips on a lattice
shared by every unit, so the two should agree wherever the model sees the
same imagery around a pixel. Near the edge of a mosaic it does not: the
last chips reach past the imagery. This module measures the difference by
distance from the edge of the smaller unit, which is what decides how far
inside a box its candidates can be trusted.

Inputs: ``<data_root>/naip_chm/<unit>/<year>/`` of both units, one GeoTIFF
per year each.
Outputs: one row per year and band of distance.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from csdv_core.disturbance.inputs import discover_years
from csdv_core.grid.units import GridPaths

logger = logging.getLogger(__name__)

#: Upper edges of the bands of distance from the unit's edge, metres. The
#: model's chips make heights depend on the mosaic within about 260 m.
BAND_EDGES_M = (50.0, 150.0, 260.0)
STRIP_ROWS = 1024

__all__ = ["BAND_EDGES_M", "BandDifference", "compare_units", "write_comparison"]


@dataclass
class BandDifference:
    """Height difference of two units in one year and band of distance.

    Attributes:
        unit: The unit compared (the smaller one).
        against: The unit it is compared with.
        year: NAIP year.
        date_unit: Acquisition date in the unit's CHM file name.
        date_against: The same of the other unit. Different dates mean
            different flights, and the difference then says nothing about
            the model.
        band: Distance from the unit's edge, such as ``50-150 m``.
        pixels: Pixels with a height in both.
        rmse_m: Root mean square difference, metres.
        mean_m: Mean of unit minus against, metres.
        mae_m: Mean absolute difference, metres.
        share_over_half_m: Share of pixels that differ by more than 0.5 m.
    """

    unit: str
    against: str
    year: int
    date_unit: str
    date_against: str
    band: str
    pixels: int
    rmse_m: float
    mean_m: float
    mae_m: float
    share_over_half_m: float


def _band_names(edges: tuple[float, ...]) -> list[str]:
    lower = (0.0, *edges)
    names = [f"{a:g}-{b:g} m" for a, b in zip(lower, edges, strict=False)]
    return [*names, f"{edges[-1]:g} m and more"]


def _year_files(paths: GridPaths, unit_id: str) -> dict[int, object]:
    root = paths.data_root / "naip_chm" / unit_id
    return {y.year: y for y in discover_years(root)}


def compare_units(
    paths: GridPaths,
    unit_id: str,
    against_id: str,
    *,
    edges_m: tuple[float, ...] = BAND_EDGES_M,
) -> list[BandDifference]:
    """Compare the CHMs of ``unit_id`` with those of ``against_id``.

    Every year both units have is compared on the pixels of ``unit_id``.
    The other unit is read on that grid (nearest pixel, which changes
    nothing when the two share the 0.6 m lattice, as units of one UTM zone
    do).

    Raises:
        FileNotFoundError: If a unit has no CHM folder.
        ValueError: If the units share no year.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window

    mine, theirs = _year_files(paths, unit_id), _year_files(paths, against_id)
    years = sorted(set(mine) & set(theirs))
    if not years:
        raise ValueError(
            f"{unit_id} ({sorted(mine)}) and {against_id} ({sorted(theirs)}) "
            "share no year"
        )
    names = _band_names(edges_m)
    out: list[BandDifference] = []
    for year in years:
        a, b = mine[year], theirs[year]
        if len(a.chm_paths) != 1 or len(b.chm_paths) != 1:
            logger.warning("%d: more than one CHM file in a unit, skipped", year)
            continue
        n = np.zeros(len(names))
        s1, s2, sa, big = (np.zeros(len(names)) for _ in range(4))
        with (
            rasterio.open(a.chm_paths[0]) as src_a,
            rasterio.open(b.chm_paths[0]) as src_b,
        ):
            height, width, px = src_a.height, src_a.width, abs(src_a.transform.a)
            with WarpedVRT(
                src_b,
                crs=src_a.crs,
                transform=src_a.transform,
                width=width,
                height=height,
                resampling=Resampling.nearest,
            ) as vrt:
                cols = np.arange(width)
                d_cols = np.minimum(cols, width - 1 - cols) * px
                for row0 in range(0, height, STRIP_ROWS):
                    rows = np.arange(row0, min(row0 + STRIP_ROWS, height))
                    window = Window(0, row0, width, len(rows))
                    va = src_a.read(1, window=window)
                    vb = vrt.read(1, window=window)
                    valid = np.ones(va.shape, dtype=bool)
                    for values, nodata in (
                        (va, a.chm_nodata),
                        (vb, vrt.nodata if b.chm_nodata is None else b.chm_nodata),
                    ):
                        if nodata is not None:
                            valid &= values != nodata
                        if np.issubdtype(values.dtype, np.floating):
                            valid &= np.isfinite(values)
                    diff = va.astype("float32") * a.chm_scale - (
                        vb.astype("float32") * b.chm_scale
                    )
                    d_rows = np.minimum(rows, height - 1 - rows) * px
                    dist = np.minimum.outer(d_rows, d_cols)
                    band = np.digitize(dist, edges_m)
                    for k in range(len(names)):
                        d = diff[valid & (band == k)]
                        n[k] += d.size
                        s1[k] += float(d.sum())
                        s2[k] += float((d.astype("float64") ** 2).sum())
                        sa[k] += float(np.abs(d).sum())
                        big[k] += float((np.abs(d) > 0.5).sum())
        dates = [str(x.acquisition_dates[0] or "") for x in (a, b)]
        if dates[0] != dates[1]:
            logger.warning(
                "%d: %s was flown %s and %s %s, so they differ as flights do",
                year,
                unit_id,
                dates[0],
                against_id,
                dates[1],
            )
        for k, name in enumerate(names):
            if n[k] == 0:
                continue
            out.append(
                BandDifference(
                    unit=unit_id,
                    against=against_id,
                    year=year,
                    date_unit=dates[0],
                    date_against=dates[1],
                    band=name,
                    pixels=int(n[k]),
                    rmse_m=float(np.sqrt(s2[k] / n[k])),
                    mean_m=float(s1[k] / n[k]),
                    mae_m=float(sa[k] / n[k]),
                    share_over_half_m=float(big[k] / n[k]),
                )
            )
    return out


def write_comparison(rows: list[BandDifference], path: Path) -> Path:
    """Write the rows as CSV, with a first line that says what a row is."""
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(
            "# canopy height of one unit minus another's on the first unit's "
            "pixels, per year and band of distance from the first unit's edge\n"
        )
        pd.DataFrame([asdict(r) for r in rows]).to_csv(
            fh, index=False, float_format="%.4f"
        )
    return path
