"""Fixtures for the unit tests: a small grid-run export for the review tools."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box

from csdv_core.grid.export import NODATA_U8, write_cog

REVIEW_TAG = "default"
#: Unit, CRS, number of objects. Two UTM zones, as in the real pilot.
REVIEW_UNITS = [("q4407159", "EPSG:26919", 12), ("q3912112", "EPSG:26910", 5)]
REVIEW_YEARS = [2015, 2018, 2021]
_CELL = 19.2


def _write_review_unit(root: Path, unit: str, crs: str, n: int) -> None:
    run = root / "runs" / REVIEW_TAG / unit
    run.mkdir(parents=True)
    left, top = 300000.0, 4500000.0
    rows, geoms = [], []
    for i in range(n):
        x0 = left + (i % 4) * 200.0
        y0 = top - (i // 4) * 200.0
        geoms.append(box(x0, y0 - 4 * _CELL, x0 + 4 * _CELL, y0))
        onset = REVIEW_YEARS[1 + i % 2]
        rows.append(
            {
                "object_id": i + 1,
                "onset_year": onset,
                "year_before": REVIEW_YEARS[REVIEW_YEARS.index(onset) - 1],
                "interval": f"{REVIEW_YEARS[REVIEW_YEARS.index(onset) - 1]}-{onset}",
                "onset_index": 1 + i % 2,
                "n_cells": 16,
                "area_ha": 16 * _CELL * _CELL / 1e4,
                "earlier_share": 0.0,
                "growth": i % 3 == 0,
                "mean_prob": 0.9,
                "max_prob": 1.0,
                "mean_z": -3.0,
                "follow_up": onset != REVIEW_YEARS[-1],
            }
        )
    table = pd.DataFrame(rows)
    objects = gpd.GeoDataFrame(table, geometry=geoms, crs=crs)
    objects.to_file(run / "disturbances.gpkg", layer="objects")
    # Every other object has a refined outline, a little inside its cells.
    refined = objects.iloc[::2][["object_id", "onset_year", "year_before", "interval"]]
    refined = gpd.GeoDataFrame(
        refined, geometry=objects.iloc[::2].geometry.buffer(-5.0).values, crs=crs
    )
    refined.to_file(run / "disturbances.gpkg", layer="refined_outlines")
    table.to_csv(run / "objects.csv", index=False)
    params = {"params": {"inputs": {"years": REVIEW_YEARS}}}
    (run / "run_params.json").write_text(json.dumps(params))
    record = {
        "unit_id": unit,
        "label": f"Site_{unit}",
        "bbox_4326": [-71.0, 44.0, -70.9, 44.1],
        "years": {str(y): {"source": "pc", "coverage": 1.0} for y in REVIEW_YEARS},
    }
    (run / "unit.json").write_text(json.dumps(record))
    transform = from_origin(left - 100, top + 100, _CELL, _CELL)
    onset = np.zeros((60, 60), dtype=np.uint8)
    onset[10:14, 10:14] = 18
    onset[-1] = NODATA_U8
    write_cog(run / "onset_year_u8.tif", onset, crs=crs, transform=transform)
    looks = root / "quicklooks" / unit
    chm_transform = from_origin(left - 100, top + 100, 3.0, 3.0)
    for k, year in enumerate(REVIEW_YEARS):
        chm = np.full((400, 400), 80 - 30 * k, dtype=np.uint8)  # 20, 12.5, 5 m
        write_cog(looks / f"chm_{year}_3m.tif", chm, crs=crs, transform=chm_transform)


@pytest.fixture()
def review_exports(tmp_path: Path) -> Path:
    """An export folder with two units of one run tag."""
    root = tmp_path / "exports"
    for unit, crs, n in REVIEW_UNITS:
        _write_review_unit(root, unit, crs, n)
    return root
