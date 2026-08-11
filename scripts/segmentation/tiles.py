"""Build and freeze the tile manifest for the segmentation sweep.

Tiles are chosen once, written to ``tiles.geojson``, and committed. Every other
script reads that file, so no result depends on a tile someone picked by eye
while looking at an answer they liked.

Selection is stratified rather than random. A random sample of a mostly closed
forest returns mostly closed forest, and the parameter sets differ most where
the canopy is broken, short or missing. Strata come from the interpreter labels
where a labelled stand covers enough of the tile, and from canopy statistics
otherwise.

A 300 m tile is 9 ha. At the densities under test that holds several hundred
crowns, which is comfortably above the count at which crown CV settles, even
after edge crowns are excluded.

Run:
    .micromamba/envs/CSDV/bin/python scripts/segmentation/tiles.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    CANOPY_FLOOR_M,
    REFERENCE_YEAR,
    SITE,
    TILE_PATH,
    TILE_SIZE_M,
    configure_logging,
    elkinsville_chm,
    stands,
    transfer_sites,
)

logger = logging.getLogger("tiles")

#: How many tiles per stratum, and which split each stratum feeds.
#: Twelve tune and six held out, balanced so no stratum sits only in one side.
QUOTAS = {
    "undisturbed_tall": (3, 1),
    "undisturbed_mixed": (2, 1),
    "selection_harvest": (2, 1),
    "clearcut": (2, 1),
    "wind": (2, 1),
    "edge_and_gap": (1, 1),
}

#: A labelled stand must cover this much of a tile before the tile takes its
#: stratum. Below it the disturbance is a detail rather than the subject.
MIN_STAND_COVER = 0.15

STRATUM_FROM_LABEL = {
    "Uneven-age / selection harvest": "selection_harvest",
    "Clearcut": "clearcut",
    "Clearcut with reserves": "clearcut",
    "Shelterwood establishment cut": "clearcut",
    "Overstory removal": "clearcut",
    "Wind": "wind",
    "Flood": "wind",
    "Tree mortality": "wind",
}


def candidate_grid(chm_path: Path) -> gpd.GeoDataFrame:
    """Every whole 300 m tile inside the CHM footprint."""
    with rasterio.open(chm_path) as src:
        left, bottom, right, top = src.bounds
        crs = src.crs
    xs = np.arange(left, right - TILE_SIZE_M, TILE_SIZE_M)
    ys = np.arange(bottom, top - TILE_SIZE_M, TILE_SIZE_M)
    cells = [box(x, y, x + TILE_SIZE_M, y + TILE_SIZE_M) for x in xs for y in ys]
    return gpd.GeoDataFrame({"geometry": cells}, crs=crs)


def describe_tiles(grid: gpd.GeoDataFrame, chm_path: Path) -> pd.DataFrame:
    """Canopy statistics per tile, read once from the reference year."""
    rows = []
    with rasterio.open(chm_path) as src:
        nodata = src.nodata
        for geom in grid.geometry:
            win = from_bounds(*geom.bounds, transform=src.transform)
            arr = src.read(1, window=win).astype("float32")
            if nodata is not None:
                arr[arr == nodata] = np.nan
            arr[arr < -100] = np.nan
            total = arr.size
            finite = np.isfinite(arr)
            canopy = finite & (arr >= CANOPY_FLOOR_M)
            heights = arr[canopy]
            rows.append(
                {
                    "nodata_fraction": float(1.0 - finite.sum() / total),
                    "canopy_fraction": float(canopy.sum() / max(total, 1)),
                    "height_median": (
                        float(np.median(heights)) if heights.size else float("nan")
                    ),
                    "height_p90": (
                        float(np.percentile(heights, 90))
                        if heights.size
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def assign_strata(grid: gpd.GeoDataFrame, stand_gdf: gpd.GeoDataFrame) -> pd.Series:
    """Label each tile by what dominates it."""
    tile_area = TILE_SIZE_M * TILE_SIZE_M
    overlay = gpd.overlay(
        grid.reset_index(names="tile_idx")[["tile_idx", "geometry"]],
        stand_gdf[["stand_id", "dist_label", "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )
    overlay["cover"] = overlay.area / tile_area
    best = (
        overlay.sort_values("cover", ascending=False)
        .drop_duplicates("tile_idx")
        .set_index("tile_idx")
    )

    strata = pd.Series("undisturbed", index=grid.index, dtype=object)
    labels = pd.Series("", index=grid.index, dtype=object)
    stand_ids = pd.Series("", index=grid.index, dtype=object)
    for idx, row in best.iterrows():
        if row["cover"] >= MIN_STAND_COVER:
            mapped = STRATUM_FROM_LABEL.get(row["dist_label"])
            if mapped:
                strata.at[idx] = mapped
                labels.at[idx] = row["dist_label"]
                stand_ids.at[idx] = row["stand_id"]
    grid["dist_label"] = labels
    grid["stand_id"] = stand_ids
    return strata


def main() -> int:
    """Build the manifest and write it."""
    configure_logging()
    chm = elkinsville_chm(REFERENCE_YEAR)
    grid = candidate_grid(chm)
    logger.info("%d candidate tiles at %g m", len(grid), TILE_SIZE_M)

    stats = describe_tiles(grid, chm)
    grid = pd.concat([grid, stats], axis=1)
    grid = gpd.GeoDataFrame(grid, geometry="geometry", crs=stats.index.name or None)
    grid = gpd.GeoDataFrame(grid, geometry="geometry")
    grid.set_crs(candidate_grid(chm).crs, inplace=True)

    strata = assign_strata(grid, stands())

    # A tile carrying a lot of nodata or very little canopy is the pathology
    # case. Segmentation has to survive field edges, roads and holes, and those
    # never appear in a comfortable interior tile.
    broken = (grid["nodata_fraction"] > 0.02) | (grid["canopy_fraction"] < 0.75)
    strata[broken & (strata == "undisturbed")] = "edge_and_gap"

    # Split the remaining undisturbed tiles by stature, so the reference
    # condition is not only the tallest closed canopy in the scene.
    undisturbed = strata == "undisturbed"
    if undisturbed.any():
        cut = grid.loc[undisturbed, "height_median"].median()
        strata[undisturbed & (grid["height_median"] >= cut)] = "undisturbed_tall"
        strata[undisturbed & (grid["height_median"] < cut)] = "undisturbed_mixed"
    grid["stratum"] = strata

    # Deterministic pick: rank within a stratum by how typical the tile is,
    # measured as distance from the stratum's median canopy fraction, then take
    # the quota in that order. No random seed, so the choice is reproducible
    # from the code alone.
    chosen = []
    for stratum, (n_tune, n_hold) in QUOTAS.items():
        pool = grid[grid["stratum"] == stratum].copy()
        if pool.empty:
            logger.warning("Stratum %s has no candidates", stratum)
            continue
        target = pool["canopy_fraction"].median()
        pool["typicality"] = (pool["canopy_fraction"] - target).abs()
        pool = pool.sort_values(["typicality", "canopy_fraction"], kind="mergesort")
        want = n_tune + n_hold
        take = pool.head(want).copy()
        if len(take) < want:
            logger.warning(
                "Stratum %s: wanted %d tiles, found %d", stratum, want, len(take)
            )
        # Alternate the split so both sides see the same range of typicality.
        take["split"] = ["tune" if i % 3 != 2 else "holdout" for i in range(len(take))]
        n_actual_hold = int((take["split"] == "holdout").sum())
        logger.info(
            "%-18s %2d tiles (%d tune, %d holdout)",
            stratum,
            len(take),
            len(take) - n_actual_hold,
            n_actual_hold,
        )
        chosen.append(take)

    tiles = gpd.GeoDataFrame(pd.concat(chosen, ignore_index=True), crs=grid.crs)
    tiles["site"] = SITE
    tiles["tile_id"] = [
        f"{SITE}-{row.stratum}-{i:02d}" for i, row in enumerate(tiles.itertuples())
    ]

    # Transfer sites get their own tiles, and are never used for tuning.
    extra = []
    for site in transfer_sites():
        sub = candidate_grid(site.chm)
        sub_stats = describe_tiles(sub, site.chm)
        sub = gpd.GeoDataFrame(pd.concat([sub, sub_stats], axis=1), crs=sub.crs)
        sub = sub[(sub["nodata_fraction"] < 0.02) & (sub["canopy_fraction"] > 0.8)]
        if sub.empty:
            logger.warning("Transfer site %s produced no usable tiles", site.name)
            continue
        sub = sub.sort_values("canopy_fraction", ascending=False).head(4).copy()
        sub["site"] = site.name
        sub["stratum"] = "transfer"
        sub["split"] = "transfer"
        sub["dist_label"] = ""
        sub["stand_id"] = ""
        sub["tile_id"] = [f"{site.name}-transfer-{i:02d}" for i in range(len(sub))]
        extra.append(sub)
        logger.info("%-18s %2d tiles (transfer)", site.name, len(sub))

    # Elkinsville is EPSG:26916 and the transfer sites are EPSG:5070, so the
    # native bounds travel as plain columns and are the authoritative extent
    # every reader uses. The geometry column is reprojected to WGS84 purely so
    # one GeoJSON can hold tiles from both, and so the manifest renders.
    rows = []
    for part in [tiles] + extra:
        native_epsg = part.crs.to_epsg()
        wgs = part.to_crs(4326)
        for (_, row), geom in zip(part.iterrows(), wgs.geometry, strict=True):
            b = row.geometry.bounds
            rows.append(
                {
                    "tile_id": row["tile_id"],
                    "site": row["site"],
                    "stratum": row["stratum"],
                    "split": row["split"],
                    "dist_label": row.get("dist_label", ""),
                    "stand_id": row.get("stand_id", ""),
                    "minx": b[0],
                    "miny": b[1],
                    "maxx": b[2],
                    "maxy": b[3],
                    "epsg": native_epsg,
                    "nodata_fraction": row["nodata_fraction"],
                    "canopy_fraction": row["canopy_fraction"],
                    "height_median": row["height_median"],
                    "height_p90": row["height_p90"],
                    "geometry": geom,
                }
            )
    tiles = gpd.GeoDataFrame(rows, crs=4326)

    TILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tiles.to_file(TILE_PATH, driver="GeoJSON")
    logger.info("Wrote %s with %d tiles", TILE_PATH, len(tiles))
    print(
        tiles.groupby(["site", "stratum", "split"]).size().rename("tiles").to_string()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
