"""Export CHM and NDVI forest masks as GeoTIFFs.

Reads co-registered NAIP (RGBN, uint8) and NAIP-CHM (metres) rasters in strips,
builds forest masks with :mod:`csdv_core.masks.forest` at a chosen block size,
and writes one coded ``uint8`` GeoTIFF per year and source. File names carry a
hash of the parameters, so tuned exports sit next to earlier ones.

Inputs: per-year NAIP and CHM paths, a :class:`ForestMaskProfile`, and an
optional reference mask for NDVI statistics. Outputs: GeoTIFFs, a JSON sidecar
per export set, and rows appended to ``masks_index.csv``.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window

from csdv_core.config import ForestMaskProfile
from csdv_core.masks import forest as fm

logger = logging.getLogger(__name__)

#: Rows of output cells read per strip.
STRIP_CELLS = 64
#: Pixels sampled for NDVI reference statistics and Otsu thresholds.
SAMPLE_PX = 4_000_000


@dataclass(frozen=True)
class MaskExport:
    """One written mask raster."""

    path: Path
    year: int | None
    layer: str
    block_px: int
    tag: str
    forest_fraction: float
    threshold: float | None = None


@dataclass(frozen=True)
class MaskGrid:
    """Output grid for a block size nested in the 0.6 m source grid."""

    rows: int
    cols: int
    transform: Affine
    crs: str
    block_px: int

    @property
    def res_tag(self) -> str:
        return f"{abs(self.transform.a):.1f}m".replace(".", "p")


def mask_grid(src_path: Path, block_px: int) -> MaskGrid:
    """Output grid with whole blocks only (partial edge blocks are dropped)."""
    with rasterio.open(src_path) as src:
        t = src.transform
        rows, cols = src.height // block_px, src.width // block_px
        crs = src.crs.to_string()
    transform = Affine(t.a * block_px, 0, t.c, 0, t.e * block_px, t.f)
    return MaskGrid(rows, cols, transform, crs, block_px)


def write_mask(
    path: Path, data: np.ndarray, grid: MaskGrid, tags: dict[str, str] | None = None
) -> Path:
    """Write a coded uint8 mask (nodata 255) as a tiled, compressed GeoTIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff",
        height=grid.rows,
        width=grid.cols,
        count=1,
        dtype="uint8",
        crs=grid.crs,
        transform=grid.transform,
        nodata=fm.NODATA,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    if grid.rows < 256 or grid.cols < 256:
        profile.update(tiled=False)
        profile.pop("blockxsize")
        profile.pop("blockysize")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
        if tags:
            dst.update_tags(**tags)
    return path


def _strips(grid: MaskGrid):
    """Yield ``(row0, nrows, window)`` strips of whole output cells."""
    b = grid.block_px
    step = max(1, STRIP_CELLS if b > 1 else STRIP_CELLS * 16)
    for r0 in range(0, grid.rows, step):
        n = min(step, grid.rows - r0)
        yield r0, n, Window(0, r0 * b, grid.cols * b, n * b)


def chm_cover(
    chm_path: Path, grid: MaskGrid, height_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell share of CHM pixels above ``height_m``, and any-valid flag."""
    cover = np.zeros((grid.rows, grid.cols), "float32" if grid.block_px > 1 else bool)
    valid = np.zeros((grid.rows, grid.cols), bool)
    with rasterio.open(chm_path) as src:
        for r0, n, win in _strips(grid):
            h = src.read(1, window=win).astype("float32")
            if src.nodata is not None:
                h[h == src.nodata] = np.nan
            ok = np.isfinite(h)
            c, v = fm.block_fraction(np.nan_to_num(h) > height_m, ok, grid.block_px)
            cover[r0 : r0 + n], valid[r0 : r0 + n] = c, v
    return cover, valid


def _ndvi_strips(naip_path: Path, grid: MaskGrid):
    """Yield ``(row0, nrows, pixel_ndvi)`` strips (NaN where all bands are 0)."""
    with rasterio.open(naip_path) as src:
        for r0, n, win in _strips(grid):
            rgbn = src.read(window=win)
            nd = fm.ndvi(rgbn[0], rgbn[3])
            nd[(rgbn == 0).all(0)] = np.nan
            yield r0, n, nd


def ndvi_sample(
    naip_path: Path, grid: MaskGrid, where: np.ndarray | None = None, seed: int = 0
) -> np.ndarray:
    """Random sample (about ``SAMPLE_PX``) of block-mean NDVI where ``where`` is true.

    Works strip by strip, so a 0.6 m scene is never held in memory as floats.
    """
    rng = np.random.default_rng(seed)
    frac = min(1.0, SAMPLE_PX / (grid.rows * grid.cols))
    parts = []
    for r0, n, nd in _ndvi_strips(naip_path, grid):
        v = fm.block_mean(nd, grid.block_px)
        ok = np.isfinite(v)
        if where is not None:
            ok &= where[r0 : r0 + n]
        vals = v[ok]
        if frac < 1:
            vals = vals[rng.random(vals.size) < frac]
        parts.append(vals)
    return np.concatenate(parts)


def ndvi_mask(
    naip_path: Path,
    grid: MaskGrid,
    *,
    threshold: float,
    mode: str,
    stats: fm.RobustStats | None,
) -> np.ndarray:
    """Coded NDVI forest mask built strip by strip (no patch cleaning)."""
    out = np.full((grid.rows, grid.cols), fm.NODATA, "uint8")
    for r0, n, nd in _ndvi_strips(naip_path, grid):
        res = fm.ndvi_forest_mask(
            nd, threshold=threshold, mode=mode, stats=stats, block_px=grid.block_px
        )
        out[r0 : r0 + n] = res.mask
    return out


def _clean_coded(coded: np.ndarray, min_patch: int, fill_holes: int) -> np.ndarray:
    if min_patch <= 1 and fill_holes <= 1:
        return coded
    valid = coded != fm.NODATA
    forest = fm.clean_mask(coded == fm.FOREST, min_patch, fill_holes)
    return fm.encode(forest & valid, valid)


def _area_to_cells(area_m2: float, grid: MaskGrid) -> int:
    return int(round(area_m2 / abs(grid.transform.a) ** 2))


def _profile_params(p: ForestMaskProfile) -> dict:
    """Parameters that define an export set (NDVI fields only when used)."""
    d = p.model_dump()
    if p.source == "chm":
        for k in ("ndvi_mode", "ndvi_threshold"):
            d.pop(k)
    if p.block_px == 1:
        d.pop("min_cover")
    return d


def export_forest_masks(
    chm_paths: dict[int, Path],
    naip_paths: dict[int, Path],
    out_dir: Path,
    profile: ForestMaskProfile,
    *,
    reference: np.ndarray | None = None,
    site: str = "",
) -> list[MaskExport]:
    """Export per-year CHM masks, stable-forest summaries and, if asked, NDVI masks.

    CHM masks are always written because they define the NDVI reference
    sample: stable forest (CHM forest in every year) intersected with
    ``reference`` when given (e.g. the interpreter's no-change area).

    Args:
        chm_paths: Year to CHM GeoTIFF.
        naip_paths: Year to NAIP RGBN GeoTIFF on the same grid.
        out_dir: Output directory.
        profile: Mask parameters.
        reference: Optional boolean mask on the output grid that restricts the
            NDVI reference sample.
        site: Site name written into file tags and the index.

    Returns:
        One :class:`MaskExport` per written raster.
    """
    years = sorted(chm_paths)
    grid = mask_grid(chm_paths[years[0]], profile.block_px)
    params = _profile_params(profile)
    tag = fm.params_tag(params)
    min_patch = _area_to_cells(profile.min_patch_m2, grid)
    fill_holes = _area_to_cells(profile.fill_holes_m2, grid)
    if reference is not None and reference.shape != (grid.rows, grid.cols):
        raise ValueError(
            f"reference shape {reference.shape} != grid {grid.rows, grid.cols}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    base_tags = {"site": site, "params": json.dumps(params), "tag": tag}
    exports: list[MaskExport] = []
    sidecar: dict = {"site": site, "tag": tag, "params": params, "years": {}}

    def _name(layer: str, year: int | str) -> Path:
        return out_dir / f"forest_{layer}_{year}_{grid.res_tag}_{tag}.tif"

    chm_masks: dict[int, np.ndarray] = {}
    for year in years:
        t0 = time.time()
        cover, valid = chm_cover(chm_paths[year], grid, profile.height_m)
        forest = cover >= profile.min_cover if grid.block_px > 1 else cover > 0
        forest = fm.clean_mask(forest & valid, min_patch, fill_holes)
        coded = fm.encode(forest, valid)
        chm_masks[year] = coded
        path = write_mask(
            _name("chm", year), coded, grid, base_tags | {"year": str(year)}
        )
        res = fm.ForestMaskResult(coded)
        exports.append(
            MaskExport(path, year, "chm", grid.block_px, tag, res.forest_fraction)
        )
        sidecar["years"].setdefault(str(year), {})[
            "chm_forest_fraction"
        ] = res.forest_fraction
        logger.info(
            "%d CHM mask: %.1f%% forest -> %s (%.0f s)",
            year,
            100 * res.forest_fraction,
            path.name,
            time.time() - t0,
        )

    # Accumulate year by year: stacking six 0.6 m masks needs several GB.
    count = np.zeros((grid.rows, grid.cols), "uint8")
    any_nodata = np.zeros((grid.rows, grid.cols), bool)
    for m in chm_masks.values():
        np.add(count, m == fm.FOREST, out=count, casting="unsafe")
        any_nodata |= m == fm.NODATA
    stable = fm.encode(count == len(years), ~any_nodata)
    count[any_nodata] = fm.NODATA
    del any_nodata
    for layer, arr in (("chm-stable", stable), ("chm-count", count)):
        span = f"{years[0]}-{years[-1]}"
        path = write_mask(_name(layer, span), arr, grid, base_tags | {"years": span})
        frac = (
            fm.ForestMaskResult(stable).forest_fraction
            if layer == "chm-stable"
            else np.nan
        )
        exports.append(MaskExport(path, None, layer, grid.block_px, tag, frac))

    del count
    ref = stable == fm.FOREST
    if reference is not None:
        ref &= reference
    del stable
    if profile.source != "chm":
        for year in years:
            t0 = time.time()
            stats = None
            if profile.ndvi_mode == "z":
                sample = ndvi_sample(naip_paths[year], grid, ref)
                stats = fm.robust_stats(sample, np.ones(sample.shape, bool))
                threshold = float(profile.ndvi_threshold)
            elif profile.ndvi_mode == "otsu":
                threshold = fm.otsu_threshold(ndvi_sample(naip_paths[year], grid))
            else:
                threshold = float(profile.ndvi_threshold)
            coded = ndvi_mask(
                naip_paths[year],
                grid,
                threshold=threshold,
                mode=profile.ndvi_mode,
                stats=stats,
            )
            res = fm.ForestMaskResult(_clean_coded(coded, min_patch, fill_holes))
            del coded
            yinfo = sidecar["years"][str(year)]
            yinfo.update(
                ndvi_threshold=threshold,
                ref_median=None if stats is None else stats.median,
                ref_scale=None if stats is None else stats.scale,
                ndvi_forest_fraction=res.forest_fraction,
            )
            path = write_mask(
                _name("ndvi", year),
                res.mask,
                grid,
                base_tags | {"year": str(year), "ndvi_threshold": f"{threshold:.6f}"},
            )
            exports.append(
                MaskExport(
                    path,
                    year,
                    "ndvi",
                    grid.block_px,
                    tag,
                    res.forest_fraction,
                    threshold,
                )
            )
            if profile.source in ("and", "or"):
                comb = fm.combine_masks(chm_masks[year], res.mask, profile.source)
                cres = fm.ForestMaskResult(comb)
                path = write_mask(
                    _name(profile.source, year),
                    comb,
                    grid,
                    base_tags | {"year": str(year)},
                )
                exports.append(
                    MaskExport(
                        path,
                        year,
                        profile.source,
                        grid.block_px,
                        tag,
                        cres.forest_fraction,
                    )
                )
            logger.info(
                "%d NDVI mask (%s, t=%.4f): %.1f%% forest (%.0f s)",
                year,
                profile.ndvi_mode,
                threshold,
                100 * res.forest_fraction,
                time.time() - t0,
            )

    (out_dir / f"forest_masks_{grid.res_tag}_{tag}.json").write_text(
        json.dumps(sidecar, indent=2, default=float)
    )
    _append_index(out_dir / "masks_index.csv", exports, params, site)
    return exports


def _append_index(
    path: Path, exports: list[MaskExport], params: dict, site: str
) -> None:
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(
                [
                    "file",
                    "site",
                    "year",
                    "layer",
                    "block_px",
                    "tag",
                    "forest_fraction",
                    "threshold",
                    "params",
                    "written",
                ]
            )
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        for e in exports:
            d = asdict(e)
            w.writerow(
                [
                    e.path.name,
                    site,
                    d["year"],
                    e.layer,
                    e.block_px,
                    e.tag,
                    f"{e.forest_fraction:.5f}",
                    e.threshold,
                    json.dumps(params),
                    stamp,
                ]
            )
