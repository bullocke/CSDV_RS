"""Identical inputs for both engines, Python label growth, and label metrics.

A *surface* is one smoothed, masked CHM block that both engines read, so any
difference between them comes from the algorithms. Python smooths and masks
it once (``smooth_chm`` and ``mask_below``, as ``segment_crowns`` does) and
writes it as float32 GeoTIFF with NaN outside the canopy, and lidR reads that
file with no smoothing of its own.

NAIP-CHM surfaces are the 36 strata blocks (0.6 m) with their halo. Lidar
surfaces are the 1024 m NEON blocks from ``fetch_units.py blocks`` (1 m), with
a margin of one halo inside the block edge treated as the halo.

Metrics are taken on the interior only: crowns whose tree top lies in it and
canopy pixels in it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import _lib as L
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from csdv_core.preprocess.chm import mask_below, smooth_chm
from csdv_core.segmentation.chm_watershed import (
    _bound_extent,
    locate_seeds,
    smooth_kernel_px,
)
from csdv_core.segmentation.flood import grow_crowns
from csdv_core.segmentation.params import SegmentationParams
from csdv_core.zonal.crowns import default_overlap_px

#: Lidar blocks with less canopy than this are skipped (JORN).
MIN_LIDAR_CANOPY = 0.01


@dataclass
class Surface:
    """One prepared block."""

    name: str
    kind: str  # "naip" or "lidar"
    biome: str
    smoothed: np.ndarray
    transform: object
    crs: object
    inner: tuple[slice, slice]
    path: Path | None = None

    @property
    def px(self) -> float:
        return float(abs(self.transform.a))

    @property
    def interior(self) -> np.ndarray:
        m = np.zeros(self.smoothed.shape, dtype=bool)
        m[self.inner] = True
        return m


def prepare(chm: np.ndarray, px: float, params: SegmentationParams) -> np.ndarray:
    """Smooth and mask exactly as ``segment_crowns`` does."""
    kernel = smooth_kernel_px(params.smooth_radius_m, px)
    smoothed = smooth_chm(np.asarray(chm, dtype="float32"), kernel=kernel)
    return mask_below(smoothed, threshold_m=params.min_height_m)


def naip_surfaces(params_for, halo_m: float = 30.0) -> list[Surface]:
    """The strata blocks. ``params_for(biome)`` gives the set to smooth with."""
    out = []
    blocks = L.strata_blocks()
    for i, row in blocks.iterrows():
        p = params_for(row["biome"])
        halo = max(default_overlap_px(p, 0.6), int(np.ceil(halo_m / 0.6)))
        b = L.read_block(row, halo)
        out.append(
            Surface(f"{row['stratum']}#{i % 3}", "naip", row["biome"],
                    prepare(b.chm, 0.6, p), b.transform, b.crs, b.inner)  # fmt: skip
        )
    return out


def lidar_surfaces(params_for) -> list[Surface]:
    """The NEON lidar blocks, with a halo-wide margin as the outside."""
    table = pd.read_csv(L.TABLE_DIR / "lidar_blocks.csv")
    out = []
    for row in table.itertuples():
        if row.canopy_frac < MIN_LIDAR_CANOPY:
            continue
        p = params_for(row.biome)
        with rasterio.open(L.DATA_DIR / "lidar_blocks" / f"{row.site}.tif") as src:
            chm = src.read(1).astype("float32")
            transform, crs = src.transform, src.crs
        chm[~np.isfinite(chm)] = np.nan
        halo = default_overlap_px(p, 1.0)
        n = chm.shape[0]
        inner = (slice(halo, n - halo), slice(halo, chm.shape[1] - halo))
        out.append(
            Surface(
                row.site,
                "lidar",
                row.biome,
                prepare(chm, 1.0, p),
                transform,
                crs,
                inner,
            )
        )
    return out


def write_surface(s: Surface, path: Path) -> Path:
    profile = {
        "driver": "GTiff", "width": s.smoothed.shape[1], "height": s.smoothed.shape[0],
        "count": 1, "dtype": "float32", "crs": s.crs, "transform": s.transform,
        "nodata": np.nan, "compress": "deflate",
    }  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(s.smoothed.astype("float32"), 1)
    s.path = path
    return path


# ---------------------------------------------------------------------------
# Seeds and labels
# ---------------------------------------------------------------------------
def python_seeds(
    s: Surface, params: SegmentationParams
) -> tuple[np.ndarray, np.ndarray]:
    valid = ~np.isnan(s.smoothed)
    return locate_seeds(
        s.smoothed,
        valid,
        s.px,
        params.window,
        min_separation_m=params.min_separation_m,
        window_metric=params.window_metric,
    )


def seeds_to_xy(s: Surface, rows: np.ndarray, cols: np.ndarray) -> pd.DataFrame:
    x, y = s.transform * (cols + 0.5, rows + 0.5)
    return pd.DataFrame({"x": x, "y": y, "z": s.smoothed[rows, cols]})


def xy_to_rc(s: Surface, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inv = ~s.transform
    c, r = inv * (np.asarray(x), np.asarray(y))
    return np.floor(r).astype(int), np.floor(c).astype(int)


def python_labels(
    s: Surface,
    rows: np.ndarray,
    cols: np.ndarray,
    params: SegmentationParams,
    mode: str,
) -> np.ndarray:
    """Crown labels (1..n in seed order) from one of the Python growers.

    ``flood_apex`` and ``flood_mean`` are the flood bound under each rule;
    ``post_trim`` is the earlier watershed-then-trim engine.
    """
    h = s.smoothed
    apex = h[rows, cols].astype("float64")
    radius = (
        None if params.max_crown_radius_m is None else params.max_crown_radius_m / s.px
    )
    if mode in ("flood_apex", "flood_mean"):
        rule = "apex" if mode == "flood_apex" else "mean"
        th_seed = params.th_seed if (rule == "apex" or params.th_seed) else 0.45
        labels, _ = grow_crowns(
            h, rows, cols, apex, th_cr=params.th_cr, th_seed=th_seed,
            min_crown_depth_m=params.min_crown_depth_m if rule == "apex" else None,
            max_radius_px=radius, rule=rule,
        )  # fmt: skip
        return labels
    if mode == "post_trim":
        valid = ~np.isnan(h)
        markers = np.zeros(h.shape, dtype="int32")
        markers[rows, cols] = np.arange(1, rows.size + 1)
        ws = watershed(
            np.where(valid, -h, np.inf).astype("float32"), markers=markers, mask=valid
        )
        labels, _ = _bound_extent(
            ws, h, rows, cols, apex, th_cr=params.th_cr, max_radius_px=radius
        )
        return labels
    raise ValueError(mode)


def read_labels(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype("int32")


# ---------------------------------------------------------------------------
# Label metrics
# ---------------------------------------------------------------------------
def crown_table(
    s: Surface, labels: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> pd.DataFrame:
    """One row per crown whose tree top lies in the interior."""
    n = rows.size
    area_px = np.bincount(labels.ravel(), minlength=n + 1)[1 : n + 1]
    inside = s.interior[rows, cols] & (area_px > 0)
    area = area_px * s.px * s.px
    return pd.DataFrame(
        {
            "label": np.arange(1, n + 1)[inside],
            "apex_h_m": s.smoothed[rows, cols][inside],
            "area_m2": area[inside],
            "crown_diam_m": 2.0 * np.sqrt(area[inside] / np.pi),
        }
    )


def canopy_assigned(s: Surface, labels: np.ndarray) -> float:
    canopy = s.interior & ~np.isnan(s.smoothed)
    return float((labels[canopy] > 0).mean()) if canopy.any() else np.nan


def _edges(labels: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    a = labels[:, :-1] if axis == 1 else labels[:-1, :]
    b = labels[:, 1:] if axis == 1 else labels[1:, :]
    shared = (a > 0) & (b > 0) & (a != b)
    return shared, np.minimum(a, b).astype(np.int64) * 10_000_000 + np.maximum(a, b)


def valley_score(s: Surface, labels: np.ndarray, apex_by_label: np.ndarray) -> float:
    """Median over crown pairs of boundary height over the shorter apex.

    Boundaries that follow the valleys between crowns sit low relative to the
    shorter crown. Boundaries set by distance alone sit higher, part way up a
    crown flank.
    """
    h = np.nan_to_num(s.smoothed)
    vals = []
    for axis in (0, 1):
        shared, pair = _edges(labels, axis)
        a = labels[:, :-1] if axis == 1 else labels[:-1, :]
        b = labels[:, 1:] if axis == 1 else labels[1:, :]
        ha = h[:, :-1] if axis == 1 else h[:-1, :]
        hb = h[:, 1:] if axis == 1 else h[1:, :]
        inside = (s.interior[:, :-1] if axis == 1 else s.interior[:-1, :]) & shared
        shorter = np.minimum(apex_by_label[a[inside]], apex_by_label[b[inside]])
        vals.append(
            pd.DataFrame(
                {"pair": pair[inside], "r": 0.5 * (ha[inside] + hb[inside]) / shorter}
            )
        )
    d = pd.concat(vals)
    return float(d.groupby("pair")["r"].mean().median()) if len(d) else np.nan


def straight_edge_share(
    s: Surface, labels: np.ndarray, min_run_m: float = 3.0
) -> float:
    """Share of shared crown boundary in straight axis-aligned runs of at least
    ``min_run_m``. Growth by distance alone draws straight edges where two
    crowns meet; boundaries that follow the CHM rarely run straight."""
    min_run = max(2, int(round(min_run_m / s.px)))
    total, straight = 0, 0
    for axis in (0, 1):
        shared, pair = _edges(labels, axis)
        inside = (s.interior[:, :-1] if axis == 1 else s.interior[:-1, :]) & shared
        p = np.where(inside, pair, 0)
        # Runs run across the boundary direction: along rows for vertical
        # edges (axis 1), along columns for horizontal edges (axis 0).
        seq = p.T if axis == 1 else p
        flat = np.concatenate(
            [seq, np.zeros((seq.shape[0], 1), dtype=seq.dtype)], axis=1
        ).ravel()
        change = np.flatnonzero(np.diff(np.concatenate([[0], flat])) != 0)
        starts = change
        ends = np.append(change[1:], flat.size)
        values = flat[starts]
        lengths = ends - starts
        edge_runs = values != 0
        total += int(lengths[edge_runs].sum())
        straight += int(lengths[edge_runs & (lengths >= min_run)].sum())
    return straight / total if total else np.nan


def compactness(labels: np.ndarray, table: pd.DataFrame) -> float:
    """Median Polsby-Popper compactness, 4 pi A / P^2, on pixel edges."""
    lab = labels
    perim = np.zeros(int(lab.max()) + 1, dtype=np.int64)
    padded = np.pad(lab, 1)
    core = padded[1:-1, 1:-1]
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nb = padded[
            1 + dr : padded.shape[0] - 1 + dr, 1 + dc : padded.shape[1] - 1 + dc
        ]
        edge = (core > 0) & (nb != core)
        perim += np.bincount(core[edge], minlength=perim.size)[: perim.size]
    area = np.bincount(lab.ravel(), minlength=perim.size)
    ids = table["label"].to_numpy()
    pp = 4.0 * np.pi * area[ids] / np.maximum(perim[ids], 1) ** 2
    return float(np.median(pp)) if len(pp) else np.nan


def matched_iou(a: np.ndarray, b: np.ndarray, ids: np.ndarray) -> float:
    """Median IoU of crowns with the same label (same seed) in two label maps."""
    n = int(max(a.max(), b.max())) + 1
    inter = np.bincount(a[(a > 0) & (a == b)], minlength=n)
    area_a = np.bincount(a.ravel(), minlength=n)
    area_b = np.bincount(b.ravel(), minlength=n)
    union = area_a + area_b - inter
    iou = inter[ids] / np.maximum(union[ids], 1)
    return float(np.median(iou)) if len(iou) else np.nan


def plateau_of(h: np.ndarray, r: int, c: int) -> np.ndarray:
    """Pixels in the 8-connected plateau of equal height containing (r, c)."""
    r0, r1 = max(0, r - 15), min(h.shape[0], r + 16)
    c0, c1 = max(0, c - 15), min(h.shape[1], c + 16)
    win = h[r0:r1, c0:c1] == h[r, c]
    lab, _ = ndi.label(win, structure=np.ones((3, 3)))
    rr, cc = np.nonzero(lab == lab[r - r0, c - c0])
    return np.column_stack([rr + r0, cc + c0])
