# The crown segmentation algorithm

This page describes what the Python engine computes, step by step, and the choices behind each step. The code is in `src/csdv_core/segmentation/`: `chm_watershed.py` runs the pipeline, `flood.py` grows the crowns, and `params.py` holds every parameter. The comparison with lidR is in [Python against lidR](lidr_comparison.md).

The engine is a seeded region-growing segmentation of a canopy height model (CHM) in the family of Hyyppä et al. (2001) and Dalponte and Coomes (2016). It finds tree tops as local maxima of a smoothed CHM, grows one crown from each, and stops each crown at a height floor tied to its own tree top. Every parameter is in metres, so one parameter set means the same thing on a 0.6 m NAIP-derived CHM and a 1 m lidar CHM.

## Pipeline

| Step | Function | What it does |
| --- | --- | --- |
| 1. Smooth | `preprocess.chm.smooth_chm` | Square mean filter, side `2 round(smooth_radius_m / px) + 1` pixels |
| 2. Mask | `preprocess.chm.mask_below` | Pixels below `min_height_m` leave the canopy |
| 3. Tree tops | `chm_watershed.locate_seeds` | Variable-window local maximum filter |
| 4. Crowns | `flood.grow_crowns` | Priority flood from all tree tops, each crown bounded by its own floor |
| 5. Polygons | `chm_watershed.segment_crowns` | One polygon per crown with area, diameter, apex height and tree-top position |

### 1. Smoothing

A mean filter with a square kernel removes the small maxima that texture produces on a crown surface. The kernel side is derived from a radius in metres, `2 round(smooth_radius_m / px) + 1`, so the default radius of 0.6 m gives a 3 x 3 kernel at 0.6 m and also at 1 m. NaN pixels are left out of each local mean, as `terra::focal(na.rm = TRUE)` does, and the raster edge is reflected. A radius of zero disables smoothing.

### 2. Canopy floor

Pixels of the smoothed CHM below `min_height_m` become NaN. They take no part in tree-top detection or crown growth.

### 3. Tree tops

A pixel is a tree top when no pixel within `ws(h) / 2` metres of it is higher, where `h` is that pixel's own smoothed height and `ws` is the search-window diameter. This is the variable-window local maximum filter of Popescu and Wynne (2004) as lidR's `lmf()` implements it, with a circular window. The window grows with height because taller trees have wider crowns, so a fixed window either splits large crowns or merges small ones.

Two window forms are available. Both are diameters in metres, clipped to `[lo, hi]`:

| Form | Window | Use |
| --- | --- | --- |
| `poly` | `ws(h) = k (a + b h + c h²)` | Fixed windows (`b = c = 0`), Popescu and Wynne's crown-width curve, linear rules |
| `power` | `ws(h) = k a h^b` | A crown-width allometry `CW = a h^b` scaled by `k`, as in the Tallo windows below |

Distances are exact by default (`window_metric="exact"`): a pixel lies in the window when the distance between pixel centres is at most the radius. The earlier engine rounded the radius to whole pixels first (`window_metric="pixel"`). Rounding is not harmless. `np.rint` rounds halves to even, so at 1 m a 3 m window (radius 1.5 px) and a 5 m window (radius 2.5 px) both became the same 2 px disk.

A pixel that ties with an equal neighbour is kept, not rejected, and each plateau of tied maxima is collapsed to one tree top at its centre of mass, or at its highest pixel when the centre of mass falls off an irregular plateau. The radius never drops below `min_separation_m` (default 1 m), so one flat crown top does not become several trees.

Every tree top is also the highest pixel of its 4-neighbourhood, because every window contains that cross. The detector therefore finds those candidates with one 3 x 3 maximum filter and then checks each candidate against its own disk in compiled code. The result is identical to testing every pixel against its own window, and the cost is one filter plus a disk check per candidate, whatever the number of distinct window radii.

### 4. Crown growth

All crowns grow at once in a priority flood (Meyer 1991). A heap holds the pixels at the edge of every crown, ordered by height, highest first, with ties broken by insertion order. The highest pixel is taken from the heap and joins the crown that reached it, unless another crown has claimed it already. Its four neighbours are then offered to that crown, and a neighbour enters the heap only when it passes the crown's admission test:

1. It is canopy (not NaN).
2. It is no higher than `1.05` times the crown's apex, the ceiling lidR also applies.
3. It lies within `max_crown_radius_m` of the crown's tree top, measured as a circle.
4. It is at or above the crown's floor.

Under the default apex rule, the floor depends only on the crown's apex height `H`:

```text
floor = max( th_seed H,  min( th_cr H,  H - min_crown_depth_m ) )
```

raised to `H - max_crown_depth_m` when a maximum depth is set. With the defaults (`th_cr = 0.70`, no `th_seed`, no depth bounds), a 20 m tree keeps pixels down to 14 m. `min_crown_depth_m` exists because a fixed fraction of height is a thin slice of a short tree: at `th_cr = 0.70` a 6 m tree keeps only 1.8 m of crown. `max_crown_depth_m` is the absolute height difference that Dalponte and Coomes (2016) describe alongside the relative one.

A pixel one crown rejects stays free, so a crown that reaches it later can still take it. That is the difference from the earlier engine (`bound_mode="post_trim"`), which first ran an unconstrained watershed, so that every canopy pixel belonged to some crown, and then trimmed each crown to its own floor. In that order the tallest tree of a clump floods down into the saddles it shares with shorter trees, the trim removes those pixels against the tall tree's floor, and the shorter trees whose floors they pass never see them. On twelve NAIP-CHM strata that left 12 to 38 percent of the canopy unassigned for this reason alone.

Growth is 4-connected and starts at the tree top, so every crown is one connected region that contains its tree top. Because the heap releases pixels in height order, a boundary between two crowns forms where their flood fronts meet, which is the lowest path between them. Each pixel is labelled once and pushed at most four times per neighbouring crown, so the cost is O(N log N) in the number of canopy pixels and does not depend on the crown radius cap.

The `mean` rule (`th_cr_ref="mean"`) replaces the floor test with lidR's: a pixel must exceed `th_seed` times the apex and `th_cr` times the crown's current mean height. It exists to separate the effect of the rule from the effect of lidR's implementation and is not recommended for production. The crown mean falls as a crown takes lower pixels, so the bar falls with it, and a crown keeps growing down its flanks and across low canopy until `th_seed` or the radius cap stops it.

### 5. Polygons and attributes

Crown labels are converted to polygons with `rasterio.features.shapes` at 4-connectivity, so each crown is one polygon. Each crown carries:

| Column | Meaning |
| --- | --- |
| `segment_id` | Crown label |
| `area_m2` | Polygon area |
| `crown_diam_m` | Area-equivalent diameter, `2 sqrt(area / pi)` |
| `apex_h_m` | Smoothed height at the tree top |
| `seed_x`, `seed_y` | Tree-top position |
| `capped_frac` | Share of the crown's boundary set by the radius cap rather than by its floor, a gap or a neighbour |

Apex height is read at the tree top, never as the maximum over the crown, because a maximum over a region grows with the region's size and would build a height-to-size relationship into the measurement. Crowns smaller than `min_crown_area_m2` are dropped. A crown with a large `capped_frac` has a censored diameter, a lower bound rather than a measurement, and should be excluded from allometric fits.

## Large rasters

`zonal.crowns.segment_scene_crowns` processes a scene in blocks (2048 px by default). Each block is read with a halo of `ceil(2 max_crown_radius_m / px)` pixels plus the smoothing kernel, so that crowns on the block edge and their neighbours are complete, and a crown is kept by the block whose interior holds its polygon centroid. Blocked and single-pass runs give the same crowns, which a unit test checks.

Tiles of the national NAIP-CHM overlap their neighbours by about 160 m. `zonal.crowns.clip_to_cell` keeps a crown in the tile whose nominal quarter-quad cell holds its tree top, with half-open cell edges, so each crown is counted once.

The result does not depend on the run: the flood breaks height ties by insertion order, and tree-top detection has no order dependence.

## Parameters

| Parameter | Default | Unit | Meaning | lidR counterpart |
| --- | --- | --- | --- | --- |
| `min_height_m` | 2.0 | m | Canopy floor | `lmf(hmin)`, `dalponte2016(th_tree)` |
| `smooth_radius_m` | 0.6 | m | Mean-filter radius, 0 disables | none (smooth before) |
| `window` | fixed 5 m | m | Search-window diameter as a function of height | `lmf(ws)` |
| `window_metric` | `exact` | | Exact distances or radius rounded to pixels | exact |
| `th_cr` | 0.70 | fraction | Crown floor relative to the apex | `th_cr`, relative to the crown mean |
| `th_cr_ref` | `apex` | | What `th_cr` is relative to | `mean` |
| `th_seed` | 0 | fraction | Floor relative to the apex | `th_seed` (default 0.45) |
| `min_crown_depth_m` | none | m | Least vertical extent kept for every crown | none |
| `max_crown_depth_m` | none | m | Greatest vertical extent of any crown | none |
| `max_crown_radius_m` | 12 | m | Circular radius cap | `max_cr`, in pixels, a square half-width |
| `min_crown_area_m2` | 1 | m² | Crowns smaller than this are dropped | none |
| `min_separation_m` | 1 | m | Least tree-top separation | none |
| `bound_mode` | `flood` | | `flood`, or `post_trim` for the earlier engine | |

`SegmentationParams.key` is a short hash of every value. Crown files are written under that key, so a run cannot reuse crowns produced by different parameters. Fields added after the optimization study are left out of the hash while they hold the value that reproduces the earlier engine, so keys recorded by that study still name the same crowns.

## Presets

| Name | Bound | Window | `th_cr` | Cap | Origin |
| --- | --- | --- | --- | --- | --- |
| `DEFAULT_PARAMS` | flood, exact windows | fixed 5 m | 0.70 | 12 m | The optimization study's set, moved to the flood bound |
| `OPTIMIZED_POST_TRIM_PARAMS` | post-trim, rounded windows | fixed 5 m | 0.70 | 12 m | Reproduces the [optimization study](optimization/README.md) |
| `tallo_params(biome)` | flood, exact windows | Tallo, per biome | 0.55 | 15 m | The refAGB production set |

`DEFAULT_PARAMS` keeps `th_cr = 0.70`. A rule fixed before the comparison chose it among `th_cr` from 0.50 to 0.70 and minimum depths of 0, 2 and 3 m on 491 NEON calibration plots: no set enlarged crowns less than the earlier engine, and the fallback, the calibration R² of Tallo crown AGB, moved only from 0.333 to 0.340 across the grid. Lower values assign more canopy at no cost in AGB skill. At `th_cr = 0.55` the median plot puts 95 percent of its canopy in a crown, against 86 percent at 0.70. The selection is in `scripts/segmentation/compare_lidr/before_after.py`.

`tallo_params(biome)` reproduces the lidR settings of the refAGB project in metres: `hmin = th_tree = 1` m, `th_seed = 0.45`, `th_cr = 0.55`, a 15 m crown cap, a 1 m smoothing radius (the 3 x 3 filter refAGB applies at 1 m), and the window `w(h) = clip(0.5 a h^b, 3, 25)` m with `a` and `b` fitted per RESOLVE biome to the Tallo database (Jucker et al. 2022). Its `th_cr` is apex-relative, unlike refAGB's lidR runs. Pass `th_cr_ref="mean"` for lidR's rule.

| RESOLVE biome | `a` | `b` | Trees in fit | Window exceeds 3 m above | Window at 30 m |
| --- | --- | --- | --- | --- | --- |
| Temperate Broadleaf & Mixed Forests | 0.962 | 0.630 | 76,119 | 18.3 m | 4.1 m |
| Temperate Conifer Forests | 0.905 | 0.545 | 16,329 | 32.1 m | 3.0 m |
| Temperate Grasslands, Savannas & Shrublands | 0.727 | 0.757 | 6,248 | 16.3 m | 4.8 m |
| Deserts & Xeric Shrublands | 2.057 | 0.363 | 744 | 19.1 m | 3.5 m |
| Mediterranean Forests, Woodlands & Scrub | 1.880 | 0.467 | 76,833 | 12.0 m | 4.6 m |

With `k = 0.5`, the 3 m floor binds below 12 to 32 m, so for most trees the Tallo window is a fixed 3 m. Coefficients for eight more biomes are in `segmentation/coefs/tallo_window_coefs.csv`.

## Use

```python
import rasterio
from csdv_core.io.raster import read_chm_metres
from csdv_core.segmentation.chm_watershed import segment_crowns
from csdv_core.segmentation.params import DEFAULT_PARAMS, tallo_params

with rasterio.open("chm.tif") as src:
    chm = read_chm_metres(src, 0.01)  # uint16 centimetres, as the NAIP-CHM
    crowns = segment_crowns(chm, src.transform, src.crs, params=DEFAULT_PARAMS)

crowns_tallo = segment_crowns(
    chm, src.transform, src.crs, params=tallo_params("Temperate Conifer Forests")
)
```

From the command line, `csdv segment-crowns` takes the same parameters (`--th-cr`, `--min-crown-depth-m`, `--window`, `--tallo-biome` and others) and writes GeoPackage or, for a `.parquet` suffix, GeoParquet. `--engine lidr` runs lidR through `segmentation/scripts/crown_segmentation.R` with the same parameters, as an independent check.

The flood and the tree-top check are compiled with Numba (`numba` in `environment.yml`). On CHPC, add it to an existing environment with `mamba install -n <env> -c conda-forge numba`. The first call in a process compiles the kernels and caches them next to the module, which takes a few seconds once.

## References

Dalponte, M. and Coomes, D. A. (2016). Tree-centric mapping of forest carbon density from airborne laser scanning and hyperspectral data. *Methods in Ecology and Evolution* 7, 1236-1245.

Hyyppä, J., Kelle, O., Lehikoinen, M. and Inkinen, M. (2001). A segmentation-based method to retrieve stem volume estimates from 3-D tree height models produced by laser scanners. *IEEE Transactions on Geoscience and Remote Sensing* 39, 969-975.

Jucker, T. et al. (2022). Tallo: A global tree allometry and crown architecture database. *Global Change Biology* 28, 5254-5268.

Meyer, F. (1991). Un algorithme optimal pour la ligne de partage des eaux. *8ème Congrès de Reconnaissance des Formes et Intelligence Artificielle*, Lyon, 847-857.

Popescu, S. C. and Wynne, R. H. (2004). Seeing the trees in the forest: using lidar and multispectral data fusion with local filtering and variable window size for estimating tree height. *Photogrammetric Engineering and Remote Sensing* 70, 589-604.

Roussel, J.-R. et al. (2020). lidR: An R package for analysis of Airborne Laser Scanning (ALS) data. *Remote Sensing of Environment* 251, 112061.
