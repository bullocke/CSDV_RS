# Crown segmentation

`csdv_core.segmentation` delineates tree crowns from a canopy height model (CHM). It finds tree tops with a variable-window local maximum filter and grows one crown from each with a priority flood, stopping each crown at a height floor tied to its own tree top. It is written in Python with NumPy, SciPy, scikit-image and Numba, needs no R, and every parameter is in metres, so one parameter set means the same thing on a 0.6 m NAIP-derived CHM and a 1 m lidar CHM.

The engine follows the rules of lidR's `dalponte2016` (Dalponte and Coomes 2016, Roussel et al. 2020) closely enough to be compared with it parameter for parameter, and departs from it where its implementation produces artefacts. [Python against lidR](lidr_comparison.md) measures both on the same inputs.

On the same inputs, the two engines find the same tree tops (99.4 to 99.9 percent at the same pixel) and grow different crowns from them. Python detects tree tops and grows crowns 8 to 22 times faster on one thread, and its growth time does not depend on the crown radius cap. lidR's does, because it makes one full-image pass per ring of growth. Crown AGB from the two engines agrees within the uncertainty of the allometry. On NEON lidar CHMs, crown AGB beats plot mean height as a predictor of field AGB, and on the NAIP-CHM it does not.

## Contents

| Page | What it covers |
| --- | --- |
| [The algorithm](algorithm.md) | Each step, the parameters and their units, presets, tiling, and use from Python and the command line |
| [Python against lidR](lidr_comparison.md) | Tree detection, crown growth, speed and crown AGB, both engines on the same NAIP-CHM and NEON lidar inputs |
| [Optimization study](optimization/README.md) | How the default parameters were chosen on Indiana NAIP-CHM and checked on two lidar sites (2026, earlier engine) |

## Quick start

```python
import rasterio
from csdv_core.io.raster import read_chm_metres
from csdv_core.segmentation.chm_watershed import segment_crowns
from csdv_core.segmentation.params import DEFAULT_PARAMS

with rasterio.open("chm.tif") as src:
    chm = read_chm_metres(src, 1.0)  # metres. Use 0.01 for the uint16 NAIP-CHM
    crowns = segment_crowns(chm, src.transform, src.crs, params=DEFAULT_PARAMS)

crowns.to_parquet("crowns.parquet")
```

```bash
csdv segment-crowns --chm chm.tif --out-crowns crowns.parquet
csdv segment-crowns --chm chm.tif --out-crowns crowns.parquet --tallo-biome "Temperate Conifer Forests"
```

Each crown carries its area, area-equivalent diameter, apex height, tree-top position and `capped_frac`, the share of its boundary set by the crown radius cap. Large rasters are processed in blocks with a halo, and blocked and single-pass runs give the same crowns.

## What the crowns are

On the national NAIP-CHM the crowns are canopy patches, not individual trees. At 14 NEON sites the engine finds about one crown per 3.5 field canopy trees, and no parameter setting resolves more, because the model-inferred CHM is smooth enough that the crowns of subdominant trees never appear in it ([optimization study](optimization/README.md), and the national feasibility analysis). On airborne lidar CHMs the same engine separates many more crowns. Read crown statistics from the NAIP-CHM as canopy-patch structure.
