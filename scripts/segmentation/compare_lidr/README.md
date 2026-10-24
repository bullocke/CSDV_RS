# Python against lidR: comparison drivers

The scripts behind [the crown segmentation guide](../../../docs/guides/crown_segmentation/README.md): the choice of the flood bound's defaults, the before-and-after comparison, and the comparison with lidR `dalponte2016` in [Python against lidR](../../../docs/guides/crown_segmentation/lidr_comparison.md). The run order and run times are in that page's "Reproducing" section.

| File | Role |
| --- | --- |
| `_lib.py` | Paths, the strata block manifest, NEON plot inputs, crown and fit metrics |
| `strata_blocks.csv` | The 36 NAIP-CHM benchmark blocks in 12 CONUS strata, with RESOLVE biome |
| `fetch_units.py` | NAIP-CHM and NEON lidar chips of the 626 NEON field plots, and the 16 NEON lidar blocks |
| `before_after.py` | Flood defaults by a pre-registered rule, then the earlier bound, the flood and lidR on plots and strata |
| `strata.py` | Runs Python parameter sets and lidR on the strata blocks, and draws maps and CDFs |
| `surfaces.py` | Identical smoothed inputs for both engines, the Python growers, and label metrics |
| `compare.py` | Detection, growth from shared tree tops, constructed scenes, speed and memory, crown AGB |
| `bench_lidr.R` | Times lidR `locate_trees` and `dalponte2016` over a manifest with packages loaded once |
| `figures.py` | The guide's figures |

Inputs outside the repository: the refAGB methods-paper repository (`REFAGB_METHODS_DIR`) for the NEON plots and field trees, Earth Engine access for the NEON lidar CHM, and the NTSG NAIP-CHM server for tiles.
