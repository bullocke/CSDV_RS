# Crown segmentation

The crown segmentation engine, its parameters, the crown allometry and the national NAIP-CHM tooling moved to the sibling repository [naip-crowns](https://github.com/bullocke/naip-crowns) (package `naipcrowns`) in October 2026. Its documentation covers [the engine](https://github.com/bullocke/naip-crowns/blob/main/docs/guide/segmentation.md), [the algorithm](https://github.com/bullocke/naip-crowns/blob/main/docs/guide/algorithm.md), [Python against lidR](https://github.com/bullocke/naip-crowns/blob/main/docs/studies/lidr_comparison.md) and [the national feasibility study](https://github.com/bullocke/naip-crowns/blob/main/docs/studies/national_feasibility.md).

CSDV uses the engine in two places. `csdv segment-crowns` is `naipcrowns segment` under its CSDV name, and the stand-level crown statistics in `csdv_core.zonal.crowns` read the crowns it writes. Install naipcrowns next to this repository before running either:

```bash
git clone https://github.com/bullocke/naip-crowns.git ../naip-crowns
.micromamba/envs/CSDV/bin/python -m pip install --no-deps --no-build-isolation -e ../naip-crowns
```

`csdv check` reports when it is missing.

This folder keeps one study that belongs to CSDV: the [Indiana parameter optimization](optimization/README.md), which tuned the engine for CSDV's stand metrics on Elkinsville NAIP-CHM and two lidar sites (2026, earlier engine). Its scripts are in `scripts/segmentation/`.
