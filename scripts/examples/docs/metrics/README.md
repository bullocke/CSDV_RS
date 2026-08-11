# Metric example figures

## What this is

Each script here builds one figure for one metric in [`docs/metrics.md`](../../../../docs/metrics.md). The figures show a metric changing on a real stand, the imagery and canopy height behind the change, and where the value falls against the stage envelopes. `_lib.py` holds what the scripts share so a new metric is mostly layout and a caption.

Figures are written to `docs/images/metrics/` and are committed, so `docs/metrics.md` renders for anyone who clones the repository.

## How to run

Each script declares its figures as named presets, so every figure in the docs can be rebuilt by name.

```bash
P=.micromamba/envs/CSDV/bin/python
$P scripts/examples/docs/metrics/gap_fraction.py                    # clearcut
$P scripts/examples/docs/metrics/gap_fraction.py --preset closure
$P scripts/examples/docs/metrics/crown_cv.py                        # stable
$P scripts/examples/docs/metrics/crown_cv.py --preset sparse
$P scripts/examples/docs/metrics/gap_persistence.py
$P scripts/examples/docs/metrics/height_mean.py                     # clearcut
$P scripts/examples/docs/metrics/height_mean.py --preset reserves
$P scripts/examples/docs/metrics/glcm_texture.py
$P scripts/examples/docs/metrics/edge_density.py                    # clearcut
$P scripts/examples/docs/metrics/edge_density.py --preset selection
$P scripts/examples/docs/metrics/row_directionality.py              # clearcut
$P scripts/examples/docs/metrics/row_directionality.py --preset scale
$P scripts/examples/docs/metrics/ndvi_mean.py
$P scripts/examples/docs/metrics/ndvi_seasonal_amplitude.py
$P scripts/examples/docs/metrics/ndvi_trend.py
```

Scripts anchor every path to their own location, so the working directory does not matter.

| Preset | Stand | Years | Output |
|---|---|---|---|
| `gap_fraction.py --preset clearcut` | `ELKNE-U9-0-0` | 2016, 2018, 2022 | `gap_fraction.png` |
| `gap_fraction.py --preset closure` | `ELKNE-U4-0-0` | 2016, 2020, 2022 | `gap_fraction_closure.png` |
| `crown_cv.py --preset stable` | `ELKNE-U44-0-0` | 2016, 2018, 2022 | `crown_cv_stable.png` |
| `crown_cv.py --preset sparse` | `ELKNE-U13-0-0` | 2016, 2018, 2022 | `crown_cv_sparse.png` |
| `gap_persistence.py --preset clearcut` | `ELKNE-U9-0-0` | 2018, 2020, 2022 | `gap_persistence.png` |
| `height_mean.py --preset clearcut` | `ELKNE-U13-0-0` | 2016, 2018, 2022 | `height_mean.png` |
| `height_mean.py --preset reserves` | `ELKNE-U9-0-0` | 2016, 2018, 2022 | `height_mean_reserves.png` |
| `glcm_texture.py --preset selection` | `ELKNE-U47-0-0` | 2016, 2018, 2022 | `glcm_texture.png` |
| `edge_density.py --preset clearcut` | `ELKNE-U12-0-0` | 2016, 2018, 2022 | `edge_density.png` |
| `edge_density.py --preset selection` | `ELKNE-U47-0-0` | 2016, 2018, 2022 | `edge_density_selection.png` |
| `row_directionality.py --preset clearcut` | `ELKNE-U13-0-0` | 2016, 2018, 2022 | `row_directionality.png` |
| `row_directionality.py --preset scale` | `ELKNE-U13-0-0` | 2016 | `row_directionality_scale.png` |
| `ndvi_mean.py --preset clearcut` | `ELKNE-U16-0-0a` | 2012, 2016, 2022 | `ndvi_mean.png` |
| `ndvi_seasonal_amplitude.py --preset clearcut` | `ELKNE-U16-0-0a` | 2005, 2016, 2025 | `ndvi_seasonal_amplitude.png` |
| `ndvi_trend.py --preset clearcut` | `ELKNE-U16-0-0a` | 2010, 2016, 2020 | `ndvi_trend.png` |

`--years` does not always mean NAIP dates. On `gap_persistence.py` each year names a pair, so the 2018 column draws the 2016 and 2018 canopy height models together. On the three satellite scripts the years are Landsat years and are unrelated to the NAIP record, except on `ndvi_mean.py`, where the context row is NAIP.

Every script takes the same options. Anything given explicitly overrides the preset.

| Option | Default | Purpose |
|---|---|---|
| `--preset` | first in the table | Named figure to build |
| `--stand-id` | from the preset | Which stand to render |
| `--years` | from the preset | Comma separated years for the image columns |
| `--detail-m` | from the preset | Crop the image row to a square box of this size, centred on the stand |
| `--pad-fraction` | `0.12` | Chip padding as a fraction of the stand's longer side |
| `--min-pad-m` | `25.0` | Smallest chip padding in metres |
| `--max-px` | `500` | Cap on the longer side of each chip read |
| `--dpi` | `150` | Output resolution |
| `--out` | from the preset | Where to write |
| `--no-optimize` | off | Skip the palette re-encode |

To retune a zoom, change `--pad-fraction`, or `--detail-m` on a stand large enough to need a crop. To try another stand, change `--stand-id` and `--years` together, because the interesting years differ per stand.

## Data this depends on

The stand pipeline must have been run for the site first. These facts were checked against disk and are easy to get wrong.

| Item | Value |
|---|---|
| Geodatabase | `data/calibration/Indiana-ElkinsvilleNE_revised.gdb/Indiana-ElkinsvilleNE_revised.gdb`. The name **is** doubled. The outer directory is not a valid FileGDB |
| NAIP | `data/naip/ElkinsvilleNE/<year>/`, 4-band Byte, 0.6 m, EPSG:26916 |
| Canopy height | `data/naip_chm/ElkinsvilleNE/<year>/`, **float32 metres**, nodata -9999, so `chm_scale = 1.0`, not `0.01` |
| Stand metrics | `results/stands/ElkinsvilleNE/stand_metrics.parquet`, 40 stands × 6 years |
| Landsat annual | `results/stands/ElkinsvilleNE/satellite_annual.parquet`, 40 stands × 41 years, 1985 to 2025 |
| Landsat per scene | `data/satellite/ElkinsvilleNE/landsat_observations.parquet`, one row per stand per scene, **before** quality control |
| Crowns | `results/stands/ElkinsvilleNE/crowns/crowns_<year>.gpkg`. **Not** where `ProjectPaths.crowns_dir` points |
| NAIP years | 2012, 2014, 2016, 2018, 2020, 2022 |
| Native resolution | 1.0 m for 2012 and 2014, 0.6 m for 2016 onward, all resampled to a common 0.6 m grid |

Roots come from `csdv_core.io.paths.project_paths()`, so `CSDV_DATA_ROOT` and `CSDV_RESULTS_ROOT` are honoured when set.

Two consequences of the resolution change. Prefer years with the same native resolution for the image columns, so the only thing changing between panels is the forest. `metric_panel` already draws 2012 and 2014 as hollow markers for the same reason.

### Working with satellite tables

`stand_metrics.parquet` carries no `ndvi_*` columns. The satellite join has not been re-run since the segmentation re-tune, so `load_satellite_annual(site, stand_id)` reads `satellite_annual.parquet` directly. That is the source of record in any case, since it holds 41 years against six NAIP dates.

`load_satellite_observations(site, stand_id)` applies `filter_observations` with the parameters from `satellite.yaml`, which is what `annual_table` does before it computes anything. Skipping that step draws a cloud of observations the annual value was never fitted to. On `ELKNE-U16-0-0a` it keeps 551 of 977 rows, and 362 of the 426 it drops are scenes where every pixel of the stand was masked.

The two time bases do not line up. The satellite metrics are annual and reach back to 1985. The NAIP metrics have six dates from 2012. `satellite_panel` ticks the NAIP dates along the bottom of a satellite series for exactly this reason.

### Working with crowns

`load_stand_crowns(site, year, geometry)` handles this, but know what it does. Each GeoPackage is about 125 MB and holds roughly 40,000 scene-wide crowns, so the read is filtered by bounding box first and only then by centroid, matching `zonal/crowns.py::crowns_in_stand`. A full read takes long enough to be worth avoiding. Expect a crown figure to take a couple of minutes for six dates.

Crown polygons carry interior rings, so `crown_overlay` builds a compound path per polygon. Walking only `.exterior` fills the holes in.

Segment counts are low. Median segment diameter is 31 m at about 10 per hectare, so a stand under about 10 acres carries fewer than 30 segments and `crown_cv` is unstable there. `MIN_CROWNS = 3` gates with `<`, so a value can rest on three segments. Check `crown_count` before building a crown example on any stand.

`crown_diam_m` is the area-equivalent circle diameter, `2 * sqrt(area / pi)`. It is not a major axis or a fitted ellipse.

`segment_id` is unique within a year and meaningless across years. Do not use it to track a crown through time.

## Adding a new metric

1. Pick a stand. Read `stand_metrics.parquet` and find one where the metric moves, ideally in both directions. A stand that only rises shows half the story.
2. Pick years. Three columns is the norm. Keep them resolution matched. Check that the metric actually differs between the years you chose.
3. Copy the closer template. `gap_fraction.py` for a per-pixel metric, `crown_cv.py` for anything built from crown polygons. Change `METRIC`, the `PRESETS` dict, the row labels and the y limits.
4. Decide what the panels should be. This is the part that varies, and it is worth thinking about rather than copying. `gap_fraction` uses three image rows because the metric is a pixel count, so the third row is the threshold applied pixel by pixel. `crown_cv` uses one image row plus a distribution panel, because the metric summarises a set of measurements rather than a set of pixels. A texture metric would want the quantized grey levels. If the metric is a plain reduction with nothing to show per pixel, drop to two image rows.
5. Check the numbers. Recompute the metric straight from the source with the matching function in `csdv_core.zonal` and confirm it equals the value in `stand_metrics.parquet`. If the panel does not reproduce the number, the panel is wrong.
6. Look at the figure. Open the PNG and read it as a reader would. Regenerate until it is clear.
7. Write the caption and wire up the links. Add `**Example.**` with a link in the metric entry in `docs/metrics.md`, and a link back to the entry at the end of the example section.

## Figure conventions

On the figure: year labels on the top row only, row labels down the left edge, one colour key per row that needs one, axis labels on the series panel.

Not on the figure: titles, metric values as text, annotations, arrows, callouts, stand identifiers. Everything explanatory goes in the caption. A reader who wants the number reads the caption or the table.

A metric the pipeline withholds still gets drawn. `crown_cv` returns NaN below `MIN_CROWNS`, and plotting only the reported dates leaves isolated points with no line between them, which reads as missing data rather than as a value the sample was too small to trust. `crown_cv.py` computes the value at every date, draws the unsupported ones hollow on a dashed line, and legends them. Note that hollow means withheld on that panel, where `metric_panel` uses hollow for the coarser 1.0 m dates. A panel that reuses a marker for a second meaning has to carry its own legend.

Colours come from `csdv_core.viz.style`. Do not hardcode hex. The two-class canopy height key reuses the stage palette on purpose: the open class takes the colour of `ESI`, the most open stage, and the canopy class the colour of `LSE`, the closed-canopy stage.

One quantity, one ramp, across every panel that shows it. Crown diameter uses `CROWN_CMAP` for both the map overlay and the strip-plot dots, so the two rows read as one system and need only one key. It is deliberately not `viridis`, which the canopy height row already owns.

Pixels outside the polygon are washed out in any panel that stands in for a per-pixel metric. The metric counts in-stand pixels only, and a reader should be able to see which pixels those are. A panel showing an *input* rather than the metric is not washed out. The near infrared row in `glcm_texture.py` keeps its full range across the whole chip, because dimming everything outside would make the stand look darker in the near infrared than the forest around it, which is an artefact of the dimming.

A figure may carry synthetic inputs, but only to anchor a scale, and only under two conditions. Every synthetic value has to be produced by the metric's own function at draw time rather than written into the caption, and the synthetic columns have to be labelled so no reader mistakes them for delivery data. `row_directionality.py --preset scale` is the case that earns it. The metric is uncalibrated and has no stage envelope, so a reader has nothing to judge 0.86 against, and the pair of columns that carries the figure is one random field drawn twice, once raw and once blurred. Real stands cannot make that comparison because no two of them differ in only one property. Everything else in the set stays on real stands.

Grey and washed out mean different things, and a figure that uses both has to keep them apart. In `height_mean.py` the third row sets every pixel below 2 m to NaN, so `band_panel` paints it `MUTED` grey, while the pixels outside the polygon stay washed out as usual. Grey is in the stand and not counted. Washed out is not in the stand. The metric excludes both, for different reasons, and a reader who cannot tell them apart cannot see why the sample changed size.

A spread drawn per date is a whisker, not a band. `height_mean.py` first drew the within-stand 10th to 90th percentile as a `fill_between`, which sat at almost the same value as the grey module band behind it and read as one shape. Whiskers separate cleanly and, more to the point, they refuse to interpolate a spread across the two years between NAIP dates. Nothing measured that.

Where no stage envelope exists, `reference_band` puts the rest of the module behind the series instead: the 10th to 90th percentile across all 40 stands at each date, with the median through it. `edge_density` and `ndvi_trend` have no envelope at all, and without a backdrop their panels would be a bare grid with no way to judge whether a value is unusual. `glcm_texture.py` draws both, because the reference band is what shows the metric moving with the acquisition rather than with the forest.

A satellite figure's context row is not a measurement. `ndvi_mean.py` draws three NAIP panels above a Landsat series and labels the row "NAIP, context only", because nothing on the figure is computed from them and the stand was cut before the NAIP record starts.

A panel that draws a resolution-sensitive quantity ignores `--max-px`. `edge_density.py` reads its boundary row at the native 0.6 m, because the length of a boundary traced across a raster scales with the pixel size, and a decimated read would paint a boundary genuinely different from the counted one. It logs the drawn value against the table to prove it.

Keep figures near 400 KB. `finish()` re-encodes to a 256-colour palette, which cuts a full-resolution figure by about three quarters. Use median cut, not maximum coverage. Maximum coverage merges neighbouring pale stage bands into one palette entry and erases the boundary between them.

## Writing style for captions

From [`docs/conventions.md`](../../../../docs/conventions.md), plus the house additions:

- Active voice. Short sentences. Short paragraphs.
- Never use em dashes. Never use semicolons to join two clauses.
- Banned: "straightforward", "leverage" as a verb, "utilize", "facilitate", "comprehensive", "robust" outside a statistical sense, "cutting-edge", "novel", "delve", "harness", "embark", "it's worth noting", "it's important to note".
- Also avoid: "landscape", "realm", "tapestry", "pivotal", contrast setups such as "not just X, but Y", faux-insider transitions such as "Here's why that matters", and introductory fluff.
- No bullet points inside a caption. Write short paragraphs.
- Say what the reader is looking at, tie it to real numbers, then say what the metric does not tell them. That last part is usually the most useful sentence.

## Reference: labelled Elkinsville stands

Ten impact polygons carry interpreter labels and were used for the worked examples in the classification document. Disturbance years are the interpreter's `LastImageryPreDist` and `FirstImageryPostDist`.

| `stand_id` | Disturbance | Acres | Pre | Post | Bbox fill |
|---|---|---|---|---|---|
| `ELKNE-U2-0-0` | Wind | 175.9 | 2016 | 2017 | 0.28 |
| `ELKNE-U44-0-0` | Uneven-age / selection harvest | 191.3 | 2012 | 2013 | 0.26 |
| `ELKNE-U9-0-0` | Clearcut with reserves | 11.4 | 2016 | 2017 | 0.47 |
| `ELKNE-U33-0-0` | Tree mortality | 5.0 | 2012 | 2014 | 0.39 |
| `ELKNE-U13-0-0` | Clearcut | 3.3 | 2016 | 2017 | 0.72 |
| `ELKNE-U23-0-0` | Shelterwood establishment cut | 3.2 | 2012 | 2013 | 0.77 |
| `ELKNE-U48-0-0` | Clearcut | 6.0 | 2007 | 2008 | 0.35 |
| `ELKNE-U16-0-0a` | Clearcut | 2.8 | 2007 | 2008 | 0.33 |
| `ELKNE-U16-0-0b` | Clearcut | 1.8 | 2007 | 2008 | 0.28 |
| `ELKNE-U17-0-0` | Wind | 21.8 | 2016 | 2017 | 0.47 |

Stands with a low `bbox_fill` are thin or irregular. Their chips carry a lot of out-of-stand area, and `row_directionality` is not reported below 0.50. Only `U13` at 0.71 and `U23` at 0.77 clear that gate, so eight of the ten labelled stands carry no value, including both wind events. The gate turns out not to protect anything measurable. See [Known gaps](../../../../docs/metrics.md#known-gaps).

The last three were disturbed in 2007, before the NAIP record here starts, so they show recovery only. `mark_disturbance` draws nothing when the event falls outside the plotted range, which is deliberate.

## Files

| Path | Role |
|---|---|
| `_lib.py` | Site resolution, the panels `csdv_core.viz` lacks, crown and satellite loading, stage envelope bands and the module reference band, presets and the shared command line, PNG compression |
| `gap_fraction.py` | Two `gap_fraction` figures. The template for a per-pixel metric |
| `crown_cv.py` | Two `crown_cv` figures. The template for a metric built from crown polygons |
| `gap_persistence.py` | One figure. The template for a metric computed across a pair of dates |
| `height_mean.py` | Two figures. The template for a metric whose sample membership changes between dates |
| `glcm_texture.py` | One figure. Shows the intermediate arrays a texture metric passes through |
| `edge_density.py` | Two figures. The template for a metric whose drawn pixels must be the counted pixels |
| `row_directionality.py` | Two figures. The template for a metric that has to be drawn through its transform, and for anchoring an uncalibrated scale |
| `ndvi_mean.py` | One figure. The template for a satellite metric on the Landsat time base |
| `ndvi_seasonal_amplitude.py` | One figure. Refits the harmonic so the curve and the check are the same call |
| `ndvi_trend.py` | One figure. The template for a metric reported with a confidence interval |

`_lib.py` deliberately holds only what is missing from `csdv_core.viz`. Anything already there is imported. If a helper here proves useful to three or more metrics, move it into `csdv_core.viz`.
