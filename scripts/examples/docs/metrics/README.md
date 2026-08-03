# Metric example figures

## What this is

Each script here builds one figure for one metric in [`docs/metrics.md`](../../../../docs/metrics.md). The figures show a metric changing on a real stand, the imagery and canopy height behind the change, and where the value falls against the stage envelopes. `_lib.py` holds what the scripts share so a new metric is mostly layout and a caption.

Figures are written to `docs/images/metrics/` and are committed, so `docs/metrics.md` renders for anyone who clones the repository.

## How to run

```bash
.micromamba/envs/CSDV/bin/python scripts/examples/docs/metrics/gap_fraction.py
```

Scripts anchor every path to their own location, so the working directory does not matter.

Every script takes the same options:

| Option | Default | Purpose |
|---|---|---|
| `--stand-id` | per script | Which stand to render |
| `--years` | per script | Comma separated years for the image columns |
| `--pad-fraction` | `0.12` | Chip padding as a fraction of the stand's longer side |
| `--min-pad-m` | `25.0` | Smallest chip padding in metres |
| `--max-px` | `500` | Cap on the longer side of each chip read |
| `--dpi` | `150` | Output resolution |
| `--out` | `docs/images/metrics/<metric>.png` | Where to write |
| `--no-optimize` | off | Skip the palette re-encode |

To retune a zoom, change `--pad-fraction`. To try another stand, change `--stand-id` and `--years` together, because the interesting years differ per stand.

## Data this depends on

The stand pipeline must have been run for the site first. These facts were checked against disk and are easy to get wrong.

| Item | Value |
|---|---|
| Geodatabase | `data/calibration/Indiana-ElkinsvilleNE_revised.gdb/Indiana-ElkinsvilleNE_revised.gdb`. The name **is** doubled. The outer directory is not a valid FileGDB |
| NAIP | `data/naip/ElkinsvilleNE/<year>/`, 4-band Byte, 0.6 m, EPSG:26916 |
| Canopy height | `data/naip_chm/ElkinsvilleNE/<year>/`, **float32 metres**, nodata -9999, so `chm_scale = 1.0`, not `0.01` |
| Stand metrics | `results/stands/ElkinsvilleNE/stand_metrics.parquet`, 40 stands × 6 years |
| NAIP years | 2012, 2014, 2016, 2018, 2020, 2022 |
| Native resolution | 1.0 m for 2012 and 2014, 0.6 m for 2016 onward, all resampled to a common 0.6 m grid |

Roots come from `csdv_core.io.paths.project_paths()`, so `CSDV_DATA_ROOT` and `CSDV_RESULTS_ROOT` are honoured when set.

Two consequences of the resolution change. Prefer years with the same native resolution for the image columns, so the only thing changing between panels is the forest. `metric_panel` already draws 2012 and 2014 as hollow markers for the same reason.

## Adding a new metric

1. Pick a stand. Read `stand_metrics.parquet` and find one where the metric moves, ideally in both directions. A stand that only rises shows half the story.
2. Pick years. Three columns is the norm. Keep them resolution matched. Check that the metric actually differs between the years you chose.
3. Copy `gap_fraction.py`. Change `METRIC`, `DEFAULT_STAND`, `DEFAULT_YEARS`, `ROW_LABELS` and the y limits.
4. Decide what the third image row should be. This is the part that varies. `gap_fraction` uses a two-class split at the threshold, because the metric is a pixel count against a threshold. A crown metric would want crown polygons drawn instead. A texture metric would want the quantized grey levels. If the metric is a plain reduction of the canopy height model with nothing to show pixel by pixel, drop to two image rows.
5. Check the numbers. Recompute the metric straight from the source with the matching function in `csdv_core.zonal` and confirm it equals the value in `stand_metrics.parquet`. If the panel does not reproduce the number, the panel is wrong.
6. Look at the figure. Open the PNG and read it as a reader would. Regenerate until it is clear.
7. Write the caption and wire up the links. Add `**Example.**` with a link in the metric entry in `docs/metrics.md`, and a link back to the entry at the end of the example section.

## Figure conventions

On the figure: year labels on the top row only, row labels down the left edge, one colour key per row that needs one, axis labels on the series panel.

Not on the figure: titles, metric values as text, annotations, arrows, callouts, stand identifiers. Everything explanatory goes in the caption. A reader who wants the number reads the caption or the table.

Colours come from `csdv_core.viz.style`. Do not hardcode hex. The two-class canopy height key reuses the stage palette on purpose: the open class takes the colour of `ESI`, the most open stage, and the canopy class the colour of `LSE`, the closed-canopy stage.

Pixels outside the polygon are washed out in any panel that stands in for a per-pixel metric. The metric counts in-stand pixels only, and a reader should be able to see which pixels those are.

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

Stands with a low `bbox_fill` are thin or irregular. Their chips carry a lot of out-of-stand area, and `row_directionality` is not reported below 0.50.

The last three were disturbed in 2007, before the NAIP record here starts, so they show recovery only. `mark_disturbance` draws nothing when the event falls outside the plotted range, which is deliberate.

## Files

| Path | Role |
|---|---|
| `_lib.py` | Site resolution, the panels `csdv_core.viz` lacks, stage envelope bands, shared command line, PNG compression |
| `gap_fraction.py` | The `gap_fraction` example. Copy this to start a new metric |

`_lib.py` deliberately holds only what is missing from `csdv_core.viz`. Anything already there is imported. If a helper here proves useful to three or more metrics, move it into `csdv_core.viz`.
