# Map viewer for the pair-model maps

`csdv view` opens the exported pair-model maps in a browser, over NAIP, with two runs side by side. It is meant for judging what a model gets right and wrong at each site, and for leaving notes where something was seen. For the maps themselves see [pair_model.md](pair_model.md).

## Start

```bash
csdv view            # http://localhost:8767, stop with Ctrl+C
csdv view --no-open  # do not open a browser tab
```

The viewer reads `results/chpc_exports` (or `--export-dir`, or `$CSDV_EXPORT_DIR`). It lists every folder of `pairmodel/` that has a `map` subfolder, so a run pulled from CHPC appears after a reload of the page. No preparation step is needed. The page works without the internet on the local 3 m quicklooks. The 0.6 m imagery needs it.

## What it shows

A site is one unit (a 7.5 minute quad). Each run maps one site. The model picker of a panel lists the runs of the site and the Disturbance Index baseline of the unit (`runs/default/<unit>`).

| Run name | Listed as |
|---|---|
| `look_<unit>` | Non-holdout model |
| `holdout_<unit>` | Holdout |
| `holdout_sf0_<unit>` | Holdout, stable-forest control (0%) |
| `holdout_sf_<unit>` | Holdout with stable forest (15%) |
| `holdout_r1_<unit>` | Holdout, round-2 control (round 1 alone) |
| `holdout_r2_<unit>` | Holdout with round 2 |
| `holdout_r2sf_<unit>` | Holdout with round 2 and stable forest (15%) |
| other names | the run name |

A run stores loss probability and agent with one band per interval, and no date raster. The viewer derives three layers from them at a threshold.

| Layer | Interval "All" | One interval |
|---|---|---|
| Date | later year of the first interval that reaches the threshold | cells that reach it in that interval |
| Type | agent of that first interval | agent where the interval reaches it |
| Probability | highest probability over the intervals | probability of the interval |

The threshold of a panel starts at the threshold of its run (`compare/compare.json`). The slider changes it, and "reset" returns to it. With one interval and the run's threshold, the Type layer equals the exported `type_<unit>_<interval>.tif`. "Forest only" uses `prob_forest_<unit>.tif`, in which cells that were not forest before the interval count as no loss.

Layers are thresholded on the full-resolution cells and drawn with nearest neighbour, so a patch keeps its size at every zoom. Date colours run from yellow (2010) to purple (2025) on one scale for all sites and for the DI onset map, so two panels can be compared by colour. Type colours are those of the shared figures, not the colour table inside the type rasters.

The DI baseline has a Date layer (onset year) and a Probability layer (highest over intervals only). It has no Type layer.

Each panel header shows the scores of its run on the label chips of the unit (average precision, precision, recall and F1 at the run's threshold). They do not change with the slider.

## Background

The year buttons show the acquisition month where an interval names the date, because season explains many false alarms. Choosing an interval moves the image to its later date. `B` and `A` then switch between the image before and after.

NAIP comes from two sources. The local 3 m quicklook draws at once. From zoom 15 the 0.6 m imagery replaces it, through the tile routes of the review app (Planetary Computer, or the USDA service for a year the run took from there). The USDA service holds only its newest collection, which can be a later year than the button says. Untick "0.6 m" to stay on the quicklooks.

"Canopy height" shows the 3 m CHM quicklook of the year. "Height change" shows the change from the previous CHM year to the chosen one, red for loss and blue for gain, and nothing below 2 m.

"Labels" draws the outlines of the label run (`label/scale_test/labels/labels.gpkg`): events in the colour of their agent (white without one) and reviewed areas as dashed white lines. With one interval chosen, only the labels of that interval are drawn.

## Reading a cell and leaving a note

A click reads every run of the site at that cell. The table lists loss probability and agent per interval, and shades the cells that reach the threshold of their run (the slider's value for a run that is on screen).

The same click opens a note at that place. Pick a tag or type, and the note is kept. There is no save button: a change is written half a second after the last key and when the note closes. A note with no tag and no text is not kept. Tags are false positive, missed, wrong type, wrong date, good and other.

Notes are stored in `data/reference/map_notes/pair_maps.geojson` (or `--notes`). The file holds points in EPSG:4326 and opens in QGIS. Each note records its position, site, tag, comment, the two runs on screen, layer, interval, image year, background, thresholds, zoom, and the values of every run at the cell (`values`, as JSON text). Every change is also appended to `pair_maps_log.jsonl` beside it, which is never rewritten, so a deleted note can be recovered. The viewer reads the file again when another program has changed it.

The list on the right filters by site and tag. A click on a note returns to the view it was written in.

## Keys

| Key | Action |
|---|---|
| `,` `.` | previous and next site |
| Left, Right | image year |
| `[` `]` | interval |
| `B` `A` | image before and after the chosen interval |
| `D` `T` `P` | date, type, probability |
| Space (hold) | hide the map layer |
| `O` | map layer on or off |
| `L` | label outlines |
| `S` | one or two panels |
| `N` | note at the cursor |
| `1` to `6` | tag of the open note |
| Esc | close the note |

The address bar holds the whole view (site, runs, layer, interval, year, thresholds, position), so a reload or a bookmark returns to it. Adding `?pick=<lat>,<lon>` before the `#` opens a cell on load.

## Limits

- One person on one machine. Two browser tabs can write notes at once, and the later save of the same note wins.
- At zoom 13 and below a screen pixel is wider than a 9.6 m cell and shows one of the cells under it, so sparse single cells can drop out of the overview.
- The server holds the rasters of recently viewed runs in memory, about 0.4 GB after a session over several sites.
- The `prob_skip_*` and `prob_pair_*` rasters are not shown.

## Code

`src/csdv_core/view/`: `catalog.py` (runs and sites), `derive.py` (date, type and probability layers), `tiles.py` (tiles of the layers and of the quicklooks), `notes.py` (the note file), `server.py` (Starlette app), `page/` (the page, with Leaflet 1.9.4 in `page/vendor/`). Tests: `tests/unit/test_view_*.py`.
