# Reviewing detected disturbances

`csdv review` turns the exported grid runs into reference data. It draws a random sample of detected disturbance objects, serves a browser app in which each object is judged against NAIP of every year with all neighbouring detections in view, and scores the detections from the saved labels. The app also has a page for browsing any unit without a sample. The code is in `src/csdv_core/review/`. The exports it reads are described in [grid_processing.md](grid_processing.md) and the detection method in [disturbance_detection.md](disturbance_detection.md).

## Quick start

```bash
scripts/chpc/pull_exports.sh                           # copy the exports from CHPC
csdv review sample --run-tag default --n-per-site 6    # once: 6 objects per unit
csdv review app                                        # open http://localhost:8765
csdv review summary                                    # scores so far
```

The app runs in WSL and opens in the Windows browser at the printed address. It needs internet access for the NAIP imagery. Stop it with Ctrl+C. Labels are saved on every save, so the app can be stopped and restarted at any time, and it reopens at the first sample polygon without a label.

## The sample

`csdv review sample` draws the same number of objects from every exported unit of a run tag (equal allocation), at random among all of the unit's objects. Units with fewer objects contribute all of them. The sample IDs (`R001`, `R002`, ...) run in one random order across the units, so a session never works through one site in a block. Each row records the share of its unit's objects in the sample (`incl_prob`), which the scores use as a weight.

To interpret more, grow the sample to a larger number per unit:

```bash
csdv review extend --run-tag default --n-per-site 10
```

Existing IDs and labels never change. The new rows are shuffled among themselves and numbered after the existing ones. Every object has a fixed random key (from the seed and the unit), and the sample takes the objects with the smallest keys, so growing the sample in steps gives the same objects as one larger draw. Use the same `--seed` and `--exclude-growth` as the first draw. Units added to the export later, for example after re-running failed downloads, join the sample with the full number at the next `extend`. `--exclude-growth` leaves out growth objects, which are smaller than the minimum mapping unit and were kept only because they touch an earlier detection (about 29% of the pilot objects).

The assessed geometry of an object is its refined 0.6 m outline where the run made one, and its 19.2 m cells otherwise. The form header says which (`refined` or `object`).

Everything is stored in `data/reference/grid_pilot/review_<run_tag>.gpkg`, with the layers `sample`, `interpretations` (a table) and `redrawn` (EPSG:5070). Each save is also appended to `review_log.csv` next to it, with any redrawn outline as WKT. The `data/` folder is not committed.

## The review page

The map shows NAIP of the selected year with the outline under review in cyan and every detection within 1.5 km, coloured by onset year as in the exported onset raster. Overlapping outlines of different years mean that the cells were detected in more than one interval. Clicking an outline shows its object ID, interval and area under the polygon details.

NAIP comes from the source the run used. Years up to 2023 are a Planetary Computer mosaic of the exact quarter quads of that program year at 0.6 m (or 1 m for older years). The newest year comes from the USDA FPAC image service, which keeps only the latest collection of an area. When a source cannot be reached, the year falls back to the 3 m quicklook of the export. The line next to the controls lists the source of each year.

The overlays are drawn from the exported 3 m quicklooks and run rasters:

| Overlay | Shows |
|---|---|
| CHM | Canopy height of the selected year, 0 to 30 m |
| CHM change | Height of the selected year minus the previous year. Red is loss, blue gain, and changes under 2 m are transparent. |
| onset | The run's onset raster, with undetected forest transparent |
| probability | Highest loss probability over the intervals, transparent below 10% |

"Swipe prev | year" splits the map between the previous and the selected year. Under the map, a strip shows every year of NAIP and CHM at 3 m around the polygon, with the detected interval framed in blue (before) and red (after). It also plots the mean canopy height inside the outline and in a 30 m ring around it. It is usually the quickest way to see whether, and when, the canopy changed.

## What to record

| Field | Values | Meaning |
|---|---|---|
| Verdict | confirmed, likely, uncertain, rejected | Confirmed: clear canopy removal with support in the imagery (bare soil, slash, skid trails, dead crowns). Likely: a persistent new opening with weaker support. Uncertain: the imagery cannot settle it. Rejected: no disturbance (a false detection). |
| Change type | clearcut, partial_harvest, thinning, gap_mortality, wind, fire, insect_disease, flood, clearing_infrastructure, mining, other, or an artefact class | The agent, or for rejected objects the cause of the false detection: artefact_chm_noise, artefact_seam, artefact_shade, phenology_illumination |
| Interval | an interval of the unit, or unsure | When the disturbance happened. It defaults to the detected interval and is not saved for rejected objects. |
| Outline | about_right, too_large, too_small, wrong_shape, unsure | How well the assessed outline matches the disturbance |
| Context | part_of_larger_event, overlaps_other_interval, seam_or_season_problem, unit_edge | Any that apply |
| Confidence | 1 to 3 | Low to high |

The verdicts and change types match the ElkinsvilleNE adjudication (`data/reference/ElkinsvilleNE/README_supplemental_disturbance.md`), so the two reference sets can be pooled.

When the outline is wrong, draw the right one with the drawing tools on the left of the map. "Edit detected outline" copies the assessed outline into the drawing layer, where its vertices can be moved, cut or erased with the toolbar. "Clear drawing" removes it. A drawing that is present when the label is saved becomes the redrawn outline of the sample, and it is shown again when the polygon is reopened. Several parts can be drawn. They are merged into one multipolygon.

## Keyboard shortcuts

Shortcuts work whenever the cursor is not in a text field.

| Key | Action |
|---|---|
| 1, 2, 3, 4 | Verdict: confirmed, likely, uncertain, rejected |
| Enter | Save and open the next polygon without a label |
| s | Save and stay |
| b, a | Show the year before or after the detected interval |
| Left, Right | Previous or next year |
| w | Swipe on or off |
| h | Hide or show the outlines |
| c | Show or hide the 19.2 m cells |
| n, p | Next or previous sample ID |

Any polygon can be opened by typing its ID (or just its number) into "Go to ID", or by its address, for example `http://localhost:8765/?id=R017`.

## The explore page

"Explore" shows a whole unit with all of its detections over NAIP of any year, with the same overlays and swipe. Clicking an outline shows its attributes. When the object is in the sample, a button opens it on the review page. Otherwise "Add to review" adds it as an opportunistic pick (`X001`, ...). Opportunistic picks are labelled like sample polygons and kept as reference polygons, but they are not part of the random sample and are left out of the scores.

## Scores

`csdv review summary` writes `review_<run_tag>_summary.csv` and `.md` next to the GeoPackage and prints the table. For every unit, and pooled over units, it gives:

| Column | Meaning |
|---|---|
| precision | True detections (confirmed or likely) over labelled objects that are not uncertain |
| uncertain_share | Share of labels that are uncertain |
| area_precision | Precision weighted by the assessed area, an estimate of the share of detected area that is real disturbance |
| timing_match | Among true detections with a known interval, the share detected in the right interval. n_late and n_early count those detected one or more intervals late or early. |
| outline_ok | Among true detections with an outline judgement, the share judged about right |
| n_redrawn, iou, area_ratio | Number of redrawn outlines, their median intersection over union with the assessed outline, and the median of detected over redrawn area |

The `pooled` row weights each object by the inverse of its inclusion probability, so it estimates the scores over all detected objects of the run tag. The `site_mean` row is the unweighted mean over units. The sample covers detections only, so these are commission and agreement scores. Omission needs a separate sample of undetected forest, which is not part of the tool yet.

## Notes

The app is meant for one reviewer on one machine. It holds the store in memory and reloads it before each save if another process changed the file, so `csdv review extend` can run while the app is open. The new polygons appear after the app is restarted.

The first time a unit is opened, its Planetary Computer quads are looked up (a few seconds per year), and the IDs are cached in `.cache/review/pc_items.json`. A background thread does this for every unit when the app starts.

The year strip and the overlays use the 3 m quicklooks, so they show only the unit itself. Outside it the strip panels are grey. The 0.6 m CHM stays on CHPC.
