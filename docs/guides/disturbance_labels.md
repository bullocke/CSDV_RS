# Disturbance labels: training windows from candidates and vision models

`csdv label` turns the detections of the disturbance runs into labelled windows of multi-date NAIP for a segmentation model. Two vision models judge every candidate in a window, their agreed answers become label rasters, and everything they could not settle is left unlabelled. The thresholds are in `src/csdv_core/config/label.yaml`. The design and its test are described in `analysis/vlm_labeling/2_scale_test/PLAN.md`.

## Quick start

The steps run on two machines. The laptop holds the API keys and the exported runs. CHPC holds the NAIP mosaics. A small run can skip CHPC, because `render` can also read NAIP from Planetary Computer and the USDA image service.

```bash
# Laptop, once: the cell layers the label rules need (about 0.4 GB)
scripts/chpc/pull_label_layers.sh

LABEL=results/chpc_exports/label/scale_test

# Laptop: candidates, then the sample (dry run first, then once for good)
REVIEWED=data/reference/grid_pilot/review_default.gpkg
csdv label candidates --label-dir $LABEL
csdv label sample --label-dir $LABEL --reviewed $REVIEWED --dry-run
csdv label sample --label-dir $LABEL --reviewed $REVIEWED
scripts/chpc/push_label_run.sh $LABEL scale_test

# CHPC: render the request images and the four-band chips
source scripts/chpc/env.sh
LABEL="$CSDV_EXPORT_DIR/label/scale_test"
N=$(($(wc -l < "$LABEL/units.csv") - 1))
sbatch $CSDV_SBATCH_CPU --array=1-1 --output="$CSDV_LOG_DIR/%x-%A_%a.out" \
    scripts/slurm/run_label_render.sbatch "$LABEL"

# Laptop: fetch the images, the dates and the cell NDVI (one login)
scripts/chpc/pull_label_run.sh scale_test

# Laptop: ask the models about the first stage, look at it, then the rest
csdv label verdict --label-dir $LABEL --stage 1 --cap 12 --dry-run
csdv label verdict --label-dir $LABEL --stage 1 --cap 12
csdv label build --label-dir $LABEL --stage 1 --no-imagery
csdv label verdict --label-dir $LABEL --cap 90 --dry-run
csdv label verdict --label-dir $LABEL --cap 90
csdv label build --label-dir $LABEL --no-imagery
```

The cap is for the whole label run, not for one command. Set it to what the run may cost so far: a cap below the forecast of `--dry-run` stops the run part of the way.

A run without CHPC replaces the push, the batch job and the pulls with `csdv label render --label-dir $LABEL`, and drops `--no-imagery`.

## The label run folder

Every step reads and writes one folder.

| Path | Content | Written by |
|---|---|---|
| `windows.csv`, `units.csv` | The windows and the units that have one. `windows.csv` can be edited by hand, followed by `csdv label units`. | `sample` |
| `candidates/<unit>.gpkg` | The candidates of a unit in its UTM CRS. | `candidates` |
| `stacks/<window>/` | Four-band NAIP of the window, one GeoTIFF per year, and `meta.json` with the dates. | `render` |
| `requests/<window>/` | Request JPEGs, `requests.jsonl` and, later, `polygons.jsonl`. | `render` |
| `answers.jsonl` | One record per model answer. | `verdict` |
| `ledger.jsonl`, `responses/` | What each call cost, and the response cache. | `verdict` |
| `chips/` | Training chips and `chips.parquet`. | `render --chips`, `build` |
| `labels/` | `labels.gpkg` and the tables `intervals.csv`, `events_kept.csv` and `suspects.csv`. | `build` |
| `sheets/` | Review sheets. | `sheets` |

## Candidates

A window is labelled completely only if everything that might be a disturbance in it has been judged. The candidate set is therefore inclusive, and it has three sources. `di` objects are the detections of the run, with the refined 0.6 m outline where the run has one. `weak` objects are connected cells whose disturbance probability is at least 0.2 but which were not detected. `chm` objects are connected cells that left the forest mask with a drop in canopy cover of at least 0.2. The last source exists because the detection standardises change against nearby forest, so a fire that affects a whole neighbourhood is largely removed. False candidates cost one call each and nothing else, because the models reject them.

`csdv label candidates` writes the candidates and prints their number and area per unit and source. Check that table before sampling. A unit where `chm` objects cover thousands of hectares in an interval has either a landscape-scale event or a canopy height model that sits low on one date, and the models will tell the two apart.

## The sample

`csdv label sample` draws three strata and records the inclusion probability of every seed. `di` windows are seeded by a detected object, with the same number per unit, shared among sub-strata (last interval or not, probability above or below 0.9, three size classes). `other` windows are seeded by a weak or CHM-loss object. `quiet` windows are forest chips with no candidate and low probability throughout, which measure omission and add plain background. A window is the block of 512 px chips that covers its seed with a margin, at most three chips on a side. A larger candidate is seen through a window placed on a random point inside it. `--reviewed data/reference/grid_pilot/review_default.gpkg` adds a window for every polygon already interpreted in `csdv review`, as a fixed reference inside the run.

Units differ widely in how much quiet forest they hold. Each unit gets the same number of quiet windows as far as it has quiet chips, and what a unit cannot take goes to the others, up to `--quiet-max-per-unit` (25). In the 17 pilot units five have no quiet chip and seven have 12 or fewer, so 125 of the 150 quiet windows lie in five units. Their inclusion probabilities say so.

`sample` also writes a `stage` for every window. Stage 1 holds the reviewed windows, the quiet windows (no candidate, so no call) and, per unit, `--stage1-di` windows seeded by a detected object and `--stage1-other` windows seeded by another object, chosen by a hash of their own. It is a random part of the sample, small enough to judge and look at before the rest is paid for. `verdict`, `build` and `sheets` take `--stage`. The example windows were mostly chosen on large events, and in the eight random ones both models reported loss for 39% of the candidates against 76% in the others, so the first stage is where the yield of a random sample is first seen.

A sample is drawn once per label folder. Window IDs follow the order of the sample, so another design gives other windows the same IDs. `sample` refuses a folder that has been rendered or judged, and `render` refuses a stack folder that holds other ground than its window.

The inclusion probability belongs to the seed. Other candidates in a window were not drawn with a known probability, and windows can overlap, so rates for a paper should be estimated from seeds only.

## Rendering

`csdv label render` fetches every year of a window on the 0.6 m lattice of its UTM zone and writes the images the models will see. Imagery finer than 0.6 m is averaged and coarser imagery is interpolated, on both machines with the same code. With `--naip-source local` the mosaics under `$CSDV_DATA_ROOT/naip/<unit>/<year>/` are read, and a missing mosaic is an error, because a compute node has no internet. `--check` lists the mosaics without rendering.

The mosaics written by `csdv download` flag their fourth band as alpha. A warp straight from such a file treats near-infrared as transparency, which removes pixels where it is zero and weights the others by it. `render` reads the pixels into an array first, so the flag has no effect. Stacks and chips written here carry band names and no alpha flag.

A window near the edge of its unit reaches past the unit's mosaics. In the sample of the scale test this holds for about one window in nine. Ground outside the unit's cell grid has no cell layer and is always `no_data` in the labels, so imagery is not required there. A year is kept when its imagery covers at least 98% of the part of the window, with its collar, that lies inside the cell grid. The ground beyond the unit is left empty (black) in the request images and in the chips, whatever the source, so a window looks the same on both machines. The remote sources could fill it from the neighbouring quads, but the mosaics on CHPC end at the unit. A test on the two example windows of this kind (22 candidates, judged again with the strip) changed the loss label of one candidate, which is what a plain repeat of identical requests also does. A window that still lacks a year is skipped, because the cell layers and the verdict states are indexed by the unit's years. `render` names the skipped windows in its last line, and `stacks/<window>/meta.json` records the reason under `failed` and the share of the stack inside the unit under `inside_unit`. A later `render` tries the failed years again.

The acquisition date of a window is the date of the quarter quad that fills most of it. The date is drawn on every request image, listed in the prompt and stored with the chips. A unit's mosaic is named after the median date of all its quads, which was up to 168 days from the date of a window's own quad in the example set, so the file name is not used. The mosaic's `.provenance.json` lists the quads in the order they were pasted, and a pixel keeps the first quad that reaches it. `render` follows that order over the quads' footprints (the nominal quarter quad and 150 m around it) and takes the quad that fills most of the window. On the 40 example windows this gives the true date in 281 of 282 window-years. `meta.json` records the rule per year (`date_rule`: `quad` from a sidecar, `item` from a remote read, `mosaic_name` when a mosaic has no usable sidecar), the share of the window flown on that date and any other date in it (`mixed_dates` when the share is under 0.75, about one window-year in sixty). `render --check` counts the mosaics without a sidecar, and the last lines of `render` count the window-years by rule. A line that mentions `mosaic_name` means wrong dates and should stop the run.

`render` also writes `stacks/<window>/cell_ndvi.tif`: the NDVI of every 19.2 m cell of the unit under the window, one band per year, from the sums of the near-infrared and red digital numbers of the cell. It is a few kilobytes, and it is what the never-green rule needs on a machine without the imagery (see the label rules).

Each candidate gets one image per date with a thin yellow outline. The view follows the rule adopted after the pilot: zoom in on a small candidate (down to 150 m), use 384 m at 640 px when the candidate fits, and widen the view only when it does not, at up to 1,280 px so that resolution stays at 0.6 m up to 768 m. After the verdicts, `csdv label render --kind polygon` writes two images (before and after the agreed loss) for every accepted candidate whose outline the models doubt.

A unit needs a line of location text (state and physiographic region) in `LOCATIONS` in `src/csdv_core/label/requests.py`. Rendering stops for a unit without one, because a prompt without a location has not been tested. Site labels such as "CA_CampFire" name the answer and are never shown to a model.

On CHPC, render the 40 example windows first and compare them with the laptop's renders before the full array:

```bash
csdv label compare-renders results/vlm_labeling/2_scale_test/examples/requests \
    results/chpc_exports/label/examples/requests
```

The two renders start from the same quads, but the CHPC mosaics were resampled once when they were written, so small differences are expected. A median difference of a few grey levels is normal. A large one means the two sides read different dates or different bands. The command compares pixels only. The dates are in `requests.jsonl` on each side, and they should be compared too: the first CHPC render of the example windows had the right pixels and the unit's median date on every window. Images are compared where both renders have imagery, and the command lists the windows where one render lacks imagery over more than 5% of the view, which are the windows on the edge of a unit.

## Verdicts and cost

`csdv label verdict` sends the rendered requests to Gemini 3.7 Flash and GPT-6.1 Terra (the prompt is `label/prompts/verdict_v3.txt`). The keys `GEMINI_API_KEY` and `OpenRouter_API_KEY` are read from the environment or from `~/.bashrc`. `--cap` is required and has no default. A call is refused when its estimated cost would take the run past the cap, and the ledger survives between sessions, so the cap holds for the whole label run. Every answer is cached on a hash of the prompt and the images, so repeating a finished run costs nothing. Answers are stored with their prompt version, and only answers to the current version are used, so a run can be re-judged after a prompt change while the old answers stay in `answers.jsonl` for comparison. A verdict that arrives but cannot be read as one (a wrong number of states, or a word that is not a state) is asked for once more. On the 82 probe sites one answer in 164 was of this kind. A window without rendered requests adds no call, and `verdict` says how many of the selected windows are in that state, so a forecast that looks small can be told from one that is complete. An answer of "too many requests" or "overloaded" (HTTP 429, 503, 529) is waited for, up to six times and two minutes each, without using up the three attempts of an ordinary failure.

The answers are not exactly repeatable. Thirty example candidates asked again with identical images and prompt kept their status in 27 cases and their loss label in 29. Terra changed the state of some date for six of them and Gemini for one.

Two verdicts on one candidate cost $0.016 in the example run, which is more than the pilot's $0.012 because many candidates needed the larger image. A polygon costs $0.004.

## What counts as loss

Canopy loss means that live trees were removed or died between two dates. Two things that change the image are not loss, and the prompt names both.

Defoliation is not loss. A canopy that is leafless, brown or thin in one growing-season image and green again later was defoliated. The models record such a date as `stressed`, a fifth state beside forest, partial loss, full loss and open. They use the same state when they cannot tell from the images whether standing trees are dead. An interval that ends on a stressed date is left unlabelled.

The removal or fall of trees that were already dead is not a second loss. Once the trees have died the ground is no longer forest, so the loss belongs to the interval in which they died, and later intervals are `open`.

The first version of the prompt had neither rule. In the example run it dated an oak stand's loss to the July image in which spongy moth had stripped it, called every later date open, and so labelled the interval in which the trees were removed as background.

## Routing and outlines

No person takes part in routing. A candidate is accepted when both models report canopy loss beginning on the same date, and rejected when neither reports loss on any date. When both report loss but name different first dates, the candidate is kept. The interval of the later date is labelled as loss, and the intervals before it, back to the earlier date, stay unlabelled (`routing.disputed_date: later`). By the later date both models agree that the canopy is gone, and without this rule a stand that died slowly gave almost no loss labels, because the models rarely agree on the date of a gradual mortality. On the 82 adjudicated sites the rule raised the share of real disturbances that receive a loss label from 71% to 85%, and four of the six disputed real sites were dated to the right interval. With `disputed_date: ignore` the whole range from the earlier to the later date stays unlabelled. Apart from this case, an interval is labelled only where the two models agree about it.

The models date the first visible damage to the crowns. The detection dates the loss of height. The two differ often: in the example run about a third of the accepted candidates were dated one to three intervals before the interval that flagged them. The label follows the models, because the classifier reads NAIP only, with two safeguards. The interval that flagged a candidate is left unlabelled unless both models report loss in it (`routing.guard_flagged_interval`), so that a height loss the detection saw is never background on the models' say-so. Loss reported again in the interval right after an agreed loss is also left unlabelled (`routing.consecutive_loss`): in the three example cases that were checked, the second interval showed ground already cleared. A later loss after a quiet interval is a new event and is labelled.

The candidate outline is the labelled core whenever both models call it about right. Otherwise Terra draws a polygon. It replaces or extends the outline only if it covers at least half of the candidate and is at most ten times its area. For types with sharp edges (clearcut, clearing, mining, and fire with full loss) the polygon beyond the candidate is labelled as loss. For diffuse types it is left unlabelled, so that an unflagged part of a damaged stand cannot become background. A polygon that covers its candidate but is more than ten times its area says that the candidate is a fragment of a larger event. In the example run 27 of 51 polygons were of this kind. Such a polygon is kept as unlabelled ground, never as loss (`outlines.polygon_oversize`). It leaves only forest and scattered trees unlabelled. Ground that was not forest before the event (fields, houses, roads, water) stays background, and the polygon adds no ring of its own (`rules.oversize_keeps_nonforest`). Without this, a polygon drawn around a whole burn left chips that held loss and no background at all.

## Label rules and reasons

Each pixel of each interval gets a reason, and the change label is a lookup of the reason through the `policy` table in `label.yaml`. A loader can apply another table to the stored reason raster without new labels or new calls.

| Reason | Meaning | Default label |
|---|---|---|
| `accepted_full`, `accepted_partial`, `accepted_severity_disputed` | Inside an accepted candidate, both models report loss in this interval | 1 |
| `extension_full`, `extension_partial` | Model polygon beyond a crisp candidate | 1 |
| `boundary_inside`, `boundary_outside` | Band along the outline of the labelled loss: 4.8 m on both sides of a precise outline, 9.6 m outside an envelope | ignore |
| `extension_ignored`, `outline_trimmed` | Model polygon beyond a diffuse candidate, and candidate the polygon leaves out | ignore |
| `disagree`, `date_disputed`, `verdict_failed`, `unjudged` | The models split (for a disputed date: the intervals before the later date), or there is no usable pair of verdicts | ignore |
| `flagged_unconfirmed`, `loss_repeated`, `canopy_stressed` | The models place the loss in another interval than the one that flagged the candidate, report loss again right after an agreed loss, or see a leafless or thin canopy that is not a loss | ignore |
| `rejected_flagged`, `rejected_weak` | Jointly rejected candidate, in the interval that flagged it | ignore |
| `margin` | Ring of 19.2 m around a candidate that is loss or unlabelled. None around an accepted candidate with a precise outline | ignore |
| `under_min`, `raised_prob`, `forest_lost_unjudged` | Candidate under 0.3 ha, probability of 0.2 or more, or a cell that left the forest mask, none of them judged | ignore |
| `sparse_woody`, `unstable_forest`, `no_data` | Trees below the forest rule (canopy cover from 0.05, about one crown in a cell, or mean height from 2 m), forest in an interval where the unit's forest mask moves by more than 15%, missing imagery or canopy height | ignore |
| `judged_intact`, `post_event_open` | Inside a candidate, both models report no loss in this interval | 0 |
| `nonforest` | Not forest at the earlier date, and no trees to speak of | 0 |
| `stable_forest` | Forest on both dates, outside every candidate | 0 |

Where several rules claim a pixel, loss wins over ignore and ignore wins over background. The band is measured from the edge of the whole event, so a core that runs into an unlabelled extension has no band on that side.

An envelope has no band on its inside and a precise outline has no ring. Both were in the first plan and both were dropped after the example run. A 9.6 m band inside an envelope left a median of 48% of its area as labelled loss, and the uncertainty of an envelope lies outside it, where the ring already is. The ring around a precise outline hid the forest edge next to a clearcut, which is the contrast the model most needs. The widths are in `label.yaml` (`band_inside_envelope_m`, `margin_precise_m`), and `csdv label build --config FILE` builds the labels under other values without touching the packaged ones.

The forest rule of the grid runs is canopy height alone, so a roof or a rock face taller than 5 m is forest, and a lower one counts as scattered trees. `rules.never_green_ndvi` (off by default) treats a cell as non-forest when its NDVI never reaches the value on any date: `stable_forest`, `unstable_forest`, `forest_lost_unjudged`, `raised_prob` and `sparse_woody` become `nonforest` there, and the forest bands of `_aux.tif` follow. Everything a candidate paints stays as it is, so a loss that both models accept is still loss. The rule compares the largest NDVI over all dates and never one date, because a third of the forest cells of the example set fall below 0.10 on some date (leaf-off flights, uncalibrated radiometry). On the example set a value of 0.10 flags 0.4% of the forest cells, most of them roofs in downtown Flagstaff, and the only candidates that lie mostly on such cells are three roofs that both models rejected. A value of 0.15 reaches into living forest: dry conifer stands in Arizona and Colorado peak between 0.10 and 0.20. Most of the ground the rule moves is scattered trees, which change from unlabelled to background. A low value is the safer one, for a reason that showed only on the full sample of the scale test. Some flights carry almost no vegetation signal (the median NDVI of forest cells was about zero in one summer flight and in one November flight), so forest that was lost right after such a flight never has a high NDVI as forest. At 0.05 the ground vegetation that follows a harvest or a burn still lifts the cell above the value on a later date, and what stays flagged is ground with no vegetation at all: roofs, roads, rock slides, water. At 0.10 sparse regrowth stays below it, and dead or burned forest is flagged with the roofs. The scale test uses 0.05, which flags 0.15% of its forest cells. Choose the value from the run's own windows (`analysis/vlm_labeling/2_scale_test/checks/never_green.py report`) and pass it with `--config`. Keeping that file as `config.yaml` in the label run folder lets `scripts/chpc/push_label_run.sh` send it and the build job on CHPC (`scripts/slurm/run_label_build.sbatch`) apply it. The candidates, the sample and the test for an unreliable forest mask keep the height-only rule, because cell NDVI exists only inside windows.

There is one loss class. The models judge severity for the whole outline, so cleared ground inside an envelope is often called partial, and the example run found bare ground labelled partial in a mine bench, a clearcut and a log landing. The reason raster keeps what the models said (`accepted_full` against `accepted_partial`), and the policy table can map stand-replacing loss to 2 again once severity is reliable at cell scale. The `events` layer keeps the severity of each event.

Labels are stored at 0.6 m and pooled to the model's output cells by `csdv_core.label.pooling.pool_change`. A cell is ignore when half of it is ignore. Otherwise it is loss when at least half of its labelled pixels are loss, background when at most 10% are, and ignore in between.

## Chips

A chip is 512 by 512 px (307.2 m) on the global 0.6 m lattice, so it is the same ground whichever unit it is cut from. Its ID gives the EPSG code and the lattice position (`e26919_x904_y14972`).

| File | Bands |
|---|---|
| `<chip>_naip.tif` | Dates by four bands, named `YYYY-MM-DD_R` and so on |
| `<chip>_change.tif` | One per interval: 0 background, 1 loss, 255 ignore (2 is reserved for stand-replacing loss) |
| `<chip>_agent.tif` | One per interval: 1 harvest, 2 wind, 3 fire, 4 biotic mortality, 5 conversion, 6 other, 255 where there is no agreed loss or the models named different agents |
| `<chip>_reason.tif` | One per interval: the reason codes of `csdv_core.label.rules.Reason` |
| `<chip>_aux.tif` | Disturbance probability per interval (percent) and the forest mask per date. Not model inputs |

The label rasters need only the candidates, the answers and the cell layers, so `csdv label build --no-imagery` writes them on the laptop while the four-band chips stay on CHPC. The vector labels in `labels/labels.gpkg` (layers `events`, `reviewed_areas`, `candidates` and `acquisitions`, EPSG:5070) are the source of truth. Chips can be rebuilt from them.

Windows can overlap, and a chip that lies in two windows is written once, with the labels of the first window in the order of `windows.csv`. `build` writes its tables and `labels.gpkg` anew for the windows it is given, so the last build of a run should select all of them.

## Sheets and diagnostics

`csdv label sheets` draws two kinds of sheet for windows whose stacks are on this machine. An overview sheet shows every date in natural colour and colour-infrared, and the label of every interval over the later date. A pair sheet shows one training sample at native resolution: the earlier date, the later date, the later date with the 0.6 m label, the reasons as flat colours, and the target pooled to 9.6 m, with the 2.4 m and 4.8 m targets beside it.

`build` writes three tables. `intervals.csv` has one row per window and interval with the share of each reason group, the class shares after pooling to each cell size, and where cells with a strong greenness loss ended up. `events_kept.csv` says how many output cells each accepted candidate keeps as loss after the band. `suspects.csv` lists clusters of forest-background cells with a strong greenness loss, which are the places to look for a disturbance that no candidate source flagged. The greenness score (`csdv_core.label.screen`) standardises the later date's NDVI given the earlier date's, against forest background of the same window. It is a diagnostic and a guard, not a label, and it has no value where the whole window changed.

## Notes

The label of an interval describes live trees that were removed or died in it. Ground that was cleared earlier, or dead trees that are removed later, counts as background when both models report no new loss, even though the two images differ. The classifier may still respond to a defoliated canopy or to the removal of dead trees, because both change the image strongly. The labels leave the first unlabelled and call the second background where the models are sure.

Jointly rejected candidates are unlabelled only in the interval that flagged them. About one joint rejection in twelve was a real disturbance in the pilot.

The forest mask comes from the canopy height model, which can sit low over a whole unit on one date. Single-date dips are filled, and intervals in which the unit's forest area still moves by more than 15% lose their unverified forest background.

Model IDs and the prompt version are stored with every event (`label_source`), because the models will change.
