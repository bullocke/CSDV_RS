# Pair model: canopy loss between two NAIP dates

`csdv pair` trains a Siamese network on the chips of a label run (`docs/guides/disturbance_labels.md`) and maps loss probability over a grid unit for every pair of consecutive NAIP dates. This is a first look at whether the automatic labels can train a classifier. It trains one configuration with one unit held out. The fuller check (grouped folds, three cell sizes, three loss targets, a learning curve) uses the same code with other options and is not run yet.

## Quick start

```bash
# CHPC login node, once: the encoder weights (compute nodes have no internet)
wget -P $CSDV_DATA_ROOT/models/torchvision https://download.pytorch.org/models/resnet34-b627a593.pth

# CHPC: the labels beside the four-band chips, then a ten minute check, then the run
source scripts/chpc/env.sh
LABEL="$CSDV_EXPORT_DIR/label/scale_test"
B=$(sbatch --parsable $CSDV_SBATCH_CPU --output="$CSDV_LOG_DIR/%x-%j.out" \
    scripts/slurm/run_label_build.sbatch "$LABEL")
S=$(sbatch --parsable $CSDV_SBATCH_GPU --dependency=afterok:$B --time=00:30:00 \
    --output="$CSDV_LOG_DIR/%x-%j.out" scripts/slurm/run_pair_first_look.sbatch first_look --smoke)
sbatch $CSDV_SBATCH_GPU --dependency=afterok:$S --output="$CSDV_LOG_DIR/%x-%j.out" \
    scripts/slurm/run_pair_first_look.sbatch first_look

# Laptop: tables, figures and the map (under 100 MB, no checkpoints)
scripts/chpc/pull_exports.sh pairmodel/first_look
```

`CSDV_SBATCH_GPU` is the option string of a GPU allocation in `scripts/chpc/env.sh` (see `env.example.sh`). The job script names no allocation. It holds out `q4412328` by default and reads the label run in `$CSDV_EXPORT_DIR/label/scale_test`. Both can be changed through the variables listed at the top of the script.

The package needs PyTorch and torchvision and nothing else beyond the label package. On CHPC both come with `scripts/chpc/create_env.sh`. On a laptop, for the tests and for looking at samples: `python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu`.

## Samples

A sample is one chip in one interval: the image at the earlier date, the image at the later date, and the change label of that interval. A chip with T dates gives T minus 1 samples. Only consecutive dates are paired.

The target is the 0.6 m change raster pooled to the model's cells with the rule of the label package (`pool_change` and the `pooling` thresholds of `label.yaml`). A cell is ignore when half of it is ignore, loss when at least half of its labelled pixels are loss, background when at most 10% are, and ignore in between. Cells where either date has no imagery over half of their area are ignore as well. The whole outline of an accepted event counts as loss, including the part of a partial loss that still passes the forest rule.

Training draws samples with replacement. Half of the draws are samples in which at least 1% of the chip is loss. Within each half a unit's total weight is the square root of its sample count, so the two burn units, which hold a quarter of all loss, are drawn more often than a small unit and less often than their count.

Each training sample is changed at random in five ways. A 384 px crop is cut from the chip. Date B is offset against date A by up to 4 px, because NAIP dates are misregistered by a median of 3.5 m. Both dates and the target are flipped and turned together. A date whose native resolution is 0.6 m or finer is blurred half of the time, so that sharpness does not tell the dates apart. Brightness, contrast and colour balance are shifted separately for each date, because NAIP radiometry is not calibrated between years.

One sample in five is fed in reverse order, the later date first. Nothing is loss in a reversed pair: labelled cells are background, unlabelled cells stay unlabelled, and cells that gained forest between the dates are unlabelled, since fed backwards they look like loss. Older dates are always the 1 m ones, and reversed pairs keep the model from reading the order of the dates from the look of the imagery.

Digital numbers are divided by 255 and standardised with one mean and one standard deviation per band, computed once from 200 training chips and stored in the checkpoint. Pixels without imagery are set to zero after that.

## Model

The same ResNet-34 encoder reads both dates. At each level a 1 by 1 convolution joins the two feature maps and their signed difference (later minus earlier). The difference is signed because loss has a direction: an absolute difference cannot tell loss from regrowth. A short top-down decoder brings the joined features to the stride of the output cells. With 0.6 m pixels a stride of 16 is one 9.6 m cell, so the network predicts one logit per cell and never upsamples to pixels. `--cell-m 4.8` or `2.4` adds one or two decoder steps.

The encoder starts from ImageNet weights read from a file. Its first convolution takes four bands: the filters for red, green and blue are kept and the near-infrared filter starts as their mean.

The loss is binary cross-entropy plus soft Dice over the labelled cells of a batch. The optimiser is AdamW with one warm-up epoch and a cosine schedule.

## A third date

A pair of dates cannot tell a canopy that was lost from one that was leafless for a season. Spongy moth strips an oak stand in June, the July image shows bare crowns, and the stand is green again two years later. `csdv pair train --dates 3` builds a model that also reads the date after the pair, where a chip has one. The target does not change: it is still the loss between the first two dates. The third image only helps to decide it.

The same encoder reads the third image. The convolution that joins the dates at each level then takes five parts in place of three: the two feature maps of the pair, their difference, the feature map of the third date and its difference from the first. Both parts of the third date are zero for a sample that has none.

The third image is optional for every sample, because the newest interval of any map has no later image. In training it is absent for the last interval of a chip (17% of the samples of the scale test), for a pair fed in reverse, where it covers less than 90% of the crop, and for a random share of the other samples (`--third-drop`, 0.3). One set of weights therefore maps an interval from two dates when it is new and from three once the next flight exists. The third image gets the same crop, with its own small offset, blur and colour shift.

`csdv pair predict` uses the third date wherever it has imagery and writes a second map from two dates alone, `prob_pair_<unit>.tif`. The difference between `prob_<unit>.tif` and that file is what the third date changed. The last band of both is the same, since the last interval has no later date. A run with three dates takes about 1.4 times the GPU time of a pair run. A checkpoint records its number of dates, and checkpoints written before this option load as pair models.

A simpler check needs no training. `csdv pair predict --skip-pairs` also scores every pair of dates that skips one, date k against date k + 2, with the same weights and from those two images alone, and writes `prob_skip_<unit>.tif`. Band k belongs to interval k, and the last band is empty. A loss of interval k that the skipping pair shows as well has lasted to the next image. A defoliated stand that recovered should show no loss there. The model was trained on consecutive dates only, so the skipping pair spans twice the usual time and its scores are a diagnostic, not a calibrated probability.

## Agent of a loss

A second head on the same decoder features names the agent of a loss for every cell: harvest, wind, fire, biotic mortality, conversion or other, the six codes of `<chip>_agent.tif`. It is on by default (`--no-agents` turns it off). Its target is the most common agent among the loss pixels of each loss cell. Cells without loss have no agent, and neither have loss cells on which the two vision models named different agents (about one event in seven), so the head learns only from cells where the label is sure of the agent. Its cross-entropy is added to the loss of the first head with a weight of 0.5 (`--agent-weight`).

The agent head answers "if this cell is a loss, which agent", so its output is meaningful only where the first head says loss. `csdv pair predict` writes it as `agent_<unit>.tif`, the code of the most likely agent of every cell (1 harvest, 2 wind, 3 fire, 4 biotic mortality, 5 conversion, 6 other), one band per interval and 255 where there is no imagery. It has no colour table, because a GeoTIFF keeps one only on a single band. Read it under a mask of `prob_<unit>.tif`, or open the type rasters described under "Mapping a unit", which apply that mask.

Validation reports the share of agent cells named correctly (`val_agent_accuracy`) and the mean recall over the agents present (`val_agent_balanced_accuracy`), and writes the confusion matrix of the best epoch to `agent_confusion_val.csv`. The agents are far from balanced, and they are tied to units: fire comes almost wholly from two burn units, wind from one, biotic mortality from two. A held-out unit therefore tests the agents it happens to hold, and the accuracy over all cells mostly reflects harvest and fire. Read the recall per agent.

## Splits and scores

By default units are split, never chips, because chips of one unit share imagery and events. `--holdout-units` names the test units, which are neither trained nor validated on. `--val-units` names the validation units, and `--fold` with `--n-folds` takes a grouped fold of the remaining units in their place. `csdv pair train --dry-run` prints the split, the loss cells of each agent on each side, and checks that every chip file is there.

Agents are tied to units: nearly all wind events of the scale test lie in one unit. A split by unit therefore cannot test an agent whose main unit is held out, and cannot train on it either. `--split blocks` keeps the held-out units as they are and validates on ground inside every other unit. Each unit's lattice is cut into blocks of `--block-chips` chips (8 by default, 2.46 km), a window belongs to the block of its lower-left chip, and whole blocks go to validation until it holds about `--val-share` of the loss cells of every agent and of the chips of every unit. Training chips that touch a validation chip are left out. The chips of each side are stored in `config.json` and the checkpoints (`split_chips`), and a run folder refuses to resume with another split. On the full scale test the default gives validation 27 to 40% of each agent's cells.

A blocks split shows whether the model separates the agents once it has seen all of them. It is not a test of new ground: validation shares flights, and often the same storm or fire, with training, so its scores are higher than those of a held-out unit and are not comparable with a split by unit.

## Training on more than one label run

Training reads one chip folder. To add the chips of a second label run, for example stable-forest windows drawn with `csdv label sample --n-di 0 --n-other 0 --n-quiet N`, merge the runs into a new label folder and train on that:

```bash
csdv pair merge-chips --out "$CSDV_EXPORT_DIR/label/merged_sf" \
    --run scale_test="$CSDV_EXPORT_DIR/label/scale_test" \
    --run stable_forest="$CSDV_EXPORT_DIR/label/stable_forest"
CSDV_PAIR_LABEL_DIR="$CSDV_EXPORT_DIR/label/merged_sf" CSDV_PAIR_HOLDOUT=q3708104 CSDV_PAIR_SPLIT=blocks \
CSDV_PAIR_TRAIN_ARGS="--run-draws stable_forest=0.15" \
    sbatch $CSDV_SBATCH_GPU --time=02:30:00 --output="$CSDV_LOG_DIR/%x-%j.out" \
    scripts/slurm/run_pair_first_look.sbatch holdout_sf_q3708104
```

The merged `chips.parquet` has a column `run` and window ids prefixed with the run name (`stable_forest:W0001`), so the windows of two runs never share an id in the blocks split. Rasters are hard-linked where the file system allows it (the job copies real files to node scratch), symbolic links otherwise. A chip id an earlier run already holds is left out of a later run, and the count is printed. The source folders are not changed.

Half the draws go to samples with loss and half to samples without (`loss_draws`). `--run-draws NAME=SHARE` gives the samples without loss of a named run their own share, taken from the second half: with `stable_forest=0.15`, 50% of draws hold loss, 15% are stable-forest samples and 35% are the other samples without loss. Without the option the weights are exactly those of earlier runs. The share is stored in `config.json` as `run_draws`.

A share of 0 never draws the run's samples without loss and draws every other sample with the weight it has without that run. It is the control for a run with a share. A model trained on the first run's folder alone is not a fair control, because the blocks split and the band statistics depend on every chip in the folder. With the merged folder and `stable_forest=0`, both runs have the same split, statistics and validation chips, and they differ only in the draws.

The holdout reruns of West Virginia and the Smokies are such a pair per unit. Their run table, `config/pair/stable_forest.csv`, goes to the batch scripts of the defoliation test, which take any table. A last column `train_args` adds options to `csdv pair train`:

```bash
csdv pair merge-chips --out "$CSDV_EXPORT_DIR/label/merged_sf" \
    --run scale_test="$CSDV_EXPORT_DIR/label/scale_test" \
    --run stable_forest="$CSDV_EXPORT_DIR/label/stable_forest"
scripts/chpc/submit_pair_defoliation.sh --table config/pair/stable_forest.csv --dry-run
scripts/chpc/submit_pair_defoliation.sh --table config/pair/stable_forest.csv
# Laptop
scripts/chpc/pull_pair_defoliation.sh --table config/pair/stable_forest.csv
```

### A run left out, tied units, and a test on many places

`--exclude-runs NAME` never draws the samples of a label run, with loss or without, and gives every other sample the weight it has without that run. The blocks split, the validation samples and the band statistics stay those of the whole folder, so it is the control of a model trained with the run. Validation still holds the run's blocks, which favours the control a little when the best epoch is chosen. Chips of an excluded run are not decoded into the cache. The dry run prints the share of the training draws per label run, with loss and without. A unit's weight grows with the root of its sample count, so a run of many small units takes a larger share of the draws than of the samples.

`csdv pair merge-chips` also writes `chips/sites.csv`, from the `sites.csv` of every label run that has one (a run by places, such as round 2). Training reads two of its columns. A unit whose `holdout_with` names a held-out unit saw the same event or flight and is held out with it, always. `--hold-role test` holds out the units with that role as well. The dry run names the units held out this way.

`csdv pair score` scores a checkpoint on the label chips of units it did not train on, without a map. It reads each chip whole, as validation does, and writes `samples.parquet` (one row per chip and interval with the counts of loss and background cells by predicted percent, from which any grouping can be pooled later), `units.csv` and `score.json` (pooled scores, and those of the intervals that end on an image from November to April). `--role test` takes the units from `chips/sites.csv`. It does not check that the model did not train on them.

```bash
csdv pair train --chips-dir $CHIPS --holdout-units q4709438 --hold-role test \
    --exclude-runs round2 --split blocks --dry-run
csdv pair score --checkpoint best.pt --chips-dir $CHIPS --role test --out-dir score
```

The tests of round 2 use all three (`config/pair/round2.csv`, submitted by `scripts/chpc/round2.sh evaluate`). A row of kind `score` trains a model and scores it on the test places. A row of kind `rescore` scores the model of another row on the chips of its own label run.

## Loading

The chip rasters are compressed with the bands of a pixel stored together, so a crop of two dates decodes every band of the whole chip. That takes about 50 ms per sample and was two thirds of an epoch. `--cache-dir` names a folder where every chip of the run is decoded once into an uncompressed array (about 12 MB per chip of seven dates, 30 GB for the full scale test) that the loader processes open as a memory map. A sample is then read in about 1 ms and is identical. The cache is built at the start of training, in parallel, and a manifest of the size and modification time of each chip's files decides which chips are decoded again. Put it on node scratch (the job script does).

After every epoch the validation units are scored on whole chips without any random change. The scores are counted on cells: precision, recall, F1 and intersection over union at a threshold, and the average precision over all thresholds. The epoch with the highest average precision, averaged over the validation units that hold loss, is kept as `best.pt`. Training stops when eight epochs pass without a better one.

| File in the run folder | Content |
|---|---|
| `best.pt`, `last.pt` | Checkpoints: weights, settings, band statistics, split, pooling thresholds, the hash of `chips.parquet`, the git commit. `last.pt` also holds the optimiser, so a stopped job goes on from it |
| `config.json` | The same record without weights |
| `metrics.csv` | One row per epoch: losses, validation scores overall and per unit, seconds of training, of waiting for data and of validation |
| `pr_val.csv` | Precision and recall at every threshold for the best epoch |
| `agent_confusion_val.csv` | Agents of the validation cells: label in rows, model in columns, recall per agent |
| `curves.png` | Loss and validation scores by epoch |

## Mapping a unit

`csdv pair predict` reads the NAIP mosaics of a unit in blocks of 2,048 px with a halo of 256 px, encodes each date of a block once, and decodes every pair of consecutive dates. It writes `prob_<unit>.tif`, a Cloud Optimized GeoTIFF of uint8 percent with one band per interval, named by the two dates, and 255 where either date has no imagery. `prob_forest_<unit>.tif` is the same map with cells that were not forest at the earlier date set to 0. The first file is the primary one, because the forest mask of the grid runs counts roofs as forest.

For a model with an agent head it also writes `type_<unit>_<dateA>_<dateB>.tif`, one file per interval. Each holds the agent code of the cells whose loss probability is at or above a threshold and 0 on every other cell. The file has a single band, a colour table and 0 as its nodata value, so it opens in QGIS with one colour per agent and nothing drawn where the model sees no loss. The threshold is the one with the best F1 on the validation units, stored in the checkpoint, or `--threshold` (0.5 if the checkpoint holds none). The log and the last lines of the command name the threshold used. A different threshold needs no new run of the model: `flagged_agents` in `csdv_core.pairmodel.predict` cuts the same rasters from `agent_<unit>.tif` and `prob_<unit>.tif`.

A model with three dates (section "A third date") also writes `prob_pair_<unit>.tif`, its map from two dates alone, and `--skip-pairs` adds `prob_skip_<unit>.tif` for any model. Both have the bands of `prob_<unit>.tif`. The agent and type rasters follow `prob_<unit>.tif`.

The cells of the map lie on the global 0.6 m lattice of the UTM zone at multiples of the cell size, which is where the label chips pool to cells. The grid of a unit starts anywhere on the 0.6 m lattice, so the map covers the unit's extent snapped outward, and the 19.2 m layers of the disturbance run are sampled at the centre of each map cell.

## Mapping several units

`scripts/chpc/submit_pair_sites.sh` submits one job per unit of `config/pair/sites.csv` (the rows with `batch` set to `yes`), and `scripts/chpc/pull_pair_sites.sh` copies every result to the laptop in one connection.

```bash
# CHPC
git pull && source scripts/chpc/env.sh
scripts/chpc/submit_pair_sites.sh --dry-run        # prints the sbatch commands
scripts/chpc/submit_pair_sites.sh                  # mode look, model first_look_blocks2

# Laptop, when squeue is empty
scripts/chpc/pull_pair_sites.sh
```

| Mode | What a job does | Run folder | GPU time per unit | What the map is |
|---|---|---|---|---|
| `look` (default) | Maps the unit with the model of `--model` and compares it with the unit's labels. No training | `look_<unit>` | About 15 minutes | A view of what the model does there. The model trained on label chips inside the unit (5 to 10% of its ground), so the scores of the comparison are not a test |
| `holdout` | Trains with the unit held out (split by blocks), then maps and compares it | `holdout_<unit>` | About 50 minutes | An independent test of the unit. A type whose labels lie mostly in that unit is hardly trained |

A unit that the model run held out and mapped itself is skipped in both modes, because that run is already its held-out map. A unit whose run has a comparison in the export folder is skipped too, so the command can be given again after a failed job. `--units q3912112,q4010549` takes other units than the list.

The jobs run `scripts/slurm/run_pair_first_look.sbatch`. In the `look` mode it gets `CSDV_PAIR_STAGES="predict compare"` and `CSDV_PAIR_CHECKPOINT`, the checkpoint of the model run. Every run folder holds `site.json`, which names the unit and the model run behind the map.

### The defoliation test

`config/pair/defoliation.csv` lists the runs of one experiment: whether the model stops reading defoliation as loss when recovered defoliation is labelled background (`routing.stressed_recovered` of the label package, built into the label run `scale_test_stressbg`), when it reads a third date, or both. A `train` row trains a model without one unit and maps that unit. A `look` row maps a unit with the model of another row. `scripts/chpc/submit_pair_defoliation.sh` submits one job per row, and `scripts/chpc/pull_pair_defoliation.sh` copies the results.

```bash
# CHPC, after the label build of the variant (docs/guides/disturbance_labels.md)
git pull && source scripts/chpc/env.sh
scripts/chpc/submit_pair_defoliation.sh --dry-run
scripts/chpc/submit_pair_defoliation.sh                 # or --after <job ID of the label build>

# Laptop, when squeue is empty
scripts/chpc/pull_pair_defoliation.sh
```

The first three-date run is first run as a smoke job of about ten minutes (`<run>_smoke`), and every other job waits for it to end well. A `look` job waits for the job that trains its model. A run whose comparison is already in the export folder is skipped, so the command can be given again. `--only RUN,RUN` takes named rows, `--optional` adds the rows marked `optional`, and `--after JOBID` holds everything until another job has ended well. Every job passes `--skip-pairs` to the map through `CSDV_PAIR_PREDICT_ARGS`, and the number of dates and the seed to training through `CSDV_PAIR_TRAIN_ARGS`. The comparison of a run scores its map against the labels it was trained on, so the two label variants are set side by side on the laptop against one set of labels.

## Comparison

`csdv pair compare` sets a map beside the disturbance run and the label chips of the unit.

| Output | Content |
|---|---|
| `chips_agreement.csv` | Cell scores of the model and of the DI probability against the pooled labels, per interval and overall, on the cells where both have a value |
| `di_objects.csv` | For each detected object: the model's mean probability and the share of its cells at or above the threshold |
| `model_only.csv` | Patches of at least 0.3 ha at or above the threshold that no detected object touches |
| `overview.png` | Model and DI probability for every interval, with the footprints of the label chips |
| `events/*.png` | Close-ups of the largest detected objects, of the largest objects the model does not see, and of the largest model-only patches: earlier image, later image, model, DI, label and, for a map with agents, the model's type |
| `agent_confusion.csv` | For a model with an agent head: the agent it names on the label's loss cells, against the agent of the label |
| `compare.json` | The headline numbers |

The threshold is the one with the best F1 on the validation units (`--checkpoint`), or `--threshold`.

In a close-up the label panel shows loss in red, ignored cells in grey and labelled background in a pale tone. Ground that no chip covers is hatched, and the chips are outlined, so an empty label panel can be told from a labelled one without loss. When `agent_<unit>.tif` lies beside the map, a sixth panel, "Model type", colours the cells at or above the threshold by the agent the model names, with a legend of the agents present. It uses the colours of the type rasters. Cells below the threshold are near white and cells without imagery are grey.

Chips carry the flight date of their own quarter quad and the map carries the unit's median date, so their intervals are matched by position in the series, never by name.

## Crops for close-ups

The 0.6 m NAIP mosaics stay on CHPC. `csdv pair crops` cuts chosen boxes out of them, one four-band GeoTIFF per box and year, so that close-up figures can be drawn on the laptop at full resolution. The boxes are a CSV with `unit`, `box`, `epsg`, `minx`, `miny`, `maxx`, `maxy` (in the unit's UTM zone) and `years` (space-separated), at most 3 km on a side.

```bash
# Laptop: send the boxes
rsync -av boxes.csv "$CSDV_CHPC_HOST:$CSDV_CHPC_EXPORTS/pair_crops/<name>/"
# CHPC
sbatch $CSDV_SBATCH_CPU --output="$CSDV_LOG_DIR/%x-%j.out" scripts/slurm/run_pair_crops.sbatch <name>
# Laptop
scripts/chpc/pull_exports.sh pair_crops/<name>
```

The crops land in `pair_crops/<name>/<unit>/<box>_<year>.tif`. A year without a mosaic is named at the end of the log and skipped.

## Reading the scores

The labels descend from the candidates of the disturbance run (detected objects, weak objects and CHM loss). Agreement with the labels therefore favours what those sources flag, and the DI scores well against labels that it proposed. The held-out unit removes the model's memory of the ground, not this dependence. The patches in `model_only.csv` are where the map says something the labels cannot, and they have to be looked at. The blind audit of the label run is the independent measure.
