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

## Agent of a loss

A second head on the same decoder features names the agent of a loss for every cell: harvest, wind, fire, biotic mortality, conversion or other, the six codes of `<chip>_agent.tif`. It is on by default (`--no-agents` turns it off). Its target is the most common agent among the loss pixels of each loss cell. Cells without loss have no agent, and neither have loss cells on which the two vision models named different agents (about one event in seven), so the head learns only from cells where the label is sure of the agent. Its cross-entropy is added to the loss of the first head with a weight of 0.5 (`--agent-weight`).

The agent head answers "if this cell is a loss, which agent", so its output is meaningful only where the first head says loss. `csdv pair predict` writes it as `agent_<unit>.tif`, the code of the most likely agent of every cell with a colour table, one band per interval. Read it under a mask of `prob_<unit>.tif`.

Validation reports the share of agent cells named correctly (`val_agent_accuracy`) and the mean recall over the agents present (`val_agent_balanced_accuracy`), and writes the confusion matrix of the best epoch to `agent_confusion_val.csv`. The agents are far from balanced, and they are tied to units: fire comes almost wholly from two burn units, wind from one, biotic mortality from two. A held-out unit therefore tests the agents it happens to hold, and the accuracy over all cells mostly reflects harvest and fire. Read the recall per agent.

## Splits and scores

Units are split, never chips, because chips of one unit share imagery and events. `--holdout-units` names the test units, which are neither trained nor validated on. `--val-units` names the validation units, and `--fold` with `--n-folds` takes a grouped fold of the remaining units in their place. `csdv pair train --dry-run` prints the split and checks that every chip file is there.

After every epoch the validation units are scored on whole chips without any random change. The scores are counted on cells: precision, recall, F1 and intersection over union at a threshold, and the average precision over all thresholds. The epoch with the highest average precision, averaged over the validation units that hold loss, is kept as `best.pt`. Training stops when eight epochs pass without a better one.

| File in the run folder | Content |
|---|---|
| `best.pt`, `last.pt` | Checkpoints: weights, settings, band statistics, split, pooling thresholds, the hash of `chips.parquet`, the git commit. `last.pt` also holds the optimiser, so a stopped job goes on from it |
| `config.json` | The same record without weights |
| `metrics.csv` | One row per epoch: losses, validation scores overall and per unit, seconds, and seconds spent waiting for data |
| `pr_val.csv` | Precision and recall at every threshold for the best epoch |
| `agent_confusion_val.csv` | Agents of the validation cells: label in rows, model in columns, recall per agent |
| `curves.png` | Loss and validation scores by epoch |

## Mapping a unit

`csdv pair predict` reads the NAIP mosaics of a unit in blocks of 2,048 px with a halo of 256 px, encodes each date of a block once, and decodes every pair of consecutive dates. It writes `prob_<unit>.tif`, a Cloud Optimized GeoTIFF of uint8 percent with one band per interval, named by the two dates, and 255 where either date has no imagery. `prob_forest_<unit>.tif` is the same map with cells that were not forest at the earlier date set to 0. The first file is the primary one, because the forest mask of the grid runs counts roofs as forest.

The cells of the map lie on the global 0.6 m lattice of the UTM zone at multiples of the cell size, which is where the label chips pool to cells. The grid of a unit starts anywhere on the 0.6 m lattice, so the map covers the unit's extent snapped outward, and the 19.2 m layers of the disturbance run are sampled at the centre of each map cell.

## Comparison

`csdv pair compare` sets a map beside the disturbance run and the label chips of the unit.

| Output | Content |
|---|---|
| `chips_agreement.csv` | Cell scores of the model and of the DI probability against the pooled labels, per interval and overall, on the cells where both have a value |
| `di_objects.csv` | For each detected object: the model's mean probability and the share of its cells at or above the threshold |
| `model_only.csv` | Patches of at least 0.3 ha at or above the threshold that no detected object touches |
| `overview.png` | Model and DI probability for every interval, with the footprints of the label chips |
| `events/*.png` | Close-ups of the largest detected objects, of the largest objects the model does not see, and of the largest model-only patches: earlier image, later image, model, DI, label |
| `agent_confusion.csv` | For a model with an agent head: the agent it names on the label's loss cells, against the agent of the label |
| `compare.json` | The headline numbers |

The threshold is the one with the best F1 on the validation units (`--checkpoint`), or `--threshold`.

Chips carry the flight date of their own quarter quad and the map carries the unit's median date, so their intervals are matched by position in the series, never by name.

## Reading the scores

The labels descend from the candidates of the disturbance run (detected objects, weak objects and CHM loss). Agreement with the labels therefore favours what those sources flag, and the DI scores well against labels that it proposed. The held-out unit removes the model's memory of the ground, not this dependence. The patches in `model_only.csv` are where the map says something the labels cannot, and they have to be looked at. The blind audit of the label run is the independent measure.
