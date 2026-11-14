# Blind audit of a label run

`csdv audit` lets a person check the labels of a `csdv label` run against the imagery alone. It draws a seeded sample, builds a static page with every image in place, and stores one line per answer. The page shows no model answer, no detection interval and no candidate ID. For the label run itself see [disturbance_labels.md](disturbance_labels.md).

## Workflow

```bash
LABEL=results/chpc_exports/label/scale_test
AUDIT=results/vlm_labeling/2_scale_test/audit

# 1. Draw the sample once. Windows already seen with the model answers are left out.
csdv audit draw --label-dir $LABEL --audit-dir $AUDIT \
    --exclude-windows results/vlm_labeling/2_scale_test/checkin/sheet_windows.txt

# 2. Build the page. Run it again after a pull of stacks.
csdv audit build --label-dir $LABEL --audit-dir $AUDIT --name scale_test

# 3. Judge. Stop with Ctrl+C at any time. The page resumes where you stopped.
csdv audit app --audit-dir $AUDIT --interpreter eb

# 4. When you are done: rates per stratum, and agreement between interpreters.
csdv audit score --audit-dir $AUDIT
```

`draw` refuses to overwrite `items.csv`. A sample is drawn once, because answers refer to its item names. `--extend` with larger sizes, the same seed and the same excluded windows adds items after the existing ones.

## The sample

Block 1 holds judged candidates of three audit strata, shuffled so that they cannot be told apart.

| Stratum | Default size | Drawn from |
|---|---|---|
| Accepted events | 60 | Status `accepted`. Half from diffuse types (partial harvest, thinning, wind, insects and disease), each half shared among the agent groups with at least three per group |
| Rejected | 40 | `rejected` and `rejected_weak`, in proportion |
| Disagreements | 20 | `disagree` and `date_disputed`, in proportion |

Block 2 holds 60 complete windows (every candidate judged in every interval), shared among the `di`, `other` and `quiet` strata of the label sample in proportion.

Each stratum also gets a reserve of half its size, placed after the target items. The population is the seeded windows of the label run. Windows of the `reviewed` stratum and windows listed with `--exclude-windows` are left out. A candidate seen through several windows enters once, through one of them chosen at random.

`items.csv` is the key of the audit. It records for every item its stratum and sub-stratum, the size of that sub-stratum (`n_population`), its rank in the draw, the inclusion probability of its window, and the 60 items every interpreter judges (`shared`). Keep it away from the interpreters.

## Judging

Block 1 shows every date of a candidate's view in one grid, numbered from 1, with the candidate outlined in yellow.

| Step | Keys | Asked when |
|---|---|---|
| Loss | `0` or space: no loss of live canopy inside the outline. `2` to `8`: the first image in which the loss is visible. `U`: cannot tell | Always |
| Cause | `H` harvest, `W` wind, `F` fire, `M` mortality from insects or disease, `C` conversion, `O` other. Shift and a number adds a later loss | A loss was named |
| Label | `G` label holds, `L` too large, `S` too small, `X` wrong place | A loss was named and the item carries a loss label |

The label is drawn only in the third step, after the first two answers are fixed. Arrow keys enlarge an image and flick between dates, Esc returns to the grid, Backspace goes back one step, `Z` reopens the previous item, and `/` adds a note.

Block 2 shows a window with its label: red is labelled loss, hatched ground is left out of training, darkened ground has no label, and clear ground is background. Space means no loss on clear ground. A click marks a loss the label missed in the image shown, and Enter saves the marks. `T` hides the label and the wheel zooms.

The page times every item. The header shows the count, the median time per item and the time left for the target.

The target items of block 1 come first, then those of block 2, and the reserve of both blocks last. `http://localhost:8766/?block=2` keeps the page to one block, for example to judge windows before the candidates are finished.

## Files

| Path | Content |
|---|---|
| `items.csv` | The sample and its key |
| `site/` | The page (`index.html`, `app.js`, `style.css`), `items.js` and `img/<item>/` |
| `answers/<interpreter>.jsonl` | One line per answer, appended. A line with `"void": true` withdraws the item's earlier answer |
| `stack_windows.txt`, `stack_windows_reserve.txt` | Windows of block 2 whose imagery is still on CHPC |

Candidate items reuse the request images of the label run, so block 1 needs nothing beyond the run folder. A chip that lies in two windows carries the dates of the window that owns it, so where the other window was flown on another day of the same year its label is matched by the pair of years. Block 2 needs the stacks of its windows:

```bash
scripts/chpc/pull_label_run.sh scale_test --no-requests --stacks-from $AUDIT/stack_windows.txt
csdv audit build --label-dir $LABEL --audit-dir $AUDIT --name scale_test
```

`csdv audit build --refresh-labels` draws the label overlays again, for example after the labels were rebuilt.

## Other interpreters

```bash
csdv audit bundle --audit-dir $AUDIT --out-dir audit_shared
```

The folder holds the page and the items every interpreter judges (the first 30 accepted events and the first 30 windows, as far as they are built). It needs no install: an interpreter opens `index.html` in a browser, gives a name once, judges with the keyboard, presses "Download answers" and sends back the file `answers_<name>.jsonl`. Answers stay in that browser between sessions. `index.html?who=NAME` sets the name without the dialog. Put the returned files in `answers/` and run `csdv audit score`.

## Scores

`csdv audit score` writes three tables to `score/`. They hold the key, so open them only when your own audit is done.

| File | Content |
|---|---|
| `judged.csv` | Every answer beside the key of its item |
| `rates.csv` | Per interpreter and stratum: the share of items with loss, and for accepted events the share with loss in the labelled interval, with the label's agent, and with a label extent that holds. Each rate has its count, an exact (Clopper-Pearson) 95% interval and a rate weighted by inclusion probability |
| `agreement.csv` | Items judged by more than one interpreter, with every answer and whether all agree |

"Cannot tell" answers are left out of a rate and counted beside it. The weight of an item is the inverse of the probability that its window entered the label sample, times the share of its sub-stratum that the interpreter judged.

## Answer format

An answer line holds `item`, `kind`, `answer` (`none`, `loss` or `unsure`), `seconds`, `note`, `interpreter` and `saved`. A candidate answer adds `loss_at` (image number), `more_at`, `type` and `outline`. A window answer adds `marks`, each with `x` and `y` as fractions of the image from its top-left corner and the `image` number.
