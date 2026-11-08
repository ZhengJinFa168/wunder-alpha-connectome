# Wunder Alpha Connectome

Our solution for Wunder Fund's Alpha Connectome competition (deadline Nov 15, 2026; public leaderboard rank
**251 of 5,760, top 4.4%**): predict `t0` and `t1`, two
indicators of future price moves of instrument `i0`, one row at a time from 112 order-book, trade and extra features.
The organisers' original README is kept at the end ([Competition reference](#competition-reference)).

## Final submission

**`sweep11_mix.zip`: leaderboard 0.66021** (rank 251 of 5,760, top 4.4%, public leaderboard on Nov 8, 2026;
organisers' baseline 0.59353),
31 minutes on the grader (limit 60). One 2-layer, 128-unit GRU per instrument group, picked per sequence:

| Group (tick size) | Model | Trained on |
|---|---|---|
| 1-3 | sweep 10a group specialist, fine-tuned on all of that group's validation sequences, weights 0.9 fine-tune + 0.1 specialist | train, then validation |
| 4 | `ft_film~wise`: weights halfway between the student s9 and its group-conditioned fine-tune `ft_film` | train |
| fallback | the student s9 (group detection unsure: 1 of 1,873 validation sequences) | train |

The group is detected from the 99 warm-up rows (99.9% right on validation); the buffered warm-up rows are then
replayed through the chosen model only (`pipeline/solution_routed.py`).

### How it is built

1. **Scoring-mask model** (sweep 5, `pipeline/mask_model.py`): predicts which rows the metric scores from causal
   inputs (AUC ~0.89), fitted on validation half A (`split.json`); gives every train row a weight p(scored).
2. **Teacher** (sweep 9): a greedy ensemble of 9 of 32 GRUs trained on train with the true targets; its predictions on
   train are the distillation targets (`pipeline/teacher.py`, `cache/t_9.npy`).
3. **Student s9** (sweep 9): 2x128 GRU distilled from the teacher, rows weighted 0.5 + 0.5 p(scored), 10 passes,
   checkpoint picked on validation half B (`pipeline/trainer.py`, `pipeline/multi_train.py`). Half B 0.67824.
4. **Group specialists** (sweep 10a): s9 fine-tuned on each tick-size group (`pipeline/groups.py`); group 4 from a
   FiLM group-conditioned fine-tune, weight-averaged with s9 (`pipeline/routed.py`). Leaderboard 0.65693.
5. **Validation fine-tune** (sweep 11): the group 1-3 specialists fine-tuned on validation (lr 2e-4, 3 passes,
   unscored rows weighted 0.06, average of passes 2-3), first on half A / half B to score honestly by cross-fitting
   (`pipeline/crossfit.py`), then on all of validation for the submission. +0.0033 on the leaderboard.

### What did not help

- Fine-tuning group 4 on validation (leaderboard -0.00065), and training from scratch on train + validation
  (memorises the validation sequences: 0.87 on seen, 0.58 on unseen ones).
- Importance weighting train sequences by how validation-like they are; recipe tweaks, weight soups, output-layer-only
  and mixed fine-tunes (sweep 12, 0.66007); per-group 2-model ensembles (sweep 13, 0.66009, 44 minutes).
- Earlier: wider or other architectures (minGRU, TCN), extra features, auxiliary targets, larger teachers.

### Leaderboard history

| Submission | Leaderboard | Grader time |
|---|---|---|
| organisers' baseline | 0.59353 | 27m12s |
| GRU 128, weighted MSE | 0.64972 | 28m26s |
| sweep 3: distilled from an ensemble | 0.65327 | 26m52s |
| sweep 5: + rows weighted by p(scored) | 0.65444 | 26m52s |
| sweep 6: distilled from a mask-weighted teacher | 0.65660 | 28m11s |
| sweep 10a: group specialists | 0.65693 | 31m12s |
| **sweep 11: + validation fine-tune (g1-3)** | **0.66021** | 31m06s |
| sweep 12: recipe tweaks | 0.66007 | 31m01s |
| sweep 13: per-group ensembles | 0.66009 | 44m17s |

## Repository

- `pipeline/`: the code. Data loading (`data.py`), models and ONNX export (`model.py`), multi-model training
  (`trainer.py`, `multi_train.py`), fast scoring with the official metric (`fast_eval.py`), teachers and ensembles
  (`teacher.py`, `ensemble.py`), mask model, group detection and routing (`groups.py`, `routed.py`,
  `solution_routed.py`), validation training checks (`crossfit.py`, `enscheck.py`, `shift.py`, `soup.py`), export
  checks and speed tests (`test_export.py`, `speed.py`).
- `colab/`: one-cell Colab notebooks, generated from the local code by `colab/build_*.py` so Mac and Colab run the
  same version; `*_rclone` versions read and write a Google Drive through rclone. `* copia.ipynb` are saved runs.
- `WORKSPACE.md`: the lab notebook (every sweep, result and decision); `PROGRESS.md`: progress report;
  `analysis/`, `research/`: diagnostics and literature notes.
- Not in git: `datasets/` (the 34 GB starter pack), `runs/` checkpoints (in Google Drive), `zips/`.

Quick checks:

```bash
python -m pipeline.fast_eval --baseline          # reproduces the baseline's 0.617052
python -m pipeline.test_export                   # ONNX + solution.py == PyTorch for every model type
python -m pipeline.speed --per-group SUBMISSION_FOLDER
```

## Competition reference

Predict `t0` and `t1`, two indicators of future price movements of instrument
`i0`, from the L11 book-feature representation, trade and additional features.

### Data

Each input has 112 float32 features: two 52-feature book/trade groups
(`i0`, `i1`) and eight additional columns (`a0` through `a7`).
For each instrument, L11 denotes 11 price-like and 11 volume-like features
on each side of the book, bid and ask, plus the separate trade features.
The complete column order is in [Data overview](docs/data_overview.md).

| Dataset | Sequences | Rows |
|---|---:|---:|
| Train | 10,607 | 212,140,000 |
| Validation | 1,873 | 37,460,000 |

Every sequence has 20,000 rows. Steps 0–98 are warm-up; predictions are
required at steps 99–19,999. Process rows in order and reset state between
sequences.

### Metric: Global WP

The metric is **Global Weighted Pearson correlation (WP)**, using weights
`abs(clipped target)`
on rows selected by `need_prediction AND is_scored`. Compute it separately
for each target over all selected rows in the evaluated dataset, then average
the two target scores. On the selected rows, both targets and
predictions are clipped to `[-2.0, 2.0]`, and weights use the clipped targets.
Higher is better.

The validation mask is provided for local scoring. The test mask is hidden.
See [METRIC.md](METRIC.md) for the full scoring reference.

### Files

- `datasets/train.parquet`: labeled training sequences.
- `datasets/valid.parquet`: labeled validation sequences with `is_scored`.
- `datasets/valid_mask.parquet`: the validation mask as a separate table.
- `utils.py`: callback types and input-column definitions.
- `METRIC.md`: scoring specification and reference functions.
- `baseline/`: ready-to-run stateful ONNX baseline (inference only).
- `docs/`: competition website documentation.

### Baseline

The included vanilla GRU baseline scores **0.617052 WP** on the complete
validation set. It runs statefully on CPU with one ONNX Runtime compute thread.

```bash
python baseline/solution.py --validation datasets/valid.parquet
```

The baseline contains only inference code and an ONNX model. A ready-to-upload
archive is provided as `baseline/baseline_submission.zip`; no training code or
training checkpoint is included.

### Submission

Provide `solution.py` with `PredictionModel()` and `predict(data_point)`.
The callback receives the sequence identifier, step, `need_prediction` and
the current 112-feature vector. Return `None` during warm-up and two finite
predictions in `[t0, t1]` order on every required row.

Package the solution and its required model files in one ZIP, with
`solution.py` at the root. See [Quick Start](docs/quick_start.md) and
[Submission guide](docs/submission_guide.md).

### Execution limits

- **CPU:** 1 vCPU; CPU inference only, no GPU.
- **RAM:** 16 GB.
- **Time:** 60 minutes for the entire test set.
- **Submission size:** at most 20 MB for the uploaded ZIP.

The environment is Linux with Python 3.11, local SSD storage and no internet
access.

