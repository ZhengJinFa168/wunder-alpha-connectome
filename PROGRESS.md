# Alpha Connectome: progress report

Oct 26, 2026 · @Luigi

Our best leaderboard score is 0.65660, from a 128-unit GRU distilled from a mask-weighted 6-model teacher, up from the organizers' baseline of 0.59353; the top 20 sit at 0.669-0.674. The remaining sweep 6 students are training on a free Colab T4; sweep 7 waits for A100 compute units.

## The challenge

Wunder Fund's Alpha Connectome competition ([docs](https://wundernn.io/connectome/docs/quick_start)) asks for a model that predicts two hidden price-movement indicators, t0 and t1, for instrument i0. Final submissions are due Nov 15, 2026.

- **Data:** independent sequences of 20,000 time steps. Each step has 112 anonymized features: order-book prices and volumes for two instruments (i0, i1), recent trades, and 8 extra signals (a0-a7). Train has 10,607 sequences, validation 1,873, the hidden test 1,970 (39.4M rows).
- **How it runs:** the model is fed one row at a time, must keep its own memory within a sequence, and reset it between sequences. The first 99 steps are warm-up; predictions are required from step 99 on.
- **Metric:** weighted Pearson correlation, targets and predictions clipped to \[-2, 2\], each row weighted by the size of its target, averaged over t0 and t1. Only rows in a hidden scoring mask count (about 9% of rows).
- **Limits:** 1 CPU core, no GPU, 16 GB RAM, 60 minutes for the whole test set (about 91 µs per row), zip of at most 20 MB, 5 submissions a day.
- **Prizes:** $13,600 for the top 8; winners must hand in reproducible training code and a short technical report.

## What we built

A training and evaluation pipeline in PyTorch that runs both on a MacBook (Apple M4) and on Colab Pro (A100), with every result checked against the organizers' own scorer.

- **Workspace:** the 34 GB starter pack (data, baseline GRU, official scorer) in a Python 3.11 environment matching the grader's.
- **Fast evaluation:** whole sequences scored in one pass with the official scorer, 50 seconds instead of 15 minutes. It reproduces the baseline's 0.617052 exactly.
- **Training:** stateful GRU/LSTM, minGRU (parallel log-space scan), and causal dilated TCN models trained on full 20,000-step sequences in chunks, memory carried between chunks, loss only on required rows.
- **Multi-model training:** one data stream trains many models side by side, so one Colab run compares 8-16 experiments.
- **Distillation:** a student model learns from a teacher ensemble's predictions instead of (or as well as) the raw targets.
- **Ensembles and export:** models can be averaged and exported as one ONNX file; a generic solution.py runs any of them on one CPU thread.
- **Checks before every submission:** exported ONNX predictions match PyTorch to about 1e-7 for every model type, and a speed test compares each candidate with the baseline.
- **Colab notebooks:** generated from the local code, so Mac and Colab always run the same version. Data downloads are resumable and integrity-checked.

## Results

Each submission so far has beaten the last. Plain validation gains showed up on the leaderboard at roughly half their size; the mask-weighted ones in full or more.

**Leaderboard submissions** (public leaderboard, newest first)

| Submitted | Model | Validation (scored rows) | Leaderboard | Grader time |
| --- | --- | --- | --- | --- |
| Oct 26 | sweep6\_sw\_h128\_s1: 128-unit GRU distilled from the mask-weighted 6-model teacher "w", rows weighted by p(scored) (lambda 0.5), 10 passes | 0.6843 (half B 0.6782) | **0.65660** | 28m11s |
| Oct 24 | sweep5\_m50\_e10: 128-unit GRU distilled from an 8-model ensemble, rows weighted by p(scored) (lambda 0.5), 10 passes | 0.6832 (half B 0.6769) | 0.65444 | 26m52s |
| Oct 24 | st\_big\_e10: 128-unit GRU distilled from an 8-model ensemble, 10 passes | 0.6828 (half B 0.6764) | 0.65396 | 26m21s |
| Oct 23 | d100\_h128: 128-unit GRU distilled from a 4-model ensemble | 0.6821 | 0.65327 | 26m52s |
| Oct 23 | gru128\_wmse, faster ONNX export (same model) | 0.6758 | 0.64972 | 28m26s |
| Oct 23 | gru128\_wmse: 128-unit GRU, metric-matched loss | 0.6758 | 0.64972 | 37m22s |
| Oct 6 | organizers' baseline | 0.6171 | 0.59353 | 27m12s |

**Experiments** (validation on all 1,873 sequences; seed noise is about ±0.0007)

| Sweep | Models | Best result | Verdict |
| --- | --- | --- | --- |
| Mac run | 1 | 0.6758 | metric-matched loss beats the baseline by +0.059 |
| 1 (Colab) | 6 | 0.6778 | loss, input and width tweaks all within noise; pure Pearson loss -0.02 |
| 2 | 8 | 0.6826 (4-model ensemble) | ensembles help, but 5.4x too slow to submit |
| 3 | 16 | 0.6821 (distilled student) | distillation keeps the ensemble's gain at single-model speed |
| 4 | 24 | 0.6828 (student, 10 passes) | bigger or honest teachers add nothing; plateau at \~0.683 |
| 5 | 14 | half B 0.6769 (m50\_e10) | weighting rows by p(scored): +0.005 alone, +0.001 on top of distillation |
| 6 (students running) | 8 + teacher + 7 students | half B 0.6782 (student sw\_h128\_s1; teacher 0.6786) | a mask-weighted teacher lifts its students (+0.0013 half B, +0.0022 leaderboard); new features -0.001 for teachers, feature students still training |

From sweep 5 on, models are compared on validation half B (937 sequences the mask classifier never saw); seed noise there is about +-0.0004.

## What we learned

Distillation, mask weighting and a fast export are what moved the score; tuning single models did not.

**Worked**

- **Metric-matched loss:** clipping targets to \[-2, 2\] and weighting rows by target size, like the metric, beat the baseline by +0.059 validation.
- **Distillation:** a 128-unit GRU trained only on a 4-model ensemble's predictions (0.6821) nearly matches the ensemble (0.6826) at single-model speed.
- **10 passes for the student** instead of 5: +0.0005-0.0010, small but consistent across all three measures.
- **Weighting rows by p(scored)** (a classifier predicts which rows the metric scores, AUC 0.89): +0.005 without a teacher, +0.001 on top of distillation, and the full +0.0005 showed up on the leaderboard.
- **A mask-weighted teacher:** distilling from a 6-model teacher trained with the same p(scored) weighting (half B 0.6786) gave a student at half B 0.6782, +0.0013 over sweep 5, and +0.0022 on the leaderboard (0.65660).
- **Export with one state input per GRU layer:** cut the grader time from 37 to 27-28 minutes with identical predictions.

**Did not help** (all within seed noise unless noted)

- Pearson-only loss (-0.02), wider models (192, 256 units), dropout, row-to-row change inputs, more passes for plain models, EMA weights, longer training chunks at the same batch size, other learning rates.
- LSTM vs GRU at equal cost: LSTM is worse (0.6764 vs 0.6784).
- Bigger teachers: an 8-model teacher (0.6829) gives the same students as a 4-model one.
- Out-of-fold teachers: honest teacher predictions gave worse students (0.6800 vs 0.6815), because each teacher model saw only 3/4 of the data.
- Ensembles of distilled students: they make the same mistakes, so averaging adds nothing.
- Extra causal features (activity of each instrument's prices / volumes / trades, long averages of a2-a5): -0.001 to -0.0014.
- Resetting the GRU inside a sequence (every 1,000-10,000 steps), rescaling predictions per a3 regime or per sequence from its early context.
- A linear reservoir + ridge baseline posted publicly: 0.655 on half A but 0.609 on half B, and no gain in a blend.
- Online adaptation is impossible: the targets cannot be rebuilt from later inputs (best correlation \~0.14).

**Constraints to remember**

- Grader speed is about half of Colab's per-row time; anything above \~1.5x the baseline's time on Colab risks the 60-minute limit.
- The 96-unit student (0.6817) runs at 0.8x baseline time: the fallback if a future model needs headroom.

## Now: closing the gap to the top

The top 20 on the leaderboard sit at 0.669-0.674, about +0.013-0.017 above us, and a public research log describes a single 128-unit GRU, like ours, near 0.672. So the gap most likely lies in how we train, not in architecture or features.

- **Training settings never properly tested:** we update on 256 sequences at a time (\~1,600 updates per pass) and backpropagate 500 steps. Sweep 7 tests 32 / 64 / 256 sequences per update x 500 / 2,000 / 5,000 / 20,000 steps (9 GRUs).
- **Teacher diversity (`arch="mingru"` and `arch="tcn"`):** averaging more GRUs into the teacher hit diminishing returns because they make correlated errors. Sweep 7 adds 2 minGRU models (parallel log-space scan + chrono init; 1.23x baseline ONNX time) and 2 causal dilated TCNs (2,048- and 2,187-row receptive fields) to test whether non-GRU architectures strengthen the teacher ensemble (notes in `research/architectures_cnn_rnn_transformer.md`).
- **Diagnostics so far:** shuffling i1 costs -0.044 (mostly its trades), a0-a7 -0.052, i0 -0.338. High-a3 rows score 0.35 vs 0.81 for low a3, and the scored rows sit exactly there. Per-sequence level and scale are worth up to +0.036 in an oracle, but they cannot be predicted from a sequence's early context.
- **Compute:** Colab compute units ran out on Oct 25. Since Oct 26 the sweep 6 students run on a free T4: group 1 is done; group 2, the first students with the extra features, ran out of RAM (the features were computed for 256 sequences at once) and resumes with the fix (16 at a time, identical results). Sweep 7 still needs an A100.

The scoring-mask findings behind the mask weighting:

**What the mask looks like** (200 validation sequences)

- Scored rows come in short bursts: runs of median length 2 rows, gaps of median 6 rows, spread evenly along each sequence.
- At scored rows the targets are larger (mean |t0| 0.74 vs 0.64) but change less from row to row (0.037 vs 0.056).
- Extra features a3, a2 and a4 differ most (0.72, 0.53 and -0.38 standard deviations); i0 trades more and its book moves less, while i1 moves more.
- Targets are quantized: about 40 distinct values per sequence.

**How predictable it is** (gradient boosting, trained on 200 validation sequences, tested on 60 others)

| Inputs | AUC |
| --- | --- |
| all (112 features, row-to-row changes, 20-row average of a2-a4) | 0.871 |
| 112 raw features only | 0.866 |
| a0-a7 only | 0.799 |
| order book and trades only | 0.786 |
| a2, a3, a4 only | 0.720 |

The 10% of rows rated most likely hold 51% of all scored rows; the top 25% hold 77%. The strongest single signal is the 20-row average of a3.

## Plan

Next are the remaining sweep 6 students (running on a T4) and sweep 7 (waiting for A100 units), then a final model trained on all labeled data before the deadline.

- [x] **Submit st\_big\_e10** (validation 0.6828): submitted Oct 24, leaderboard 0.65396.
- [x] **Sweep 5: mask-weighted distillation:** sweep5\_m50\_e10 submitted Oct 24, leaderboard 0.65444.
- [ ] **Sweep 6 students:** 7 students of the mask-weighted teacher in 3 groups, each saved and zipped when done (`connectome_sweep6_students_t4.ipynb`, free T4). Group 1 done: sw\_h128\_s1 submitted Oct 26, leaderboard 0.65660. Groups 2-3 (feature students) training.
- [ ] **Sweep 7: training dynamics + TCN & minGRU teachers** (13 mask-weighted models: 9 GRUs across sequences-per-update x backprop length + 2 minGRUs + 2 causal TCNs). Ready as `connectome_sweep7.ipynb`, `connectome_sweep7_launcher.ipynb`, and `connectome_sweep7_onecell.py`; needs an A100 (compute units).
- [ ] **Rebuild teacher and students** with sweep 7's best training recipe and any diversity-adding TCN/minGRU teachers.
- [ ] **Final model:** retrain the best recipe on train + validation (+23% data), with fixed seeds and reproducible training code.
- [ ] **Choose final submissions** by local validation, not the public leaderboard, which scores only a sample.
- [ ] **Prize paperwork if we place top 8:** training code with fixed seeds and a 2-8 page technical report.

Open question: what training change gets a 128-unit GRU from 0.657 to \~0.67 on the leaderboard? A stronger mask-weighted teacher and sweep 7's training settings are the most likely candidates.

## Where everything lives

Google Drive holds the notebooks and every trained model; the Mac holds the source code and the data.

**Google Drive** (`MyDrive/`)

```text
wunder_challenge/                   Colab notebooks: sweeps 1-6, connectome_sweep7.ipynb, connectome_sweep6_students_t4.ipynb; PROGRESS.md
wunder-connectome/runs/             everything the notebooks write (OUT_DIR)
├── e01_wmse_h128/ … e06_…/         sweep 1 (+ summary.jsonl)
├── sweep2/, ensemble_sweep2/        sweep 2 and its 4-model ensemble (teacher "old")
├── sweep3/, ensembles_sweep3/       sweep 3 (d100_h128) and budget ensembles
├── sweep4_teachers/, teacher_big/   out-of-fold teacher models; the 8-model teacher "big"
├── sweep4/, ensembles_sweep4/       sweep 4 students (st_big_e10)
├── mask_model/                      scored-row classifier (model.joblib), split.json (validation halves A/B), report.json
├── cache/                           mask_train.npy (p(scored) per train row), t_big.npy, t_w.npy (teacher predictions)
├── sweep5/                          mask-weighted students, incl. m50_e10 (leaderboard 0.65444)
├── sweep6_teachers/, teacher_w/     8 mask-weighted teacher models and the 6-model teacher "w"
├── sweep6_students/                 students of teacher w, incl. sw_h128_s1 (best leaderboard, 0.65660)
└── zips/                            upload-ready zips (solution.py at the zip root)
```

Every model folder has the same layout: `best.pt` (PyTorch checkpoint with its config), `log.jsonl` (training log), `results.json` (full-validation, holdout and, from sweep 5, half-B scores) and `submission/` (`solution.py` + `model.onnx`, ready to zip). The mask weights and the teacher predictions still in use (`t_big.npy`, `t_w.npy`, \~850 MB each) are cached in `cache/`; the dataset lives only on Colab's local disk and is downloaded again each session.

**Mac** (`~/Desktop/wunder-connectome/`)

```text
pipeline/        training, evaluation, ensembles, distillation, mask model, features, ONNX export, speed test (source of truth)
colab/           build_notebook.py generates the notebooks from pipeline/; one-cell launchers (*_onecell.py, *_t4.py); finished runs
analysis/        mask_classifier.py, error_slices.py, seq_calibration.py, reservoir_ridge.py
datasets/        train / valid parquet files (33 GB)
baseline/        the organizers' baseline
runs/            Mac runs (gru128_wmse and its submission zips)
research/        prior-work notes + architecture reference (TCN, GRU, minGRU, Transformers)
WORKSPACE.md     running notes, commands and status
```

To stay in sync: code changes are made in `pipeline/` on the Mac and reach Colab by regenerating the notebooks; results flow back through `wunder-connectome/runs/` in Drive.
