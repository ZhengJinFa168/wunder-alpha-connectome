# Wunder Alpha Connectome — workspace

Docs: https://wundernn.io/connectome/docs/quick_start

## 1. Starter pack (done)
Unpacked into this folder: `datasets/` (33 GB: train 10,607 seqs, valid 1,873 seqs), `baseline/`,
`utils.py` (DataPoint + official scorer), `METRIC.md`, `docs/`, and the pack's own `README.md`.
Re-download if needed: https://files.wundernn.io/wnn_connectome_starterpack.tar.gz (33.7 GB, needs login on the site).

## 2. Environment (done)
Python 3.11 venv in `.venv` (the scorer uses python:3.11-slim-bookworm): pack pins from `requirements.txt`
(numpy 2.2.6, onnxruntime 1.23.2, pyarrow 19.0.1) plus a lean dev set from `requirements-dev.txt`.

    source .venv/bin/activate

`requirements-scorer.txt` is the full scorer package list, for reference: only import what's in it.

## 3. Official baseline (expect WP ≈ 0.617052, ~15 min on this Mac)
    python baseline/solution.py --validation datasets/valid.parquet
    python evaluate.py baseline --max-seqs 20     # quick check: 0.662 on the first 20 seqs

`baseline/baseline_submission.zip` is ready to upload as-is.

## 4. Your own model
- Edit `my_solution/solution.py` (`PredictionModel.predict(data_point)`; put weights in `my_solution/`, load them relative to `HERE`).
- Quick local score on a subset: `python evaluate.py my_solution --max-seqs 50` (uses the official scorer from `utils.py`, and reports µs/row against the time limit).
- The template's order-book-imbalance signal is only a placeholder: it scores about -0.10 on 20 seqs.
- Package: `./make_submission.sh` → `submission.zip`, then upload on the submit page.

## 5. GRU training pipeline (`pipeline/`)
    python -m pipeline.train --name NAME [--hidden 128 --layers 2 --diff --dropout 0.1 --loss wmse|pearson|mix --epochs 4]
    python -m pipeline.fast_eval --ckpt runs/NAME/best.pt --holdout 500
    python -m pipeline.fast_eval --baseline               # reproduces 0.617052 in ~50 s
    python evaluate.py runs/NAME/submission --max-seqs 20 # row-by-row check of the exported ONNX
    ./make_submission.sh runs/NAME/submission

- `data.py`: 4 reader threads, one ParquetFile each (re-opening parses a big footer), prefetches the next batch.
- `train.py`: batches of whole sequences, truncated BPTT in `--chunk` steps with hidden state carried; loss on steps >= 99.
  Holdout = 500 train seqs (fixed seed). Checkpoint selection = mean(valid scored WP, holdout all-rows WP).
- `model.py`: GRUNet, MinGRUNet (`arch="mingru"`), TCNNet (`arch="tcn"`), losses (wmse / weighted-Pearson / mix), ONNX step export, baseline ONNX -> PyTorch import.
- Speed on this M4 (MPS): ~10 s per 256-seq batch, ~7 min per full pass over train.
- Verified: imported baseline gives 0.617052 (fast eval) and 0.662020 (row-by-row on 20 seqs), identical to the official files.
- Baseline on *all* rows is only 0.440 (valid) / 0.414 (holdout): the scoring mask selects easier moments.

### Multi-model training, ensembles, speed
    python -m pipeline.multi_train --config experiments.json --epochs 8 --cache-eval   # N models, one data stream
    python -m pipeline.ensemble --ckpts "runs/*/best.pt" --max-members 4 --export runs/ens/submission
    python -m pipeline.speed runs/NAME/submission runs/ens/submission                  # us/row vs baseline
- Ensemble members are rescaled to a common spread, chosen greedily on the train holdout, exported as one ONNX
  (inputs h_0, h_1, ...; the same solution.py handles single models and ensembles).
- Don't put the official baseline in ensemble selection: it was probably trained on our holdout sequences.
- Our solution.py costs 1.03x the baseline per row for the same network; a 2x128 ensemble ~1.9x.

### Leaderboard (public)
| submission | score | grader time |
|---|---:|---:|
| baseline_submission.zip | 0.59353 | 27m12s |
| submission_gru128_wmse.zip | 0.64972 | 37m22s |
| submission_gru128_wmse_v2.zip (same model, new export) | 0.64972 | 28m26s |
| d100_h128.zip (sweep 3, distilled from the sweep-2 ensemble; valid 0.6821) | 0.65327 | 26m52s |
| sweep4_st_big_e10.zip (8-model teacher, 10 passes; half B 0.6764) | 0.65396 | 26m21s |
| sweep5_m50_e10.zip (mask-weighted lambda 0.5, 10 passes; half B 0.6769) | 0.65444 | 26m52s |
| sweep6_sw_h128_s1.zip (distilled from mask-weighted teacher w, lambda 0.5, 10 passes; half B 0.6782) | 0.65660 | 28m11s |
| sweep10a_routed.zip (g1-3 ft_gk specialists, g4 ft_film~wise; half B 0.67892) | **0.65693** | 31m12s |
Grader is ~2x slower than the M4 per row; our old export had 19 ONNX nodes (Slice/Concat on a stacked state) vs
the baseline's 10. New export: one state input per layer (11 nodes for a single model). `submission_gru128_wmse_v2.zip`
= same model, new export: 24% faster on the grader, identical score.
Leaderboard gain so far ~ half the validation gain (d100: +0.0063 valid -> +0.0036 LB), except the first
mask-weighted model: +0.0005 on half B -> +0.0005 LB (hint that the test mask is chosen like the valid mask).
Rule of thumb: grader time ~ 0.52 x Colab speed.py time (baseline 41 vs 80 us/row); keep Colab <= ~140 us/row.

### Sweep 3 tools
- `pipeline/teacher.py`: ensemble predictions on train -> .npy for `--teacher`; ExpConfig `distill` = teacher-loss weight.
- ExpConfig `cell` (gru|lstm), `chunk` (per-model TBPTT), `ema` (EMA weights evaluated/saved).
- `pipeline/ensemble.py --budgets 1 1.5 2`: one greedy ensemble per compute budget (x baseline multiply-adds).
- `pipeline/test_export.py`: ONNX + solution.py == PyTorch for GRU / diff / LSTM / minGRU / TCN / mixed ensembles.

### Colab
`colab/build_notebook.py` writes `connectome_multi.ipynb` (sweep 1), `connectome_sweep2.ipynb`, `connectome_sweep3.ipynb` with the
current `pipeline/` code embedded. Re-run it after code changes. Results land in Drive `wunder-connectome/runs/`.
Sweep 1 (A100, 4 passes, 22 min for 6 models): mix loss / diff inputs / h192+dropout each +0.003-0.005 holdout
over plain wmse (0.4471); pure Pearson loss -0.029. e01 reproduced the Mac run within 0.0002.
Sweep 2 (8 passes): seed noise 0.0007 valid-scored / 0.005 holdout, so sweep-1 "gains" were noise; 8 passes = 4 passes;
width 192/256 no gain. Greedy 4-model ensemble: valid 0.6826 / all-rows 0.5150 (singles ~0.6775 / 0.505), but 5.4x
baseline time on Colab's CPU (baseline 80 us/row there).

## Research
- `research/prior_work_gemini.md`: prior Wunder competitions and related work (Gemini 3.8 Flash via agy, with my review notes on top).
- `research/architectures_cnn_rnn_transformer.md`: 1D CNN (TCN) vs. GRU/LSTM vs. minGRU vs. Transformers under Alpha Connectome's online 1-vCPU constraints, measured ONNX step latencies, and post-Sweep-7 teacher/student roadmap.

## Key facts
- Each Parquet row group = one sequence of 20,000 rows. Steps 0–98 are warm-up (return `None`); steps 99–19,999 need a `(2,)` float32 `[t0, t1]`.
- Reset state when `seq_ix` changes; update state on every row; only use current + past rows.
- 112 features in fixed order: i0 bid p (0–10), ask p (11–21), bid v (22–32), ask v (33–43), trade dp (44–47), dv (48–51); same for i1 at 52–103; extra a0–a7 at 104–111. Index ≠ book depth.
- Metric: weighted Pearson per target with targets/preds clipped to [-2, 2] and weights |y|; score = mean of t0, t1 (official scorer in `utils.py`, spec in `METRIC.md`).
- Limits: 1 vCPU (AMD EPYC 9224, AVX-512), 16 GB RAM, no GPU, no internet, 60 min for the full test set (test = 1,970 seqs = 39.4M rows → ≤91 µs/row on their CPU), zip ≤ 20 MB.

### Sweep 5: mask-weighted distillation
- `pipeline/mask_model.py fit` trains a classifier for is_scored on validation half A (split.json, seed 5), AUC on half B
  (~0.89 with 150 trees); `score` writes p(scored) for every train row (float16 [10607, 20000]).
- ExpConfig `mask_weight` (lambda) + `mask_power`: row weight (1 - lambda) + lambda * p^power / mean(p^power).
- Common `--valid-split split.json`: periodic eval + checkpoint selection on half B, summary adds valid_B_scored/_all.
- `fast_eval --valid-split` scores any checkpoint on half B (to compare with earlier models fairly).

### Diagnostics (gru128_wmse, 300 valid seqs) -- analysis/error_slices.py
- GRU resets inside a sequence never help (reset every 1k/2k/5k/10k steps: -0.0019 / -0.0002 / -0.0005 / -0.0004).
- Block-swap ablation: i0 -0.338, a0-a7 -0.052, i1 -0.044 (i1 trades -0.021, volumes -0.005, prices -0.002).
- Targets can't be rebuilt from future feature changes (best corr ~0.14), so online target-based adaptation is out.
- Weak slices: high a3 rows WP 0.35 (low a3 0.81); sequences with 25-39% scored rows 0.37; calmest 20% of sequences 0.48.
  Scored rows sit where the model is weakest -> supports mask-weighted training (sweep 5).
- Calibration slope 0.61 (overconfident), but per-a3-band rescaling on held-out sequences: -0.001 (no gain).

### Sweep 6 tools + reservoir check
- `pipeline/features.py`: 36 extra causal features (group activity of i0/i1 prices, volumes, trades + EMAs of activity
  and a2-a5 at spans 20/200/2000), standardised with fixed constants; offline (lfilter) == online (solution.py ships
  wnn_features.py) to 3e-8; online cost 4.5 us/row (Mac), a 64-unit feature model ran 25 vs 16 us/row.
- ExpConfig `feats`; `ensemble.py --valid-split S --select-on validA` (pick on half A scored, report half B).
- Reservoir + ridge ("pure NumPy baseline", 1024 random EMA units): half A 0.655 but half B 0.609 (our GRU 0.676);
  prediction corr with GRU 0.88, best blend +0.0003. Not useful. Script: analysis/reservoir_ridge.py.

### Status 2026-10-25 (waiting for Colab GPU)
- Colab compute units ran out during sweep 6 phase 3 (students; stopped at pass 1/10). Phase 1 (8 mask-weighted teacher
  models: features -0.001..-0.0014 on half B, lambda 0.75 worse) and phase 2 (teacher_w, 6 models, half B 0.6786;
  cache/t_w.npy) are saved in Drive. Free T4 also refused (usage limits).
- Ready in Drive `wunder_challenge/` (browser account; the Drive connector is linked to a different account):
  `connectome_sweep6_students_t4.ipynb` (all 7 students in 3 priority groups, each saved + zipped when done; T4 settings)
  and `connectome_sweep7.ipynb` (training dynamics + minGRU/TCN teachers, 13 models; needs A100-class GPU). Both have one-cell/launcher
  variants (`colab/*_onecell.py`, `*_t4.py`, `connectome_sweep7_launcher.ipynb`) that pull base files from `connectome_sweep6.ipynb` in Drive.
- Best submission: sweep10a_routed = 0.65693 LB (2026-11-04; was sweep6_sw_h128_s1 0.65660, sweep5_m50_e10 0.65444). Top 20 on the LB: 0.669-0.674.
- Keep a browser tab connected to Colab while a run is going (Colab Pro has no background execution).
- 2026-10-26, free T4: students group 1 done (sw_h128 half B 0.6780, sw_h128_s1 0.6782; zips in runs/zips/). Group 2
  (first feature students) ran out of host RAM: features.offline worked in float64 on the whole 256-seq batch (~14 GB
  peak). Fixed: 16 sequences at a time (identical output, ~1.6 GB), loader drops its batch reference, extras joined on
  the GPU. The T4 launcher patches the sweep-6 code with these and skips students that already have results.json.
- 2026-10-26, Sweep 7 extended with `MinGRUNet` (`arch="mingru"`, parallel log-space scan + chrono init, 1.23x baseline ONNX time)
  and causal dilated `TCNNet` (`arch="tcn"`, exact chunk-boundary state buffers, 2,048–2,187-row receptive field):
  13 models total (9 GRU training-dynamics runs + 2 minGRUs + 2 TCNs) in `colab/connectome_sweep7.ipynb`,
  `colab/connectome_sweep7_launcher.ipynb`, and `colab/connectome_sweep7_onecell.py`.

### Sweep 7 part 1 (2026-10-28, A100, 23 min): training dynamics, 9 plain mask-weighted GRUs, 4 passes
Half B: b256_c500 (our default) 0.6738 > b32_c2000 0.6732, b32_c5000 0.6731, b64_c500 0.6729, b32_c500 lr1e-3 0.6725,
b64_c2000 0.6725, b256_c2000 0.6715, b32_c20000 0.6704, b32_c500 0.6688. Smaller batches / longer backprop do not
help: training dynamics are not the leaderboard gap. Small batches want a lower lr (b32: 1e-3 beats 2e-3 by +0.004).
The 4 minGRU / TCN models have not run yet: colab/connectome_sweep7_onecell.py now skips finished models.

### Sweep 8 (ready 2026-10-28): what never ran in sweeps 6 and 7, one A100 run
`colab/connectome_sweep8.ipynb` (one cell = `colab/connectome_sweep8_onecell.py`): 2 minGRU + 2 TCN (plain, 4 passes) and
the 5 feature students of sweep 6 (teacher w / big, 10 passes), one data stream (per-model `epochs`). Skips models with
results in runs/sweep7, sweep6_students or sweep8. Ends with a combined table, zips (runs/zips/sweep8_*.zip), an
ensemble check of plain GRUs + new architectures, and a speed test. Tested end to end on tiny models on the Mac.

### Sweep 8 results (2026-11-03, A100 80 GB, ~2.5 h)
Half B: sw_feat_h96 0.6778 (cost 0.61), sw_feat_h128 0.6778, sw_feat_h128_m75 0.6776, sw_feat_h128_s1 0.6773,
sb_feat_h128 0.6763 | minGRU h192 l3 0.6700, b32_c5000 0.6688, TCN c128 k3 l7 0.6695 (cost 2.48), TCN c64 l11 0.6668.
References: sw_h128_s1 0.6782 (LB 0.65660), plain GRU b256_c500 0.6738 (same 4-pass recipe as minGRU/TCN).
-> extra features add nothing to students either; teacher w > teacher big; minGRU / TCN worse than GRU.
Nothing to submit. Ensemble + speed steps were still running when the notebook was saved (not seen yet).

### Sweep 9 tools (2026-11-03)
- Feature kinds (`pipeline/features.py`): `act` (the 36 activity/EMA features), `norm` (each of the 112 inputs as a
  causal z-score against its running mean/std since the sequence start; offline == online exactly, 4.4 us/row),
  `act+norm`. ExpConfig `feats`: kind name (True = act). Models record `feat_kind`; evaluation, ensembles and teachers
  feed every model its own kind (extras once per batch and kind). solution.py picks the kind from model.onnx's x width.
- Auxiliary targets (`trainer.aux_targets`, ExpConfig `aux` = loss weight, GRU only): future change of i0's / i1's
  average price column at 10/50/200/1000 and 50/200 rows, and i0's mean |change| over the next 100 rows; scaled per
  batch, trained by a separate head that is not exported.
- `ensemble.py --export` always writes ensemble.json; the ONNX only when members share one feature kind.
- Cost model counts the extra inputs. test_export: 16 graph types OK (incl. norm, act+norm, aux head, norm ensemble).
- `colab/build_sweep9.py` writes `colab/connectome_sweep9.ipynb`: one self-contained cell (pipeline embedded).
  Phase A runs/sweep9_teachers (6 teacher members + aux/norm tests, 5 passes); phase B teacher t9 (greedy on half A
  over sweeps 5-9 plain models, runs/teacher_9, cache/t_9.npy) and born-again teacher ba (6 students, cache/t_ba.npy);
  phase C runs/sweep9 students (10 passes), zips runs/zips/sweep9_*.zip.

### Sweep 9 results (2026-11-04, A100, ~5 h)
Teachers, half B: t9_h128 seeds 0.6745-0.6763, h192_drop 0.6754/0.6761, aux03 0.6749, aux1 0.6748, norm 0.6715,
norm_aux03 0.6721. Teacher 9 (greedy 9 of 32, half A) half B 0.6784 = teacher w 0.6786: no stronger teacher.
Students, half B: s9_h128 0.67824, s9_h128_s1 0.67823, s9_aux03 0.67801, sba_h128 0.67801, s9_norm 0.67768 (1.71x
speed, too slow) vs sw_h128_s1 0.67816. -> distillation / ensembling / seeds saturated at ~0.678. Nothing to submit.
The aux targets were flawed: the 22 price columns of an instrument have mixed signs (corr with p0 +0.98 .. -0.91), so
their average cancels; aux targets corr with t0 - t1 ~0.01.

### Data structure found 2026-11-04 (checked on valid; prompted by an outside review)
- 4 instruments by tick size: min |t0| per seq in {0.1254, 0.2413, 0.4685, 0.6569}.
- gru128_wmse on 300 valid seqs, scored rows: group 1 (22% seqs, 27% scored rows) 0.789, group 2 (20%, 15%) 0.787,
  group 3 (20%, 7%) 0.693, **group 4 (38% seqs, 51% scored rows) 0.386**; global 0.660.
- Scaling group-4 predictions hurts (x0.7 0.654, x1.5 0.646, x2 0.632): MSE shrinkage is right for pooled WP.
- Row-0 nearest-centroid group accuracy 0.85 (group visible from the first row).
- t0 > 0 and t1 > 0 never together; corr(t0, t1) -0.69. Inputs clipped at +-5.1993 (quantile transform to normal).
- Full data: train groups 1305/1257/3488/4557 seqs (12/12/33/43%), valid 398/347/431/697 (21/19/23/37%): the mixes differ.
- gru128_wmse, 624 valid seqs: per group 0.786 / 0.764 / 0.707 / 0.413; pooled at valid mix 0.671, train mix 0.621,
  equal mix 0.706 -> the group mix moves the pooled score a lot (likely part of the LB 0.657 vs half B 0.678 gap).
- Group detection from warm-up rows 0-98 (mean/std of the inputs): logistic 99.6-99.7% of valid, boosting 99.9%;
  identified mostly by volume levels (i0_v17, i0_v0, i0_v9, i0_v3). Plan: PLAN_sweep10.md.

- Mask model with non-causal inputs (300 A -> 300 B seqs, same GBM): causal AUC 0.887 / AP 0.527; + targets 0.890 / 0.533;
  + next-20-row activity 0.891 / 0.535; both 0.892 / 0.539. y == 0 share scored 22.2% vs unscored 22.0%: the mask
  barely depends on the targets -> small gain at most.

- Input scales (51 valid seqs): std from 0.038 (i0_p12; ~0.03 within a sequence) to 2.94 (a7). Rows at the -5.1993
  floor (quantile-transform minimum): a7 44%, a6 18%, a5 11%. -> candidates for 10b: fixed input standardisation
  (folded into the first layer at export) and Phi(x) inputs.

- Microstructure features from the LOB literature (queue imbalance, integrated OFI tau 10/25/100, spread x imbalance,
  VPIN, Kyle's lambda, mid dislocation, i1 - i0; 28 features in rank space Phi(x), directions fitted per group on half A),
  stacking test (GBM on 208 A seqs -> 208 B seqs, gru128_wmse preds): features alone B 0.620 (GRU 0.665) = real signal;
  GRU + group 0.6632 -> + features 0.6513: g1 +0.004, g2 +0.002, g3 0, g4 0.405 -> 0.365 (worse; g4-only stack 0.354
  vs GRU 0.4135). Mostly overlaps with what the GRU extracts; group-4 relations do not transfer across sequences.
  The stack learns from ~200 sequences, a student from 10,607: a feature-input student is the remaining fair test (10b).

### Sweep 10a tools (2026-11-04)
- `pipeline/groups.py`: `label(Y)` (group from the targets), `--fit` (detector into features.py: logistic regression on
  mean/std of rows 0-98, train 99.85%, valid 99.95% = 1 error at confidence 0.51), `--check`.
- features: kind `grp` (4 detector probabilities, causal, frozen from row 98), `detect(X)`, TICKS, DET_ROWS.
- model: `GroupGRUNet` (arch "ggru", feats "grp"): FiLM on the 112 inputs + 4 heads mixed by the probabilities;
  `from_gru` starts it as exactly a plain GRU.
- trainer: ExpConfig `group` (train on one group, select on its rows), `init` (start from a best.pt); every eval and
  summary row has per-group scores (scored_g0..3 / B_g1..4).
- `pipeline/routed.py`: per-group scores of all candidates, route = best per group on half A, routed half-B score with
  true and detected groups, export (model_k.onnx + routes.json + solution_routed.py as solution.py + wnn_runner.py).
  Routed solution: all models run on warm-up rows, the detector picks one at the first scored-capable row.
- test_export: all OK incl. ggru and 3 routed configurations. `colab/build_sweep10a.py` -> connectome_sweep10a.ipynb
  (dry-run twice locally: fresh and skip-everything).
- 10a patched after review: fine-tunes distil 0.5 (half true targets), group-4 models g4y_* learn from the targets only
  (5 passes, same recipe as t9_h128_s* / t9_h192_drop_s*, which routed.py also scores per group). routed.py options per
  group: single model, blend with the shared model (prediction average, exported as a 2-member ensemble; --max-cost 2.2
  per sequence), weights halfway to the shared model (~wise, for fine-tunes); reports average cost and a per-group
  affine calibration (not exported). solution_routed buffers the warm-up rows and replays them through the chosen model.
- Offline test (gru128_wmse, 150 A / 150 B seqs): a linear ridge "skip" from (x_t, x_t - x_t-1) on the GRU residual
  hurts (B 0.6535 -> 0.6385 global, 0.6083 per group); per-group affine calibration +0.0005.

### Sweep 10a results (2026-11-04, A100 40 GB in another Colab account, Drive via rclone: colab/connectome_sweep10a_rclone.ipynb)
Training is fast: 6 passes of 8 models in ~27 min. Scored WP, half A (clean for single models) / half B
(43% of B = first 400 seqs picks the checkpoints, so B is biased upward for models fine-tuned from an already-selected one):
| model | A | B | B_g1 | B_g2 | B_g3 | B_g4 |
|---|---|---|---|---|---|---|
| s9_h128 (start) | 0.69150 | 0.67824 | 0.7944 | 0.7672 | 0.7233 | 0.4205 |
| ft_all | 0.69153 | 0.67933 | 0.7959 | 0.7676 | 0.7238 | 0.4224 |
| ft_film (1.36x slower) | 0.69264 | 0.67864 | 0.7967 | 0.7694 | 0.7251 | 0.4182 |
| routed g1-3 ft_gk, g4 ft_film~wise | (0.69316, route picked on A) | 0.67892 | 0.7965 | 0.7709 | 0.7262 | 0.4204 |
- Specialists gain on groups 1-3 on both halves (g2 +0.003/+0.004, g3 +0.0025/+0.003, g1 +0.003/+0.002), but these
  groups carry little of the pooled |y| weight, so routed pooled B is only +0.0007. Detector: same score true/detected.
- Group 4 cannot be specialised: ft_g4, g4y_h128 (0.4121 vs t9_h128 0.4169-0.4199), g4y_h192_drop (0.4173 vs
  0.4156/0.4179) are all <= the shared models. Group 4 alone does not beat training on all groups; capacity does not help.
- ft_all: +0.0011 on B but +0.00003 on A -> longer training gives nothing; the B gain is checkpoint-selection noise.
  For 10b: report A (or B minus the selection seqs) as the honest number.
- Per-group calibration +0.0004 (not exported). Speed: routed 1.03x baseline (~27 min on the grader).
- Decision rule (routed B >= 0.6792) not met; routed zip is a cheap LB probe (zips/sweep10a_routed.zip in Drive).
- LB probe: 0.65693 (+0.00033 over 0.65660) for +0.0007 on half B: again ~half the validation gain reaches the LB.
  Grader time 31m12s (vs 28m11s): the routed export runs every model on the warm-up rows.
- rclone: one mid-training upload failed (file being written, md5 mismatch), later uploads fine. rclone warns its
  shared Google client_id retires during 2026; make an own client_id if it stops working.

### Group 4 targets (2026-11-05, valid + every 4th train group-4 sequence)
- Share of the metric weight (sum |y| over scored valid rows): g1 28%, g2 16%, g3 8%, **g4 48%**.
- Group 4's (t0, t1) take only 40 joint values (in ticks, clipped): (0,-1) 29%, (-1,0) 29%, (-1,-1) 9.6% (but only
  0.9% of scored rows), (1,-3) 8.3%, (-3,1) 8.0%, (-3,2)/(2,-3) 3.5% each, ... The two targets are tied: one latent
  signed move, mapped non-linearly to both (t0 = -3 ticks whenever t1 >= +1, and so on). Rows: target unchanged from the
  previous row 78% of the time.

### Sweep 10b tools (2026-11-05)
- `groups.STATE_TICKS[3]` (the 40 group-4 states) and `state_values(g)`; `model.StateHead`: softmax over the states,
  prediction = sum p v|v| / sum p |v| per target (the |y|-weighted MSE optimum); GRUNet(states=...) stores it in its
  config. Trainer: ExpConfig `states` (group of the table), `cls` (weight of the |y|-weighted state cross-entropy, added
  to wmse / distillation). `init` from a plain model keeps the body, the head starts new. test_export OK (incl. blend
  and routed). routed.py skips ~wise for these (different keys).
- `colab/build_sweep10b.py` -> connectome_sweep10b(.ipynb | _rclone.ipynb): 7 group-4 models (12-pass from-scratch
  h128 / h192_drop, ft_g4_y, ft_g4_c2000, 3 state-head models), routing over 10a + 10b, export sweep10b_routed.

### Group-4 feature screen (2026-11-05, Mac, no GPU): nothing helps
s9_h128 predictions + 556 causal candidates on the 697 group-4 valid sequences (every 6th scored row, 142k rows half A /
144k half B): volumes as quantile ranks Phi(v) (44), all pairwise imbalances (Phi(v_j)-Phi(v_k))/(sum) within each
instrument (462), rows since the last price change + event-rate EMAs (6), rows since each volume column changed (44).
Linear |y|-weighted stack [1, pred, f] fitted on A, scored on B against the refit [1, pred] (s9_h128 g4: A 0.4472, B 0.4212).
- Best A gains +0.003 (all involve i0_v11) but every one is negative on B (-0.0005 .. -0.005); top-10 stack A +0.0043,
  B -0.0048. Per kind (top 10 by A): imbalances B -0.0017, ranks -0.0004, price age -0.0009, event rates -0.0007,
  volume ages -0.0008. Null (rolled features): A gain max 0.00075, B mean -0.0002 (sd 0.0004).
- So the GRU already uses what these carry (incl. timing since the last move); the half-A gains are half-specific.
  Feature engineering for group 4 dropped. Note: the i0_* price columns are scrambled and partly sign-flipped
  (corr with p0 from +0.99 to -0.95), all change on the same rows (39% of group-4 rows).

### Sweep 11 plan (2026-11-05): use valid (period shift)
Train vs valid sequences are separable per group (AUC ~0.97-0.99): valid (and probably test) is another period. Training
on valid is allowed (FAQ: train only on train.parquet + valid.parquet). Honest check: fit on half A -> score B, and B -> A.
- Main: fine-tune s9_h128 (and the g1-3 specialists, keep routing) on valid with the true mask: |y|-weighted MSE,
  unscored-row weight {0.06, 0.3, 1.0}, lr {2e-4, 5e-4}, 3 epochs + average of epochs 2-3, distill on/off.
- Control 1: from scratch on train + half A (half A weighted ~3x), scored on B -> if close to / adds to fine-tuning,
  rebuild teacher + students on train + valid, then fine-tune on valid at the end.
- Control 2: from scratch on half A only, scored on B (expected to overfit; tells how big the shift is).
- Later: own numba GRU step to fit fine-tuned ensembles in the 60 min limit (routed now 31 min); hedge pick 0.9 FT / 0.1 base.
- Dropped: group-4 work (10b), bid/ask mirroring, target reconstruction, per-group calibration.

### Sweep 11 tools (2026-11-05)
- Trainer: Common `pool` (train | valid | train+valid, `data.PoolStore`: keys VOFF*(copy+1)+v for validation sequences,
  validation part cached in RAM, 17 GB), `teacher_valid`, `skip_final_eval`, `pool_valid_seqs`. ExpConfig `valid_part`
  (A | B | all of split.json), `use_train`, `valid_repeat`, `unscored_weight` (validation rows: true mask, scored 1,
  unscored u, normalised to mean 1), `valid_distill` (teacher target on validation rows = (1-v) clip(y) + v teacher_valid),
  `select` (False: no checkpoint selection; required when trained on validation), `swa_from` (average of end-of-pass
  weights), `sched` (onecycle | const).
- `pipeline/crossfit.py score`: NAME_A / NAME_B pairs, each predicts the half it did not see -> A, B, pooled all-valid,
  per group; base s9 + 10a specialists; FT routed (best shared recipe, group recipes where they win); --hedge a scores
  (1-a) init + a model. `export`: routed submission from 4 given models (+ --alpha interpolation towards init).
- `colab/build_sweep11.py` -> `colab/connectome_sweep11(_rclone).ipynb`: part 1 = sweep 10b phase A; part 2 = 11a
  fine-tunes (9 recipes x A/B), 11b controls (c1_trv, c2_v, c3_trv_all), 11c cross-fit, 11d chosen recipes on all of
  valid -> zips sweep11_routed / _hedge / _single / _c3; then 10b's routing.
- Importance weights (`pipeline/shift.py`, ExpConfig `seq_weights`): per-sequence stats (mean, log std, log mean |diff|
  of the 112 inputs), per group logistic regression train vs validation half H, out-of-fold p for train, weight
  (p/(1-p))^gamma normalised per group, cap 5. Test on 800 train seqs + all valid: honest AUC (train vs the other half)
  0.985-0.998 in every group; effective sample gamma 0.5: 0.31-0.43, gamma 0.25: ~0.59. Sweep 11 trains w05 / w025
  (A/B pairs, s9 recipe on train only) and c0_tr (same recipe, no weights, final weights) as their reference.

## Sweep 11 results (run 2026-11-05, A100, ~3.5 h incl. 10b)
Cross-fitted scored-row WP over all valid (each sequence scored by a model that never saw it):
| model | all | g1 | g2 | g3 | g4 |
|---|---|---|---|---|---|
| s9 (base) | 0.68483 | 0.7985 | 0.7677 | 0.7288 | 0.4329 |
| base routed (10a spec g1-3, s9 g4) | 0.68588 | 0.8010 | 0.7712 | 0.7315 | 0.4329 |
| v_lr2_u06 (shared FT on valid) | 0.68286 | 0.8006 | 0.7705 | 0.7316 | **0.4219** |
| v_lr2_u06_anc~0.9 (best shared) | 0.68460 | 0.8008 | 0.7700 | 0.7316 | 0.4284 |
| vg{1,2,3}_lr2_u06~0.9 (group FT on valid) | - | **0.8031** | **0.7734** | **0.7340** | - |
| FT routed hedge 0.9 (g4 = anc) | 0.68614 | 0.8031 | 0.7734 | 0.7340 | 0.4284 |
| c0_tr (train only, fixed recipe) | 0.68483 | | | | |
| w05 / w025 (importance weights) | 0.68481 / 0.68482 | | | | |
| c1_trv (train + valid half, from scratch) | **0.62431** | 0.7840 | 0.7400 | 0.6592 | 0.2966 |
| c2_v (valid half only, from scratch) | 0.52620 | | | | |
- Fine-tuning on valid: +0.002 to +0.003 on g1-3, but g4 (48% of the metric weight) drops 0.004-0.011 -> never fine-tune g4 on valid.
  Larger unscored weight, higher lr, more epochs all hurt; anchoring to the teacher (anc) and hedging 0.9 limit the g4 loss.
- The crossfit route rule only chose between fine-tunes (g4 = best shared FT), so sweep11_routed/_hedge.zip carry the worse g4.
  Fix for later: let each group keep the base model when no fine-tune beats it.
- Importance weighting by valid-likeness: no effect (-0.00002). Dropped.
- c1/c2: training from scratch with valid rows (30 passes over the valid half) collapses on the other half.
  **Cause: memorisation, not a bug** (local check 2026-11-06, 160 seqs per half, scored rows):
  | | half A (c1/c2 trained on it) | half B (unseen) | A g4 | B g4 |
  |---|---|---|---|---|
  | c0_tr | 0.6923 | 0.6899 | 0.448 | 0.434 |
  | c1_trv_A | **0.8656** | 0.5839 | **0.844** | 0.180 |
  | c2_v_A | 0.8699 | 0.5392 | 0.855 | 0.179 |
  Training loss of c2 halves over 10 epochs (1.17 -> 0.58). Raw-label targets on repeated sequences get memorised (group 4
  most: 0.45 -> 0.84 in sample), and that hurts unseen sequences; teacher targets on train do not. Explains why even
  3-epoch fine-tunes hurt g4. -> No rebuild on train+valid; any training on valid stays light (few passes, low lr,
  hedged) and is judged only cross-fitted. Don't submit sweep11_c3.zip.
- 10b: no group-4 model beats ft_film~wise; sweep10b_routed.zip routes exactly like 10a -> don't submit.
- **zips/sweep11_mix.zip** (built locally from Drive files): g1-3 = sweep11_hedge models (vg*_all, hedge 0.9),
  g4 = ft_film~wise and fallback s9 from sweep10a_routed. Checked on 8 valid sequences: g1-3 predictions identical to
  sweep11_hedge, g4 identical to sweep10a_routed. Expected about +0.001 cross-fitted over 10a (g1-3 +0.0018 each).
- **LB 2026-11-06: sweep11_mix.zip 0.66021 (31m06s)** = +0.00328 over sweep10a_routed 0.65693, from g1-3 only
  (cross-fit predicted ~+0.001). Light fine-tuning on valid transfers ~3x better than the cross-fit -> the test set looks
  close to the validation period. Probes: sweep11_hedge.zip (= mix but g4 = v_lr2_u06_anc_all~0.9; cross-fit says g4
  -0.0045) and zips/sweep11_mix_a1.zip (= mix with unhedged g1-3 fine-tunes).
- **LB 2026-11-06 probes:** sweep11_mix_a1.zip 0.66017 (32m57s; unhedged g1-3 = mix -0.00004, hedge irrelevant);
  sweep11_hedge.zip 0.65956 (28m06s; g4 fine-tuned on valid = mix -0.00065 -> the cross-fit was right about g4: keep
  10a's g4). Best: sweep11_mix.zip 0.66021.

## Sweep 12 (built 2026-11-08): better group fine-tunes on validation
- New code: ExpConfig `train_part` (all | head | top: freeze the rest), `valid_frac` + `bag_seed` (bagging over the
  validation sequences), `pipeline/soup.py` (average the weights of fine-tunes of the same start), crossfit export
  copies `.onnx` routes as they are (reuse sweep10a_routed's g4 / fallback models).
- Notebook colab/connectome_sweep12_rclone.ipynb (build: colab/build_sweep12.py). 12a: per group 1-3 vgN_ref (= sweep 11
  recipe), _u0, _u02, _e6, _lr1, _head, _soup (4 x 75% bags), _mix (+ 2000 train seqs, teacher targets); group 4:
  vg4_head, _top, _light, _mix. 12b cross-fit at hedge 0.9; a group switches only if it beats vgN_ref by > 0.0002
  (g4: s9 by > 0.0005). 12c winners on all valid -> zips/sweep12_routed.zip (+ _g13 / _g4 if g4 changed).
- Dry run (2 rounds, tiny models) passed; ~1-1.5 h on an A100.
- **Sweep 12 results (2026-11-08)**, cross-fitted per-group WP at hedge 0.9 (vgN_ref reproduced sweep 11 exactly):
  g1: e6 +0.0010, u0 +0.0007, u02 +0.0004, lr1 +0.0002, soup +0.0001, mix -0.0006, head -0.0018
  g2: u02 +0.0004, u0 +0.0004, mix/lr1 +0.0001, e6 -0.0003, soup -0.0004, head -0.0016
  g3: e6 +0.0008, lr1 +0.0007, soup +0.0004, u02 -0.0001, mix/u0 -0.0017, head -0.0021
  g4 vs s9 0.4329: light -0.0020, head -0.0025, top -0.0029, mix -0.0095 -> group 4 on validation is closed.
  Soups, head-only and mixed-in train rows don't help. Chosen: vg1_e6, vg2_u02, vg3_e6 -> zips/sweep12_routed.zip
  (g4 + fallback identical to sweep11_mix; checked locally on 8 seqs; speed 1.10x baseline ~ 31-32 min on the grader).
- **LB 2026-11-08: sweep12_routed.zip 0.66007 (31m01s)** = sweep11_mix -0.00014. The +0.0004 cross-fitted gain did not
  show; differences this small are leaderboard noise. Group fine-tune tuning has hit its floor. Best stays sweep11_mix.zip 0.66021.

## Sweep 13 (built 2026-11-08): per-group 2-model ensembles
- `pipeline/enscheck.py score`: per group, single members, current + one, greedy ensemble (max 2), and an honest
  version (picked on half A, scored on half B and back); members = fine-tune pairs (cross-fitted), fixed or ~wise
  models. `enscheck export`: routed submission with a per-group average (EnsembleNet, one ONNX per group).
- `pipeline/speed.py --per-group --json`: one sequence per group (routed submissions run one group's model).
  Mac: single 21 us/row (g4 film_wise 25), 2-model 38 (1.8x), 3-model 54 (2.6x) -> at most 2 models per group.
- Notebook colab/connectome_sweep13_rclone.ipynb: 13a fine-tunes t9_h128_s0..3 per group on halves (new diverse
  members); 13b enscheck (keep if honest gain > 0.0005); 13c members on all valid, export, per-group speed vs
  sweep12_routed (31 min), drops the least valuable ensembles until the estimate is <= 50 min -> zips/sweep13_ens.zip.
- Dry run (2 rounds, tiny models) passed, including the drop-to-fit path.
- **Sweep 13 results (2026-11-08)**, 2-model per-group ensembles, scored on held-out halves (gain vs sweep11_mix's model):
  g1 best pair ref+e6 pooled +0.0015 but honest +0.0002 -> no; g2 ref+u0 honest +0.0006 -> keep; g3 ref+e6 honest -0.0002
  -> no; g4 film_wise + vg4_head (s9 output layer fine-tuned on g4 valid, hedge 0.9) honest +0.0016 / +0.0022 per half
  (+0.0019; vg4_head alone -0.0030) -> keep. The fine-tuned t9 teacher members: singles -0.004 to -0.006, no pair helps.
- zips/sweep13_ens.zip: g2 and g4 ensembles, g1/g3/fallback = sweep11_mix (g1/g3 predictions identical, checked on 8
  seqs). Speed: Colab estimate 45 min, Mac 31 x 32.6 / 22.7 = 44.5 min (per group 21 / 39 / 21 / 43 us/row vs mix 21-25).
- **LB 2026-11-08: sweep13_ens.zip 0.66009 (44m17s)** = sweep11_mix -0.00012 (noise). sweep12_routed was submitted twice:
  0.66007 both times -> the LB is deterministic; the ~0.0001-0.0002 differences between our variants are differences in
  the test sample, not signal. Plateau: sweep11_mix.zip 0.66021 stays the final pick (rank 251 / 5760 on 2026-11-08).
