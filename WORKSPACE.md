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
| sweep6_sw_h128_s1.zip (distilled from mask-weighted teacher w, lambda 0.5, 10 passes; half B 0.6782) | **0.65660** | 28m11s |
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
- Best submission: sweep6_sw_h128_s1 = 0.65660 LB (2026-10-26; was sweep5_m50_e10 0.65444). Top 20 on the LB: 0.669-0.674.
- Keep a browser tab connected to Colab while a run is going (Colab Pro has no background execution).
- 2026-10-26, free T4: students group 1 done (sw_h128 half B 0.6780, sw_h128_s1 0.6782; zips in runs/zips/). Group 2
  (first feature students) ran out of host RAM: features.offline worked in float64 on the whole 256-seq batch (~14 GB
  peak). Fixed: 16 sequences at a time (identical output, ~1.6 GB), loader drops its batch reference, extras joined on
  the GPU. The T4 launcher patches the sweep-6 code with these and skips students that already have results.json.
- 2026-10-26, Sweep 7 extended with `MinGRUNet` (`arch="mingru"`, parallel log-space scan + chrono init, 1.23x baseline ONNX time)
  and causal dilated `TCNNet` (`arch="tcn"`, exact chunk-boundary state buffers, 2,048–2,187-row receptive field):
  13 models total (9 GRU training-dynamics runs + 2 minGRUs + 2 TCNs) in `colab/connectome_sweep7.ipynb`,
  `colab/connectome_sweep7_launcher.ipynb`, and `colab/connectome_sweep7_onecell.py`.
