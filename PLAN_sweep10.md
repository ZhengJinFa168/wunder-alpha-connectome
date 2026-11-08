# Sweep 10 plan: instrument groups (drafted 2026-11-04)

## Why
- Five sweeps of distillation / ensembling / seeds / features end at ~0.678 on half B (best: s9_h128 0.67824,
  sw_h128_s1 0.67816 = LB 0.65660). The top 20 on the LB are at 0.669-0.674, so we need a structural gain.
- The data has 4 hidden instruments, told apart by tick size (see "Groups" below). Group 4 (largest tick) has
  ~half of the scored rows and our model scores only ~0.39 there (groups 1-2 ~0.79, group 3 ~0.69).
- Scaling group-4 predictions up hurts (tested), so the gain has to come from better group-4 predictions,
  not from recalibration.

## Groups
Measured 2026-11-04 on the full data (labels from the targets: min |t0| of a sequence; every target is a whole
multiple of its tick, residual < 0.006).

| group | tick (target std units) | train seqs | train share | valid seqs | valid share | gru128_wmse WP (624 valid seqs) |
|---|---|---|---|---|---|---|
| 1 | 0.1254 | 1305 | 12.3% | 398 | 21.2% | 0.786 |
| 2 | 0.2413 | 1257 | 11.9% | 347 | 18.5% | 0.764 |
| 3 | 0.4685 | 3488 | 32.9% | 431 | 23.0% | 0.707 |
| 4 | 0.6569 | 4557 | 43.0% | 697 | 37.2% | **0.413** |

- What identifies a group: mostly the volume (queue size) levels, e.g. i0_v17, i0_v0, i0_v9, i0_v3: larger tick ->
  deeper queues at the best prices. Probably 4 different contracts/markets (inferred, not stated by the organisers).
- Detection from the warm-up rows (0-98, never scored): logistic regression on the mean/std of the 112 inputs
  99.6-99.7% of the 1873 valid sequences (5-7 errors); gradient boosting 99.9%; already 99.5% from rows 0-9.
- **The group mix moves the pooled score a lot.** Same model, valid re-weighted to other mixes:
  valid mix 0.671, train mix 0.621, group 4 at 50% 0.628, equal mix 0.706. Train and valid have different mixes,
  and the test mix is unknown; the LB / half-B gap (0.657 vs 0.678) fits a test mix with somewhat more of
  groups 3-4 than valid. So group-4 gains should count at least as much on the LB as on valid.
- The shared model is fitted to the train mix (43% group 4, 12% group 1), not to the valid/test mix. Per-group
  models do not depend on the mix at all.

## Ideas kept (from the outside review + ours) and dropped
Kept:
1. One model per group, routed by a detector (main bet). Each group model starts from the best shared student
   and is fine-tuned on its group only (keeps what it learned from all groups, then specializes).
2. Group 4 specialist from scratch, h128 and h192 (does group 4 gain from more capacity? does the start matter?).
3. Group-conditioned single model: the detected group modulates the inputs (FiLM: x * (1 + G_gamma g) + G_beta g)
   and picks one of 4 output heads. Zero-initialised on top of the best student, so it starts as that model.
4. Group probabilities as plain extra inputs (4 numbers; cheapest form of conditioning).
5. Fixed auxiliary targets: future t0/t1 at +10/+50/+200 rows and p(scored), from a training-only head
   (the sweep 9 ones averaged price columns with mixed signs and cancelled out).
6. Group weights in a shared model so the training mix matches the valid mix (g1 x1.7, g2 x1.6, g3 x0.7,
   g4 x0.87). (The outside review proposed up-weighting group 4, but group 4 is already over-represented in train.)
7. Differences in uniform space: Phi(x_t) - Phi(x_t-1) (inputs are quantile-transformed to a normal, so raw
   diffs blow up in the tails).
8. Bigger student, h160 (~1.5x the multiply-adds of h128; our h128 runs 28 min of the 60-min limit).

Dropped (tested or wrong premise): scaling group-4 predictions; "fixing" the norm features (they never replaced
the inputs); clipped / global-moment Pearson loss (targets are already clipped like the metric; Pearson loss lost
in sweep 1); online target reconstruction (per-sequence calibration gave +0.000); minGRU fixes and TCN/minGRU in
the teacher (in the sweep 9 pool, not picked); conv stem (more export work, small expected gain).

## Code to write (local, tested before any Colab run)
1. `pipeline/groups.py`
   - `label(Y)`: group of a sequence from its targets (training / evaluation only).
   - Detector: multinomial logistic regression (or a tiny MLP if that reaches ~100%) on the mean and std of the
     112 inputs over the warm-up rows (rows 0-98, never scored), fitted on train, constants stored in the file
     (numpy only at inference). Target: >= 99.7% on all valid; confidence < 0.99 -> fallback model.
     `Online` version for solution.py; offline == online check.
   - Feature kind `grp`: the detector's 4 probabilities from the running mean/std so far (frozen after row 98),
     causal and identical offline / online (idea 4, and the input of idea 3).
   - Feature kind `udiff`: Phi(x_t) - Phi(x_t-1), scaled to unit std (idea 7).
2. `pipeline/trainer.py`
   - `ExpConfig.group`: train only on that group's sequences (like `fold`, groups from the batch targets), and
     select checkpoints on that group's rows.
   - `ExpConfig.init`: start from a checkpoint (same shape; extra FiLM/heads parameters start at zero).
   - `ExpConfig.group_weight`: per-group row-weight multipliers (idea 6), e.g. "valid" = match the valid mix.
   - New aux targets (idea 5), replacing the old ones.
   - Every evaluation also reports scored WP per group (so every table shows where a model gains).
3. `pipeline/model.py`: `GroupGRUNet` (GRU + FiLM on the inputs + 4 heads mixed by the group probabilities),
   ONNX step export like the others.
4. `pipeline/routed.py`
   - Score a routing (one checkpoint per group + a fallback) on validation, with true and with detected groups:
     per group and pooled, half A and half B. Per group it keeps the specialist only if it beats the shared model
     on half A (half B stays an honest check).
   - Export a routed submission: `model_g1..g4.onnx` (+ fallback), `wnn_groups.py`, routed `solution.py`:
     all models run on the warm-up rows (+~2% time), at the first scored-row-capable step the detector picks one
     and only that one runs afterwards; low detector confidence -> the shared fallback model.
5. `test_export.py`: routed submission row by row against PyTorch (all 4 routes + fallback), FiLM model, new
   feature kinds. `speed.py` on the routed submission.
6. Notebooks `colab/build_sweep10a.py` / `10b`, dry-run locally on tiny data (both rounds: fresh and re-run).

## Sweep 10a (A100, ~3.5 h): does specializing by group help?
All start from runs/sweep9/s9_h128/best.pt and distil from teacher 9 (runs/cache/t_9.npy), lr 5e-4, 6 passes,
unless noted.
| model | trains on | question |
|---|---|---|
| ft_all | all groups | control: is any gain just from training longer? |
| ft_g1 .. ft_g4 | one group each | does a specialist beat the shared model on its group? |
| ft_film | all groups, FiLM + 4 heads | can one conditioned model do the same? |
| g4_h128 | group 4, from scratch, 10 passes | does the warm start matter? |
| g4_h192 | group 4, from scratch, 10 passes | does group 4 gain from capacity? (2x cost on 38% of sequences) |
Then: `routed.py` table (true / detected groups, half A / half B), routed export + row-by-row check + speed test.

Decision:
- Routed (detected groups) pooled half B >= 0.6792 (+0.001 over s9_h128): submit the routed zip.
- Specialists help only on group 4: route group 4 only, shared model elsewhere.
- Nothing helps: group 4 is mostly noise for these inputs; go to 10b without routing.

## Sweep 10b (A100, ~4-5 h): shared-model ideas, on top of 10a's answer
Students, 10 passes, distilled from teacher 9, compared with s9_h128 (0.67824; student seed spread ~0.0002):
s10_aux2 (idea 5), s10_udiff (7), s10_h160 (8), s10_grp (4), s10_mixw (6, only if routing did not win).
If routing won, the winners of 10b are then fine-tuned per group the same way (short 10a-style run).

## Timeline (deadline Nov 15)
- Nov 4-5: code + local tests + dry runs; send 10a.
- Nov 5-6: run 10a, analyse, submit if it wins.
- Nov 7-8: run 10b, analyse, submit the best.
- Nov 9-12: final recipe (incl. the pending train + validation model), final checks.
- Nov 13-15: final submissions, buffer.

## Risks
- Group 4 may just be noisy: 10a measures it directly (specialist vs shared on group-4 rows).
- Detector mistakes on the test set: fallback model when unsure; detector accuracy reported on all of valid.
- Small per-group selection sets: routing choices on half A, the honest number on half B.
- Colab compute units: 10a and 10b are separate runs (~3.5 h and ~4-5 h); both re-runnable (finished work skipped).
