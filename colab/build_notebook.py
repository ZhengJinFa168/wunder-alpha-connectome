"""Build the Colab notebooks with the current pipeline/ code embedded (%%writefile cells), so they
need no GitHub or Drive copy of the code. Re-run after changing pipeline/.

    python colab/build_notebook.py      # writes colab/connectome_multi.ipynb and colab/connectome_sweep2.ipynb
"""
import json
import os
import pprint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["__init__.py", "data.py", "model.py", "fast_eval.py", "trainer.py", "multi_train.py", "train.py",
         "ensemble.py", "speed.py", "teacher.py", "test_export.py", "mask_model.py", "features.py",
         "solution_template.py"]
ANALYSIS = ["__init__.py", "error_slices.py"]

SWEEP1 = [
    {"name": "e01_wmse_h128"},
    {"name": "e02_mix_h128", "loss": "mix", "alpha": 0.1},
    {"name": "e03_pearson_h128", "loss": "pearson"},
    {"name": "e04_wmse_diff_h128", "diff": True},
    {"name": "e05_wmse_h192_drop", "hidden": 192, "dropout": 0.1},
    {"name": "e06_wmse_h128_lr1e-3", "lr": 1e-3},
]

SWEEP2 = [
    # noise check: same setup, three init seeds
    {"name": "s0_wmse_h128", "seed": 0},
    {"name": "s1_wmse_h128", "seed": 1},
    {"name": "s2_wmse_h128", "seed": 2},
    # sweep-1 winners combined (mix loss + diff inputs), at three widths
    {"name": "c1_mix_diff_h128", "loss": "mix", "diff": True},
    {"name": "c2_mix_diff_h128_drop", "loss": "mix", "diff": True, "dropout": 0.1},
    {"name": "c3_mix_diff_h192_drop", "loss": "mix", "diff": True, "hidden": 192, "dropout": 0.1},
    {"name": "c3b_mix_diff_h192_drop_s1", "loss": "mix", "diff": True, "hidden": 192, "dropout": 0.1, "seed": 1},
    {"name": "c4_mix_diff_h256_drop", "loss": "mix", "diff": True, "hidden": 256, "dropout": 0.1},
]

# every model here costs <= ~1x the baseline at inference (cost = multiply-adds per row / baseline's)
SWEEP3 = [
    {"name": "r0_wmse_h128", "seed": 3},                                    # reference for this sweep
    # distillation from the sweep-2 4-model ensemble
    {"name": "d50_h128", "distill": 0.5},
    {"name": "d50_h128_s1", "distill": 0.5, "seed": 1},
    {"name": "d100_h128", "distill": 1.0},
    {"name": "d50_ema_h128", "distill": 0.5, "ema": 0.999},
    # free tricks (no inference cost)
    {"name": "ema_h128", "ema": 0.999},
    {"name": "chunk2000_h128", "chunk": 2000},
    {"name": "lr4e-3_h128", "lr": 4e-3},
    # other shapes at about the same cost
    {"name": "l3_h96", "layers": 3, "hidden": 96},                         # cost 0.89
    {"name": "l1_h192", "layers": 1, "hidden": 192},                       # cost 0.92
    {"name": "lstm_h112", "cell": "lstm", "hidden": 112},                  # cost 1.05
    # small models for cheap ensembles
    {"name": "h96", "hidden": 96},                                         # cost 0.60
    {"name": "h96_s1", "hidden": 96, "seed": 1},
    {"name": "d50_h96", "hidden": 96, "distill": 0.5},
    {"name": "h64", "hidden": 64},                                         # cost 0.31
    {"name": "h64_s1", "hidden": 64, "seed": 1},
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def setup_cells(title, intro):
    cells = [
        md(f"# {title}\n\n{intro.strip()}\n\n"
           "**Runtime:** Runtime > Change runtime type > **A100 or L4 GPU** and **High-RAM**.\n"
           "Results go to Google Drive (`USE_DRIVE = True`), so they survive a disconnect.\n\n"
           "Code below is generated from `pipeline/` by `colab/build_notebook.py`; edit there, not here."),
        code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
!echo "CPUs: $(nproc)"; lscpu | grep "Model name"; free -g | head -2; df -h /content | tail -1
"""),
        code("""
USE_DRIVE = True
OUT_DIR = '/content/drive/MyDrive/wunder-connectome/runs' if USE_DRIVE else '/content/runs'
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
import os; os.makedirs(OUT_DIR, exist_ok=True); print('results ->', OUT_DIR)
"""),
        md("## 1. Data (33.7 GB; ~10-20 min)\n"
           "Downloads to a resumable file, unpacks, then checks every parquet file opens with the right number of "
           "sequences. Skipped when the data is already there and valid; re-downloaded when anything is broken."),
        code("""
%%bash
mkdir -p /content/wnn && cd /content/wnn
check() {
python3 - <<'EOF'
import sys, pyarrow.parquet as pq
ok = True
for f, n in {'train': 10607, 'valid': 1873, 'valid_mask': 1873}.items():
    try:
        g = pq.ParquetFile(f'datasets/{f}.parquet').metadata.num_row_groups
        print(f'{f}: {g} sequences', 'OK' if g == n else f'(expected {n})'); ok &= g == n
    except Exception as e:
        print(f'{f}: BROKEN ({type(e).__name__}: {e})'); ok = False
sys.exit(0 if ok else 1)
EOF
}
if check; then echo 'data OK, nothing to do'; else
  rm -rf datasets
  URL=https://files.wundernn.io/wnn_connectome_starterpack.tar.gz
  for i in $(seq 10); do   # -C - resumes where a dropped connection stopped
    curl -L -C - --retry 20 --retry-all-errors -sS -o pack.tar.gz "$URL" && break
    echo "download interrupted, resuming ($i)"; sleep 5
  done
  ls -la pack.tar.gz
  tar -xzf pack.tar.gz --strip-components=1 --warning=no-unknown-keyword && find . -name '._*' -delete
  if check; then rm pack.tar.gz; echo 'data OK'; else echo 'DATA STILL BROKEN: delete /content/wnn/pack.tar.gz and re-run this cell'; exit 1; fi
fi
du -sh datasets
"""),
        code("""
!pip -q install onnx onnxruntime
%cd /content/wnn
"""),
        md("## 2. Pipeline code"),
        code("!mkdir -p /content/wnn/pipeline"),
    ]
    for f in FILES:
        src = open(os.path.join(ROOT, "pipeline", f)).read()
        cells.append(code(f"%%writefile /content/wnn/pipeline/{f}\n" + (src if src.strip() else "\n")))
    cells.append(code("!mkdir -p /content/wnn/analysis"))
    for f in ANALYSIS:
        src = open(os.path.join(ROOT, "analysis", f)).read()
        cells.append(code(f"%%writefile /content/wnn/analysis/{f}\n" + (src if src.strip() else "\n")))
    cells += [
        md("## 3. Sanity check: baseline weights through our fast evaluator (300 seqs: 0.6246; full valid: 0.617052)"),
        code("!python -m pipeline.fast_eval --baseline --valid-seqs 300"),
    ]
    return cells


def experiments_cell(exps):
    return code("import json\nexperiments = " + pprint.pformat(exps, sort_dicts=False, width=110) +
                "\njson.dump(experiments, open('experiments.json', 'w'), indent=1)\nprint(len(experiments), 'experiments')")


def results_cell(out):
    return code(f"""
import pandas as pd
rows = [json.loads(l) for l in open(f'{out}/summary.jsonl')]
df = pd.DataFrame(rows).drop_duplicates('name', keep='last')
df['select'] = (df.valid_scored + df.holdout_all) / 2
df.sort_values('select', ascending=False)
""")


def write(nb_cells, filename):
    nb = {"cells": nb_cells,
          "metadata": {"accelerator": "GPU", "colab": {"machine_shape": "hm", "gpuType": "A100"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = os.path.join(ROOT, "colab", filename)
    json.dump(nb, open(out, "w"), indent=1)
    print("wrote", out, f"({len(nb_cells)} cells)")


def sweep1():
    cells = setup_cells("Alpha Connectome: train many GRUs from one data stream",
                        "Downloads the starter pack once, then trains every experiment in `experiments.json` side by "
                        "side: each batch of sequences is read once and used to train all models; evaluation is shared too.")
    cells += [
        md("## 4. Experiments\nPer-model fields: `name, hidden, layers, diff, dropout, init_baseline, loss "
           "(wmse|pearson|mix), alpha, lr, wd, seed`. Shared settings are flags on the run command below."),
        experiments_cell(SWEEP1),
        md("## 5. Train"),
        code("!python -m pipeline.multi_train --config experiments.json --epochs 4 --threads 8 --cache-eval "
             "--out-dir {OUT_DIR} 2>&1 | grep -v '\"event\": \"batch\"'"),
        md("## 6. Results"),
        results_cell("{OUT_DIR}"),
    ]
    write(cells, "connectome_multi.ipynb")


def sweep2():
    cells = setup_cells("Alpha Connectome, sweep 2: noise check, combined winners, 8 passes, ensembles", """
What this run answers (about 75-90 min on an A100, after the data download):
1. **Noise:** the same model with 3 seeds shows how much scores move by chance.
2. **Combined winners:** sweep 1's best changes together (mix loss + diff inputs), at widths 128 / 192 / 256.
3. **Longer training:** 8 passes instead of 4 (compare with sweep 1).
4. **Ensembles:** average the best models (chosen on the train holdout) and export the result as one ONNX.
5. **Speed:** time each candidate row by row on this machine's CPU, relative to the official baseline.

Keep the shared `seed` at 0 (default): it fixes the 500-sequence train holdout, so all sweeps are comparable.
""")
    cells += [
        md("## 4. Experiments"),
        experiments_cell(SWEEP2),
        md("## 5. Train (8 passes, one evaluation per pass)"),
        code("SWEEP = f'{OUT_DIR}/sweep2'\n"
             "!python -m pipeline.multi_train --config experiments.json --epochs 8 --evals-per-epoch 1 "
             "--threads 8 --cache-eval --out-dir {SWEEP} 2>&1 | grep -v '\"event\": \"batch\"'"),
        md("## 6. Results"),
        results_cell("{SWEEP}"),
        md("""
## 7. Ensembles
Uses the sweep-2 models plus sweep 1's useful ones (pure-Pearson e03 left out). The official baseline is left out
because it was probably trained on our holdout sequences, so its holdout score is not an honest one.
`MAX_MEMBERS` bounds inference cost: each 128-unit model costs ~1x baseline time, 192-unit ~2x, 256-unit ~3.5x.
"""),
        code("MAX_MEMBERS = 4\n"
             "!python -m pipeline.ensemble --ckpts \"{SWEEP}/*/best.pt\" \"{OUT_DIR}/e0[12456]*/best.pt\" "
             "--threads 8 --max-members {MAX_MEMBERS} --export {OUT_DIR}/ensemble_sweep2/submission 2>&1 "
             "| grep -vE 'Warning|_generic_rnn'"),
        md("""
## 8. Speed on this CPU (one thread), relative to the official baseline
The grader allows ~91 µs/row; the baseline fits comfortably, so a ratio up to ~2x is very likely safe.
Re-check finalists on the Mac with `evaluate.py` too.
"""),
        code("!python -m pipeline.speed {SWEEP}/s0_wmse_h128/submission {SWEEP}/c1_mix_diff_h128/submission "
             "{SWEEP}/c3_mix_diff_h192_drop/submission {SWEEP}/c4_mix_diff_h256_drop/submission "
             "{OUT_DIR}/ensemble_sweep2/submission"),
        md("Submission folders (solution.py + model.onnx) are in Drive under `wunder-connectome/runs/`: "
           "`sweep2/<name>/submission/` and `ensemble_sweep2/submission/`. Zip one with `solution.py` at the root to upload."),
    ]
    write(cells, "connectome_sweep2.ipynb")


# Phase 1: out-of-fold teacher models (2 configs x 4 folds; each trains on 3/4 of the training sequences)
SWEEP4_TEACHERS = [{"name": f"f{k}_wmse_h128", "fold": k} for k in range(4)] + \
                  [{"name": f"f{k}_mixdiff_h192", "fold": k, "loss": "mix", "diff": True, "hidden": 192, "dropout": 0.1}
                   for k in range(4)]


def _student(name, teacher, **kw):
    return {"name": name, "distill": 1.0, "teacher": f"/content/wnn/t_{teacher}.npy", **kw}


# Phase 3: students (all distill=1.0 unless noted). Teachers: old = sweep-2 ensemble, big = best <=8 of all
# non-distilled models, in2 = 2 full-data models, oof = the same 2 configs out-of-fold.
SWEEP4_STUDENTS = [
    _student("st_old", "old"),
    _student("st_old_s1", "old", seed=1),
    _student("st_big", "big"),
    _student("st_big_s1", "big", seed=1),
    _student("st_in2", "in2"),
    _student("st_oof", "oof"),
    _student("st_big_e10", "big", epochs=10),
    _student("st_big_h96", "big", hidden=96),
    _student("st_big_h160", "big", hidden=160),
    _student("st_big_h192", "big", hidden=192),
    _student("st_big_l3h96", "big", layers=3, hidden=96),
    _student("st_big_wnone", "big", distill_weight="none"),
    _student("st_big_wteacher", "big", distill_weight="teacher"),
    _student("st_big_lr1e-3", "big", lr=1e-3),
    _student("st_big_drop", "big", dropout=0.1),
    _student("st_big_d90", "big", distill=0.9),
]


def _mstudent(name, lam=0.0, **kw):
    return {"name": name, "distill": 1.0, "teacher": "/content/wnn/t_big.npy", "mask_weight": lam, **kw}


# Sweep 5: students of teacher `big`, rows weighted (1 - lam) + lam * p(scored) / mean(p)
SWEEP5 = [
    _mstudent("m00"),                                   # lam 0: plain distillation (reference)
    _mstudent("m00_s1", seed=1),
    _mstudent("m25", 0.25),
    _mstudent("m50", 0.5),
    _mstudent("m50_s1", 0.5, seed=1),
    _mstudent("m75", 0.75),
    _mstudent("m75_s1", 0.75, seed=1),
    _mstudent("m100", 1.0),
    _mstudent("m100_pow2", 1.0, mask_power=2.0),          # sharper focus: weight ~ p^2
    _mstudent("m50_e10", 0.5, epochs=10),
    _mstudent("m75_e10", 0.75, epochs=10),
    _mstudent("m50_h96", 0.5, hidden=96),                 # 0.8x baseline time
    {"name": "plain_m00"},                                # no teacher, no mask weights
    {"name": "plain_m50", "mask_weight": 0.5},            # no teacher, mask weights only
]


# Sweep 6, phase 1: mask-weighted teacher models (lambda 0.5, no teacher of their own), with/without extra features
SWEEP6_TEACHERS = [
    {"name": "tw_h128_s0", "mask_weight": 0.5},
    {"name": "tw_h128_s1", "mask_weight": 0.5, "seed": 1},
    {"name": "tw_h128_m75", "mask_weight": 0.75},
    {"name": "tw_h192_drop", "mask_weight": 0.5, "hidden": 192, "dropout": 0.1},
    {"name": "tw_mixdiff_h128", "mask_weight": 0.5, "loss": "mix", "diff": True},
    {"name": "tw_feat_h128_s0", "mask_weight": 0.5, "feats": True},
    {"name": "tw_feat_h128_s1", "mask_weight": 0.5, "feats": True, "seed": 1},
    {"name": "tw_feat_h192_drop", "mask_weight": 0.5, "feats": True, "hidden": 192, "dropout": 0.1},
]


def _wstudent(name, teacher="w", lam=0.5, **kw):
    return {"name": name, "distill": 1.0, "teacher": f"/content/wnn/t_{teacher}.npy", "mask_weight": lam, **kw}


# Sweep 6, phase 3: students (10 passes) of the mask-weighted teacher `w`
SWEEP6_STUDENTS = [
    _wstudent("sw_h128"),
    _wstudent("sw_h128_s1", seed=1),
    _wstudent("sw_feat_h128", feats=True),
    _wstudent("sw_feat_h128_s1", feats=True, seed=1),
    _wstudent("sw_feat_h128_m75", lam=0.75, feats=True),
    _wstudent("sw_feat_h96", feats=True, hidden=96),
    _wstudent("sb_feat_h128", teacher="big", feats=True),   # old teacher + features: compare with sweep 5's m50_e10
]


# Sweep 7: training dynamics. Plain mask-weighted GRUs (lambda 0.5, no teacher), 4 passes; the data batch stays 256
# sequences, `micro_batch` splits it into smaller optimizer steps, `chunk` is how far back gradients reach.
def _dyn(name, **kw):
    return {"name": name, "mask_weight": 0.5, **kw}


SWEEP7 = [
    _dyn("b256_c500"),                                        # today's recipe (reference)
    _dyn("b64_c500", micro_batch=64),
    _dyn("b32_c500", micro_batch=32),
    _dyn("b256_c2000", chunk=2000),
    _dyn("b64_c2000", micro_batch=64, chunk=2000),
    _dyn("b32_c2000", micro_batch=32, chunk=2000),
    _dyn("b32_c5000", micro_batch=32, chunk=5000),
    _dyn("b32_c20000", micro_batch=32, chunk=20000),          # full-sequence backprop
    _dyn("b32_c500_lr1e-3", micro_batch=32, lr=1e-3),
    # New architectures, same recipe (mask-weighted, no teacher, 4 passes): candidates for a more diverse teacher.
    # Each pairs with a GRU above that has the same step size and backprop length. Notes:
    # research/architectures_cnn_rnn_transformer.md
    _dyn("mingru_h192_l3", arch="mingru", hidden=192, layers=3),             # cost 1.00, ONNX 1.23x baseline time
    _dyn("mingru_h192_l3_b32_c5000", arch="mingru", hidden=192, layers=3,    # pairs with b32_c5000: the parallel
         micro_batch=32, chunk=5000),                                         # scan makes long backprop cheap
    _dyn("tcn_c64_l11", arch="tcn", hidden=64, layers=11, kernel=2, chunk=2000),      # sees 2048 rows, cost 0.75
    _dyn("tcn_c128_k3_l7", arch="tcn", hidden=128, layers=7, kernel=3, chunk=2000),   # sees 2187 rows, teacher only
]
SWEEP7_NEW_ARCH = [e["name"] for e in SWEEP7 if "arch" in e]


def sweep3():
    cells = setup_cells("Alpha Connectome, sweep 3: distillation, free tricks, LSTM, cheap ensembles", """
The grader timed our single 128-unit GRU at 37 min (baseline 27 min, limit 60), so anything we submit must stay
close to single-model cost. Every model here costs about 1x the baseline or less. About 70-90 min on an A100.

1. **Distillation:** students learn from the sweep-2 4-model ensemble (0.6826 valid) at single-model cost.
2. **Free tricks:** EMA of the weights, 2000-step training chunks, a higher learning rate.
3. **Same cost, other shapes:** 3 layers x 96, 1 layer x 192, LSTM x 112 (GRU vs LSTM at equal speed).
4. **Small models** (96 / 64 units, 2 seeds) for ensembles that fit the time budget.
5. **Ensembles under cost budgets** 1 / 1.5 / 2 / 2.5, each exported, then everything **timed on this CPU**.

Seed noise from sweep 2: about 0.0007 on validation (scored) and 0.005 on the holdout, so judge by validation.
Needs sweep 2's results in Drive (`runs/sweep2/` and `runs/ensemble_sweep2/`).
""")
    cells += [
        md("## 3b. Export check: ONNX + solution.py reproduce PyTorch for GRU, diff-GRU, LSTM, ensemble"),
        code("!python -m pipeline.test_export 2>&1 | grep -E 'OK|MISMATCH'"),
        md("## 4. Teacher: the sweep-2 ensemble's predictions on all training sequences (~850 MB, a few minutes)"),
        code("""
TEACHER = '/content/wnn/teacher_train.npy'
if not os.path.exists(TEACHER):
    !python -m pipeline.teacher --ensemble {OUT_DIR}/ensemble_sweep2/submission/ensemble.json --out {TEACHER} --threads 8
else:
    print('teacher already there')
"""),
        md("## 5. Experiments\nPer-model fields: `name, hidden, layers, cell (gru|lstm), diff, dropout, loss, alpha, lr, wd, "
           "seed, chunk, ema, distill`."),
        experiments_cell(SWEEP3),
        md("## 6. Train (5 passes, one evaluation per pass)"),
        code("SWEEP = f'{OUT_DIR}/sweep3'\n"
             "!python -m pipeline.multi_train --config experiments.json --epochs 5 --evals-per-epoch 1 "
             "--threads 8 --cache-eval --teacher {TEACHER} --out-dir {SWEEP} 2>&1 | grep -v '\"event\": \"batch\"'"),
        md("## 7. Results (`cost` = multiply-adds per row vs the baseline)"),
        results_cell("{SWEEP}"),
        md("""
## 8. Ensembles under cost budgets
All sweeps' models are candidates (pure-Pearson e03 left out). The cost is compute only; the fixed per-row overhead
is paid once per ensemble, so real time grows more slowly than cost. Section 9 measures it.
"""),
        code("ENS = f'{OUT_DIR}/ensembles_sweep3'\n"
             "!python -m pipeline.ensemble --ckpts \"{SWEEP}/*/best.pt\" \"{OUT_DIR}/sweep2/*/best.pt\" "
             "\"{OUT_DIR}/e0[12456]*/best.pt\" --threads 8 --max-members 5 --budgets 1 1.5 2 2.5 "
             "--export {ENS} 2>&1 | grep -vE 'Warning|_generic_rnn|warnings.warn'"),
        md("""
## 9. Speed on this CPU (one thread), relative to the official baseline
Measured so far: Colab baseline 80 us/row; grader baseline ~41 us/row (27 min) and our 128-GRU ~57 us/row (37 min).
`s0_old_export` vs `s0_new_export` shows what the new graph (one state input per layer, no Slice/Concat) saves.
Rough rule: keep the ratio here at or below ~1.5x the baseline.
"""),
        code("""
from pathlib import Path
import shutil
s0new = '/content/s0_new_export'
os.makedirs(s0new, exist_ok=True)
!python -c "from pipeline.fast_eval import load_ckpt; from pipeline.model import export_onnx; export_onnx(load_ckpt('{OUT_DIR}/sweep2/s0_wmse_h128/best.pt'), '{s0new}/model.onnx')" 2>/dev/null
shutil.copy('pipeline/solution_template.py', f'{s0new}/solution.py')
cands = [f'{OUT_DIR}/sweep2/s0_wmse_h128/submission', s0new] + \
        [f'{SWEEP}/{n}/submission' for n in ['d50_h128', 'lstm_h112', 'l3_h96', 'h96', 'h64']] + \
        sorted(str(p) for p in Path(ENS).glob('cost*/submission'))
!python -m pipeline.speed {' '.join(cands)}
"""),
        md("Submission folders (solution.py + model.onnx) are in Drive under `wunder-connectome/runs/`: "
           "`sweep3/<name>/submission/` and `ensembles_sweep3/cost<budget>/submission/`. "
           "Zip one with `solution.py` at the root to upload."),
    ]
    write(cells, "connectome_sweep3.ipynb")


def sweep4():
    cells = setup_cells("Alpha Connectome, sweep 4: better teachers, out-of-fold teachers, student recipes", """
Sweep 3 showed distillation works: a 128-unit student (0.6821 valid) nearly matched its 4-model teacher (0.6826)
at single-model speed. This run looks for a better teacher and a better student recipe. About 2-2.5 h on an A100.

* **Phase 1:** 8 out-of-fold teacher models (2 configs x 4 folds, each trained on 3/4 of the data).
* **Phase 2:** four teachers: `old` (sweep-2 ensemble, reference), `big` (best <= 8-model ensemble of all
  non-distilled models from sweeps 1-3), `in2` (2 full-data models, in-sample), `oof` (same 2 configs, out-of-fold),
  plus a quick quality check of each teacher on training vs held-out sequences.
* **Phase 3:** 16 students: one per teacher (+ seed repeats), then recipe variations on `big`: 10 passes, width
  96/160/192, 3 layers, teacher-loss weighting, lower learning rate, dropout, 10% real-target loss.
* **Phase 4:** ensembles under cost budgets, speed on this CPU, upload-ready zips in Drive.

Needs sweeps 1-3 in Drive. Seed noise: ~0.0007 on validation (scored), ~0.005 on the holdout.
""")
    cells += [
        md("## 4. Phase 1: out-of-fold teacher models"),
        experiments_cell(SWEEP4_TEACHERS),
        code("TEACH = f'{OUT_DIR}/sweep4_teachers'\n"
             "!python -m pipeline.multi_train --config experiments.json --epochs 5 --evals-per-epoch 1 "
             "--threads 8 --cache-eval --out-dir {TEACH} 2>&1 | grep -v '\"event\": \"batch\"'"),
        md("## 5. Phase 2: teachers"),
        code("""
# old: the sweep-2 ensemble (sweep 3's teacher)
!python -m pipeline.teacher --ensemble {OUT_DIR}/ensemble_sweep2/submission/ensemble.json --out /content/wnn/t_old.npy
# in2 / oof: the same two configs, trained on all data (sweep 2) vs out-of-fold (phase 1)
!python -m pipeline.teacher --ckpts {OUT_DIR}/sweep2/s1_wmse_h128/best.pt {OUT_DIR}/sweep2/c3_mix_diff_h192_drop/best.pt --out /content/wnn/t_in2.npy
!python -m pipeline.teacher --ckpts "{TEACH}/f*/best.pt" --oof --out /content/wnn/t_oof.npy
"""),
        code("""
# big: greedy ensemble (no cost limit, up to 8 members) over every non-distilled model so far
!python -m pipeline.ensemble --ckpts "{OUT_DIR}/sweep3/[!d]*/best.pt" "{OUT_DIR}/sweep2/*/best.pt" "{OUT_DIR}/e0[12456]*/best.pt" --threads 8 --max-members 8 --budgets 99 --export {OUT_DIR}/teacher_big 2>&1 | grep -vE 'Warning|_generic_rnn|warnings.warn'
!python -m pipeline.teacher --ensemble {OUT_DIR}/teacher_big/cost99/submission/ensemble.json --out /content/wnn/t_big.npy
"""),
        md("Teacher quality: weighted Pearson of each teacher vs the true targets on training sequences "
           "(in-sample for old/big/in2) and on the 500 held-out sequences (honest for all). A big gap = in-sample optimism."),
        code("""
import numpy as np
from utils import weighted_pearson
from pipeline.data import SeqStore, split_train
st = SeqStore('datasets/train.parquet', 8)
train_ids, hold_ids = split_train(st.n_seqs, 500)
sample = {'train (300 seqs)': np.sort(np.random.default_rng(0).choice(train_ids, 300, replace=False)), 'holdout (500)': hold_ids}
Ys = {k: st.load(list(v))[1][:, 99:] for k, v in sample.items()}
for t in ['old', 'big', 'in2', 'oof']:
    T = np.load(f'/content/wnn/t_{t}.npy', mmap_mode='r')
    row = []
    for k, ids in sample.items():
        P = np.asarray(T[ids], dtype=np.float32)[:, 99:]
        row.append(f"{k}: {np.mean([weighted_pearson(Ys[k][..., j].ravel(), P[..., j].ravel()) for j in range(2)]):.4f}")
    print(f'{t:4s}', '   '.join(row))
"""),
        md("## 6. Phase 3: students (5 passes; `st_big_e10` runs 10)"),
        experiments_cell(SWEEP4_STUDENTS),
        code("SWEEP = f'{OUT_DIR}/sweep4'\n"
             "!python -m pipeline.multi_train --config experiments.json --epochs 5 --evals-per-epoch 1 "
             "--threads 8 --cache-eval --out-dir {SWEEP} 2>&1 | grep -v '\"event\": \"batch\"'"),
        md("## 7. Results (`cost` = multiply-adds per row vs the baseline)\n"
           "Sweep 3 reference: d100_h128 (teacher `old`) 0.6821 valid / 0.5136 valid_all / 0.4569 holdout."),
        results_cell("{SWEEP}"),
        md("## 8. Ensembles under cost budgets (students + everything else)"),
        code("ENS = f'{OUT_DIR}/ensembles_sweep4'\n"
             "!python -m pipeline.ensemble --ckpts \"{SWEEP}/*/best.pt\" \"{OUT_DIR}/sweep3/*/best.pt\" "
             "\"{OUT_DIR}/sweep2/*/best.pt\" --threads 8 --max-members 4 --budgets 1 1.5 2 "
             "--export {ENS} 2>&1 | grep -vE 'Warning|_generic_rnn|warnings.warn'"),
        md("## 9. Speed on this CPU vs the official baseline\n"
           "Grader ≈ 0.52 × these times (baseline: 80 µs here, 41 µs there). Our v2 export took 28.5 min on the grader. "
           "Keep candidates at or below ~1.5x here."),
        code("""
from pathlib import Path
cands = [f'{SWEEP}/{n}/submission' for n in ['st_big', 'st_big_h96', 'st_big_h160', 'st_big_h192', 'st_big_l3h96']] + \
        sorted(str(p) for p in Path(ENS).glob('cost*/submission'))
!python -m pipeline.speed {' '.join(cands)}
"""),
        md("## 10. Upload-ready zips (solution.py at the zip root) in Drive `wunder-connectome/runs/zips/`"),
        code("""
import shutil, pandas as pd
os.makedirs(f'{OUT_DIR}/zips', exist_ok=True)
top = df.sort_values('select', ascending=False).name.head(3).tolist()   # df from section 7
srcs = {f'sweep4_{n}': f'{SWEEP}/{n}/submission' for n in top}
srcs.update({f'ens4_{p.parent.name}': str(p) for p in Path(ENS).glob('cost*/submission')})
for name, src in srcs.items():
    print(shutil.make_archive(f'{OUT_DIR}/zips/{name}', 'zip', src))
"""),
    ]
    write(cells, "connectome_sweep4.ipynb")


def sweep5():
    cells = setup_cells("Alpha Connectome, sweep 5: mask-weighted distillation", """
The metric only counts rows in a hidden scoring mask (~9% of rows), and a classifier can predict those rows from
the inputs (AUC ~0.88). This sweep trains students that put more weight on likely-scored rows. About 2 h on an A100.

* **Phase 1 (~10 min):** fit the mask classifier on validation half A; report its AUC on half B.
* **Phase 2 (~20 min):** p(scored) for every training row (saved to Drive, reused next time); teacher `big`.
* **Phase 3 (~70 min):** 14 students: lambda = 0 / 0.25 / 0.5 / 0.75 / 1, p^2 weighting, seed repeats, 10 passes,
  96 units, and two no-teacher models.
* **Judge on validation half B only** (the classifier never saw its masks); st_big_e10 (LB 0.65396) is re-scored on B too.

Row weight = (1 - lambda) + lambda x p / mean(p): lambda = 0 is today's training, lambda = 1 is fully mask-focused.
Seed noise on a half (936 sequences) is a bit larger than on full validation: about +-0.001.
""")
    cells += [
        md("## 3b. Export check"),
        code("!python -m pipeline.test_export 2>&1 | grep -E 'OK|MISMATCH'"),
        md("## 4. Phase 1: mask classifier (validation half A -> AUC on half B)"),
        code("""
MASK = f'{OUT_DIR}/mask_model'
if not os.path.exists(f'{MASK}/model.joblib'):
    !python -m pipeline.mask_model fit --out {MASK} --threads 8
else:
    print('mask model already in Drive:', open(f'{MASK}/report.json').read())
"""),
        md("## 5. Phase 2: p(scored) for every training row, and teacher `big` (both cached in Drive)"),
        code("""
CACHE = f'{OUT_DIR}/cache'; os.makedirs(CACHE, exist_ok=True)
ROWW, TEACHER = '/content/wnn/mask_train.npy', '/content/wnn/t_big.npy'
if os.path.exists(f'{CACHE}/mask_train.npy'):
    !cp {CACHE}/mask_train.npy {ROWW}
else:
    !python -m pipeline.mask_model score --model {MASK} --out {ROWW} --threads 12 && cp {ROWW} {CACHE}/
if os.path.exists(f'{CACHE}/t_big.npy'):
    !cp {CACHE}/t_big.npy {TEACHER}
else:
    !python -m pipeline.teacher --ensemble {OUT_DIR}/teacher_big/cost99/submission/ensemble.json --out {TEACHER} && cp {TEACHER} {CACHE}/
!ls -la /content/wnn/*.npy
"""),
        md("## 6. Phase 3: students (5 passes; `_e10` models run 10)"),
        experiments_cell(SWEEP5),
        code("SWEEP = f'{OUT_DIR}/sweep5'\n"
             "!python -m pipeline.multi_train --config experiments.json --epochs 5 --evals-per-epoch 1 --threads 8 "
             "--cache-eval --teacher {TEACHER} --row-weights {ROWW} --valid-split {MASK}/split.json "
             "--out-dir {SWEEP} 2>&1 | grep -v '\"event\": \"batch\"'"),
        md("## 7. Results on validation half B (`valid_B_scored` is the number that matters)\n"
           "The reference st_big_e10 (sweep 4, leaderboard 0.65396) is scored on half B in the next cell."),
        code("""
!python -m pipeline.fast_eval --ckpt {OUT_DIR}/sweep4/st_big_e10/best.pt --valid-split {MASK}/split.json 2>/dev/null | head -8
import pandas as pd
rows = [json.loads(l) for l in open(f'{SWEEP}/summary.jsonl')]
df = pd.DataFrame(rows).drop_duplicates('name', keep='last')
df[['name', 'cost', 'valid_B_scored', 'valid_B_all', 'valid_scored', 'valid_all', 'holdout_all']].sort_values('valid_B_scored', ascending=False)
"""),
        md("## 8. Speed of the top candidates on this CPU, and upload-ready zips in Drive `runs/zips/`"),
        code("""
import shutil
top = df.sort_values('valid_B_scored', ascending=False).name.head(3).tolist()
speed_args = ' '.join(f'{SWEEP}/{n}/submission' for n in top)
!python -m pipeline.speed {speed_args}
os.makedirs(f'{OUT_DIR}/zips', exist_ok=True)
for n in top:
    print(shutil.make_archive(f'{OUT_DIR}/zips/sweep5_{n}', 'zip', f'{SWEEP}/{n}/submission'))
"""),
    ]
    write(cells, "connectome_sweep5.ipynb")


def sweep6():
    cells = setup_cells("Alpha Connectome, sweep 6: mask-weighted teacher + cross-instrument features", """
Sweep 5: mask weighting adds +0.005 without a teacher but only +0.001 on top of distillation, because the
teacher itself was trained without it. This sweep builds a **mask-weighted teacher** and adds **extra causal
features** (pipeline/features.py): activity of each instrument's prices / volumes / trades and EMAs of a2-a5
at 20 / 200 / 2000 steps (high a3 is where the model is weakest). About 2-2.5 h on an A100.

* **Phase 1 (~45 min):** 8 teacher models trained with lambda = 0.5, half of them with the extra features.
* **Phase 2 (~20 min):** teacher `w` = best <= 8-model ensemble, **picked on validation half A** (scored rows).
* **Phase 3 (~65 min):** 7 students, 10 passes, lambda 0.5 (one at 0.75), with/without features, one 96-unit.
* **Judge on half B.** References: sweep 5 m50_e10 0.6769, sweep 4 st_big_e10 0.6764 (leaderboard 0.65396).

Needs in Drive: `mask_model/` (classifier + split.json), `cache/mask_train.npy`, `cache/t_big.npy` (sweep 5).
Extra features cost ~4.5 us/row on the Mac (solution.py ships `wnn_features.py`); section 8 times it.
""")
    cells += [
        md("## 3b. Export check (now also models with extra features, computed online in solution.py)"),
        code("!python -m pipeline.test_export 2>&1 | grep -E 'OK|MISMATCH'"),
        md("## 4. Cached inputs from sweep 5"),
        code("""
MASK, CACHE = f'{OUT_DIR}/mask_model', f'{OUT_DIR}/cache'
ROWW, SPLIT = '/content/wnn/mask_train.npy', f'{OUT_DIR}/mask_model/split.json'
!cp {CACHE}/mask_train.npy {ROWW} && cp {CACHE}/t_big.npy /content/wnn/t_big.npy && ls -la /content/wnn/*.npy
print(open(f'{MASK}/report.json').read())
"""),
        md("## 5. Phase 1: mask-weighted teacher models (5 passes)"),
        experiments_cell(SWEEP6_TEACHERS),
        code("TEACH = f'{OUT_DIR}/sweep6_teachers'\n"
             "!python -m pipeline.multi_train --config experiments.json --epochs 5 --evals-per-epoch 1 --threads 8 "
             "--cache-eval --row-weights {ROWW} --valid-split {SPLIT} --out-dir {TEACH} 2>&1 | grep -v '\"event\": \"batch\"'"),
        md("Phase 1 on half B. Sweep 5 references without a teacher: plain_m00 0.6704, plain_m50 0.6755."),
        code("""
import pandas as pd
def table(path):
    rows = [json.loads(l) for l in open(f'{path}/summary.jsonl')]
    df = pd.DataFrame(rows).drop_duplicates('name', keep='last')
    return df[['name', 'cost', 'valid_B_scored', 'valid_B_all', 'valid_scored', 'holdout_all']].sort_values('valid_B_scored', ascending=False)
table(TEACH)
"""),
        md("## 6. Phase 2: teacher `w` (greedy on validation half A, scored rows; reported on half B)"),
        code("""
TEACHER = '/content/wnn/t_w.npy'
!python -m pipeline.ensemble --ckpts "{TEACH}/*/best.pt" {OUT_DIR}/sweep5/plain_m50/best.pt --valid-split {SPLIT} --select-on validA --threads 8 --max-members 8 --budgets 99 --export {OUT_DIR}/teacher_w 2>&1 | grep -vE 'Warning|_generic_rnn|warnings.warn'
!python -m pipeline.teacher --ensemble {OUT_DIR}/teacher_w/cost99/submission/ensemble.json --out {TEACHER} && cp {TEACHER} {CACHE}/t_w.npy
"""),
        md("## 7. Phase 3: students (10 passes)"),
        experiments_cell(SWEEP6_STUDENTS),
        code("SWEEP = f'{OUT_DIR}/sweep6'\n"
             "!python -m pipeline.multi_train --config experiments.json --epochs 10 --evals-per-epoch 1 --threads 8 "
             "--cache-eval --row-weights {ROWW} --valid-split {SPLIT} --out-dir {SWEEP} 2>&1 | grep -v '\"event\": \"batch\"'"),
        md("## 8. Results on half B, speed of the top 3 on this CPU, upload-ready zips\n"
           "References on half B: m50_e10 0.6769 (sweep 5), st_big_e10 0.6764 (sweep 4, leaderboard 0.65396)."),
        code("""
import shutil
df = table(SWEEP); display(df)
top = df.name.head(3).tolist()
speed_args = ' '.join(f'{SWEEP}/{n}/submission' for n in top)
!python -m pipeline.speed {speed_args}
os.makedirs(f'{OUT_DIR}/zips', exist_ok=True)
for n in top:
    print(shutil.make_archive(f'{OUT_DIR}/zips/sweep6_{n}', 'zip', f'{SWEEP}/{n}/submission'))
"""),
        md("## 9. Did the weak spots improve? Error by slice on half B: best sweep-6 student vs sweep 5's m50_e10 "
           "(leaderboard 0.65444). Before: high-a3 rows 0.35, heavily-scored sequences 0.37, calmest sequences 0.48."),
        code("""
for name, ck in [('sweep 6 best: ' + top[0], f'{SWEEP}/{top[0]}/best.pt'), ('sweep 5 m50_e10', f'{OUT_DIR}/sweep5/m50_e10/best.pt')]:
    print('=' * 20, name)
    !python -m analysis.error_slices --ckpt {ck} --seqs 300 --valid-split {SPLIT} 2>&1 | grep -vE 'Warning|warn'
"""),
    ]
    write(cells, "connectome_sweep6.ipynb")


def sweep7():
    cells = setup_cells("Alpha Connectome, sweep 7: training dynamics + TCN & minGRU teachers", """
20+ teams sit at 0.669-0.674 on the leaderboard (we are at 0.657), and a public research log describes a single
128-unit GRU, the same size as ours, near 0.672. Sweep 7 tests two things in one shared pass over the data:

1. **GRU training dynamics (9 models):**
   * **Updates per pass:** we use 256 sequences per optimizer step (~1,600 steps per pass). Here: 256 / 64 / 32.
   * **Backprop length:** gradients reach back 500 steps. Here: 500 / 2,000 / 5,000 / 20,000 (the whole sequence).
2. **Non-GRU architectures for teacher diversity (4 models, see `research/architectures_cnn_rnn_transformer.md`):**
   * `mingru_h192_l3` (cost 1.00x, ONNX 1.23x baseline time) and `mingru_h192_l3_b32_c5000` (parallel log-space scan makes 5,000-step backprop cheap).
   * `tcn_c64_l11` (causal dilated Conv1d, kernel 2, 11 layers, 2,048-row receptive field, cost 0.75x) and `tcn_c128_k3_l7` (kernel 3, 7 layers, 2,187-row receptive field, teacher candidate).

All 13 models: mask-weighted (lambda 0.5), no teacher, 4 passes, judged on validation half B.
Reference: sweep 5 plain_m50 (same recipe, 5 passes) = 0.6755 on half B.
Needs `mask_model/` and `cache/mask_train.npy` in Drive.
""")
    cells += [
        md("## 4. Cached inputs from sweep 5"),
        code("""
MASK, CACHE = f'{OUT_DIR}/mask_model', f'{OUT_DIR}/cache'
ROWW, SPLIT = '/content/wnn/mask_train.npy', f'{OUT_DIR}/mask_model/split.json'
!cp {CACHE}/mask_train.npy {ROWW} && ls -la {ROWW}
"""),
        md("## 5. Experiments"),
        experiments_cell(SWEEP7),
        md("## 6. Train (4 passes, one evaluation per pass)"),
        code("SWEEP = f'{OUT_DIR}/sweep7'\n"
             "!python -m pipeline.multi_train --config experiments.json --epochs 4 --evals-per-epoch 1 --threads 8 "
             "--cache-eval --row-weights {ROWW} --valid-split {SPLIT} --out-dir {SWEEP} 2>&1 | grep -v '\"event\": \"batch\"'"),
        md("## 7. Results on half B (reference: sweep 5 plain_m50 = 0.6755)"),
        code("""
import pandas as pd
rows = [json.loads(l) for l in open(f'{SWEEP}/summary.jsonl')]
df = pd.DataFrame(rows).drop_duplicates('name', keep='last')
df[['name', 'cost', 'valid_B_scored', 'valid_B_all', 'valid_scored', 'holdout_all']].sort_values('valid_B_scored', ascending=False)
"""),
        md("Training time and learning curve per model (from the logs):"),
        code("""
for n in df.name:
    L = [json.loads(l) for l in open(f'{SWEEP}/{n}/log.jsonl')]
    ev = [l for l in L if l.get('event') == 'eval']
    print(f"{n:26s} evals (half-B subset): {[e['valid'] for e in ev]}")
"""),
        md("## 8. Do minGRU and TCN add diversity to a GRU teacher?\n"
           "Checks prediction correlation with the best single model and greedy ensemble selection on validation half A (reported on half B), plus single-core ONNX speed for the new architectures."),
        code("""
!python -m pipeline.ensemble --ckpts "{SWEEP}/*/best.pt" --valid-split {SPLIT} --select-on validA --threads 8 --max-members 8 --budgets 1.5 99 2>&1 | grep -vE 'Warning|_generic_rnn|warnings.warn'
!python -m pipeline.speed {SWEEP}/b256_c500/submission {SWEEP}/mingru_h192_l3/submission {SWEEP}/tcn_c64_l11/submission
"""),
    ]
    write(cells, "connectome_sweep7.ipynb")


if __name__ == "__main__":
    sweep1()
    sweep2()
    sweep3()
    sweep4()
    sweep5()
    sweep6()
    sweep7()
