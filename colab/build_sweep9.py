"""Write the sweep 9 one-cell Colab launcher (colab/connectome_sweep9_onecell.py + colab/connectome_sweep9.ipynb)
with the current pipeline/ code embedded, so the notebook needs no other notebook in Drive.

    python colab/build_sweep9.py
"""
import glob
import json
import os
import pprint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = {f"pipeline/{os.path.basename(p)}": open(p).read() for p in sorted(glob.glob(os.path.join(ROOT, "pipeline", "*.py")))}

HEADER = '''# Alpha Connectome, sweep 9: a stronger mask-weighted teacher, born-again distillation, auxiliary targets and
# per-sequence normalised inputs. One A100 run (~3.5 h), self-contained (the whole pipeline is embedded below).
#   Phase A (5 passes, no teacher): 6 new teacher members + 4 test models (aux targets 0.3 / 1.0, norm inputs,
#            norm + aux), each comparable with t9_h128_s0.                                 -> runs/sweep9_teachers
#   Phase B: teacher t9 = greedy ensemble (validation half A) over every mask-weighted plain model so far
#            (sweeps 5-9, incl. minGRU / TCN); born-again teacher ba = average of the 6 existing students.
#   Phase C (10 passes): students of t9 (2 seeds, + aux, + norm) and of ba.                 -> runs/sweep9
# Re-runnable: finished models and cached teachers (runs/cache/t_9.npy, t_ba.npy) are skipped.
# Keep this tab open while it runs.
'''

BODY = r'''import os, subprocess, sys, json, glob, shutil, time
from google.colab import drive
drive.mount('/content/drive')
OUT_DIR = '/content/drive/MyDrive/wunder-connectome/runs'
os.makedirs(OUT_DIR, exist_ok=True)

def drive_writable():
    """Drive can turn read-only (storage full, or the mount drops during the 30-min download): check before writing."""
    for attempt in range(2):
        try:
            with open(f'{OUT_DIR}/.write_test', 'w') as f: f.write('ok')
            os.remove(f'{OUT_DIR}/.write_test'); print('Drive is writable', flush=True); return
        except OSError as e:
            print(f'Drive not writable ({e}), remounting', flush=True)
            try: drive.flush_and_unmount()
            except Exception: pass
            drive.mount('/content/drive', force_remount=True)
    raise RuntimeError('Google Drive is read-only: most likely its storage is full. Free space (and empty the Drive '
                       'Trash) at drive.google.com/settings/storage, then re-run this cell; the dataset stays on this machine.')

drive_writable()

def sh(cmd):
    p = subprocess.Popen(cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end='', flush=True)
    assert p.wait() == 0, 'command failed: ' + cmd[:80]

sh('nvidia-smi --query-gpu=name,memory.total --format=csv; echo "CPUs: $(nproc)"; free -g | head -2')
DATA = r"""mkdir -p /content/wnn && cd /content/wnn
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
du -sh datasets"""
sh(DATA)
sh('pip -q install onnx onnxruntime')
os.chdir('/content/wnn')

# the pipeline code (pipeline/*.py from the Mac, 2026-11-03)
FILES = __FILES__
os.makedirs('/content/wnn/pipeline', exist_ok=True)
for rel, body in FILES.items():
    open(f'/content/wnn/{rel}', 'w').write(body)
sys.path.insert(0, '/content/wnn')
for m in [m for m in sys.modules if m == 'pipeline' or m.startswith('pipeline.')]:
    del sys.modules[m]  # a previous run in this session may have loaded older code
from pipeline.trainer import ExpConfig, N_AUX
from pipeline.features import KINDS
assert 'aux' in ExpConfig.__dataclass_fields__ and 'norm' in KINDS
print(len(FILES), 'pipeline files written; feature kinds', KINDS, '| aux targets', N_AUX, flush=True)
sh('cd /content/wnn && python -m pipeline.fast_eval --baseline --valid-seqs 300')

CACHE = f'{OUT_DIR}/cache'
ROWW, SPLIT = '/content/wnn/mask_train.npy', f'{OUT_DIR}/mask_model/split.json'
sh(f'cp {CACHE}/mask_train.npy {ROWW} && ls -la {ROWW}')
FLAGS = f'--evals-per-epoch 1 --threads 8 --cache-eval --row-weights {ROWW} --valid-split {SPLIT}'
import pandas as pd

def train(experiments, out, epochs):
    """Train what has no results.json in `out` yet (re-runnable)."""
    done = [e['name'] for e in experiments if os.path.exists(f"{out}/{e['name']}/results.json")]
    todo = [e for e in experiments if e['name'] not in done]
    print(f"\n{len(todo)} to train in {out}: {[e['name'] for e in todo]}" + (f"; already done: {done}" if done else ''), flush=True)
    if not todo:
        return
    drive_writable()
    json.dump(todo, open('experiments.json', 'w'), indent=1)
    sh(f"""set -o pipefail; cd /content/wnn && python -m pipeline.multi_train --config experiments.json --epochs {epochs} {FLAGS} --out-dir {out} 2>&1 | grep -v '"event": "batch"'""")

def table(names, dirs):
    rows = [json.loads(l) for d in dirs if os.path.exists(f'{OUT_DIR}/{d}/summary.jsonl') for l in open(f'{OUT_DIR}/{d}/summary.jsonl')]
    df = pd.DataFrame(rows).drop_duplicates('name', keep='last')
    df = df[df.name.isin(names)].sort_values('valid_B_scored', ascending=False)
    print(df[['name', 'cost', 'valid_B_scored', 'valid_B_all', 'valid_scored', 'holdout_all']].to_string(index=False), flush=True)

# ---- Phase A: new teacher members + tests of auxiliary targets and per-sequence normalised inputs (5 passes)
def tm(name, **kw):  # plain mask-weighted model, no teacher
    return {'name': name, 'mask_weight': 0.5, **kw}
PHASE_A = [tm('t9_h128_s0'), tm('t9_h128_s1', seed=1), tm('t9_h128_s2', seed=2), tm('t9_h128_s3', seed=3),
           tm('t9_h192_drop_s0', hidden=192, dropout=0.1), tm('t9_h192_drop_s1', hidden=192, dropout=0.1, seed=1),
           tm('t9_aux03', aux=0.3), tm('t9_aux1', aux=1.0), tm('t9_norm', feats='norm'), tm('t9_norm_aux03', feats='norm', aux=0.3)]
train(PHASE_A, f'{OUT_DIR}/sweep9_teachers', epochs=5)
table([e['name'] for e in PHASE_A] + ['tw_h128_s0', 'tw_h128_s1', 'tw_h192_drop'], ['sweep6_teachers', 'sweep9_teachers'])
print('compare aux / norm with t9_h128_s0 (same recipe, seed 0); sweep 6 members (also 5 passes): tw_*', flush=True)

# ---- Phase B: teachers
def cached_teacher(name, build):
    local, kept = f'/content/wnn/t_{name}.npy', f'{CACHE}/t_{name}.npy'
    if os.path.exists(kept):
        print(f'teacher {name}: cached in Drive, copying', flush=True); sh(f'cp {kept} {local}')
    else:
        build(local); drive_writable(); sh(f'cp {local} {kept}')

POOL = [f'{OUT_DIR}/sweep6_teachers/*/best.pt', f'{OUT_DIR}/sweep5/plain_m50/best.pt', f'{OUT_DIR}/sweep7/*/best.pt',
        f'{OUT_DIR}/sweep8/mingru*/best.pt', f'{OUT_DIR}/sweep8/tcn*/best.pt', f'{OUT_DIR}/sweep9_teachers/*/best.pt']
def build_t9(local):
    pool = ' '.join(f'"{p}"' for p in POOL)
    sh(f'cd /content/wnn && python -m pipeline.ensemble --ckpts {pool} --valid-split {SPLIT} --select-on validA --threads 8 '
       f'--max-members 12 --budgets 99 --export {OUT_DIR}/teacher_9 2>&1 | grep -vE "Warning|_generic_rnn|warnings.warn"')
    sh(f'cd /content/wnn && python -m pipeline.teacher --ensemble {OUT_DIR}/teacher_9/cost99/submission/ensemble.json --out {local}')
cached_teacher('9', build_t9)
print('teacher w for reference: half B 0.6786 (6 models)', flush=True)

STUDENTS = [f'{OUT_DIR}/sweep6_students/sw_h128/best.pt', f'{OUT_DIR}/sweep6_students/sw_h128_s1/best.pt',
            f'{OUT_DIR}/sweep8/sw_feat_h96/best.pt', f'{OUT_DIR}/sweep8/sw_feat_h128/best.pt',
            f'{OUT_DIR}/sweep8/sw_feat_h128_m75/best.pt', f'{OUT_DIR}/sweep8/sw_feat_h128_s1/best.pt']
cached_teacher('ba', lambda local: sh(f'cd /content/wnn && python -m pipeline.teacher --ckpts {" ".join(STUDENTS)} --out {local}'))

# ---- Phase C: students (10 passes)
def st(name, teacher, **kw):
    return {'name': name, 'distill': 1.0, 'teacher': f'/content/wnn/t_{teacher}.npy', 'mask_weight': 0.5, **kw}
PHASE_C = [st('s9_h128', '9'), st('s9_h128_s1', '9', seed=1), st('s9_aux03', '9', aux=0.3), st('s9_norm', '9', feats='norm'),
           st('sba_h128', 'ba')]
SWEEP = f'{OUT_DIR}/sweep9'
train(PHASE_C, SWEEP, epochs=10)
os.makedirs(f'{OUT_DIR}/zips', exist_ok=True)
for e in PHASE_C:  # upload-ready zips
    d = f"{SWEEP}/{e['name']}/submission"
    if os.path.isdir(d): print(shutil.make_archive(f"{OUT_DIR}/zips/sweep9_{e['name']}", 'zip', d), flush=True)
table([e['name'] for e in PHASE_C] + ['sw_h128', 'sw_h128_s1'], ['sweep6_students', 'sweep9'])
print('references on half B: sw_h128_s1 0.6782 (leaderboard 0.65660), the student of teacher w', flush=True)
sh('cd /content/wnn && python -m pipeline.speed ' + ' '.join(f"{SWEEP}/{e['name']}/submission" for e in PHASE_C))  # grader ~ 0.52 x this
'''


def main():
    body = BODY.replace("__FILES__", pprint.pformat(FILES, width=120))
    src = HEADER + body
    compile(src, "connectome_sweep9_onecell.py", "exec")
    out = os.path.join(ROOT, "colab")
    open(os.path.join(out, "connectome_sweep9_onecell.py"), "w").write(src)
    nb = {"nbformat": 4, "nbformat_minor": 0,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "machine_shape": "hm", "gpuType": "A100"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
          "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": src.splitlines(keepends=True)}]}
    json.dump(nb, open(os.path.join(out, "connectome_sweep9.ipynb"), "w"), indent=1)
    print(f"{len(src)} chars, {len(FILES)} pipeline files -> colab/connectome_sweep9_onecell.py, colab/connectome_sweep9.ipynb")


if __name__ == "__main__":
    main()
