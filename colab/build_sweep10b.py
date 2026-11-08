"""Write the sweep 10b one-cell Colab launchers with the current pipeline/ code embedded: colab/connectome_sweep10b.ipynb
(Drive mounted in the Colab account) and colab/connectome_sweep10b_rclone.ipynb (any Colab account, your Drive via
rclone). Sweep 10b: can group 4 be improved? (PLAN_sweep10.md, WORKSPACE.md "Sweep 10a results").

    python colab/build_sweep10b.py
"""
import glob
import json
import os
import pprint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = {f"pipeline/{os.path.basename(p)}": open(p).read() for p in sorted(glob.glob(os.path.join(ROOT, "pipeline", "*.py")))}

HEADER = '''# Alpha Connectome, sweep 10b: can group 4 (largest tick, 48% of the metric weight, WP ~0.42) be improved?
# One A100 run (~1.5 h), self-contained (the whole pipeline is embedded below). All models train on group 4 only.
# Needs from earlier sweeps in Drive: runs/sweep9/s9_h128, runs/cache/t_9.npy, runs/cache/mask_train.npy,
# runs/mask_model/split.json, runs/sweep9_teachers/t9_*, runs/sweep10a (its models join the routing).
#   Phase A (-> runs/sweep10b):
#     g4y_h128_e12, g4y_h192_drop_e12: group 4 from scratch on the true targets, 12 passes (sweep 10a's g4y_* had 5;
#        12 x 4557 group-4 sequences ~ the 5 x 10607 sequences the all-group teachers t9_* saw)
#     ft_g4_y: s9_h128 fine-tuned on group 4 with the true targets only (no teacher)
#     ft_g4_c2000: like 10a's ft_g4, truncated backprop over 2000 rows instead of 500 (price moves are rare here)
#     ft_g4_cls, ft_g4_cls_y, g4cls_h128_e12: a joint-state head: group 4's (t0, t1) take only 40 values, tied
#        ((0,-1) 29%, (-1,0) 29%, (1,-3) 8%, ...); softmax over them + cross-entropy, prediction = the
#        |y|-weighted optimum under that distribution (fine-tunes from s9_h128 with / without the teacher; from scratch)
#   Phase B: per-group table (half A = honest for single models), routing over sweep 10a + 10b models (per group the
#   best on half A), routed submission runs/sweep10b_routed, zips, speed test.
# Re-runnable: finished models are skipped. Keep this tab open while it runs.
'''

PRE = r'''import os, subprocess, sys, json, glob, shutil, time, threading

def sh(cmd):
    p = subprocess.Popen(cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end='', flush=True)
    assert p.wait() == 0, 'command failed: ' + cmd[:80]

'''

MOUNT = r'''from google.colab import drive
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

def sync_up(quiet=False):  # the rclone version uploads here
    pass

'''

RCLONE = r'''# ---- your own Google Drive through rclone (the Drive of the account running Colab is not used)
REMOTE = 'gd:wunder-connectome/runs'  # = My Drive/wunder-connectome/runs of the account you authorized on your Mac
OUT_DIR = '/content/runs'             # local working copy: what this sweep needs (downloaded) and makes (uploaded)
RC = 'rclone --transfers 8 --checkers 16 --drive-chunk-size 64M --retries 5 --low-level-retries 20 --stats-one-line --stats 60s'
if not shutil.which('rclone'):
    sh('cd /tmp && curl -sSLO https://downloads.rclone.org/rclone-current-linux-amd64.zip && unzip -oq rclone-current-linux-amd64.zip '
       '&& cp rclone-v*-linux-amd64/rclone /usr/local/bin/ && rclone version | head -1')

def parse_token(s):
    """The text `rclone authorize "drive"` prints (a JSON token, or the newer base64 blob), with or without the
    'Paste the following ...' / '<---End paste' lines around it."""
    import base64
    s = s.strip()
    if '--->' in s: s = s.split('--->', 1)[1]
    s = s.split('<---')[0].strip()
    try:
        t = json.loads(s)
    except ValueError:
        b = s.replace('+', '-').replace('/', '_')
        t = json.loads(base64.urlsafe_b64decode(b + '=' * (-len(b) % 4)))
    if isinstance(t, dict) and 'token' in t and 'access_token' not in t:
        t = t['token']
        t = json.loads(t) if isinstance(t, str) else t
    assert isinstance(t, dict) and 'refresh_token' in t, 'that is not the token printed by rclone authorize "drive"'
    return json.dumps(t)

if REMOTE.startswith('gd:') and 'gd:' not in subprocess.run('rclone listremotes', shell=True, capture_output=True, text=True).stdout:
    tok = ''
    try:
        from google.colab import userdata  # Colab secret (key icon on the left), if you made one
        tok = userdata.get('RCLONE_DRIVE_TOKEN') or ''
    except Exception:
        pass
    if not tok:
        import getpass
        tok = getpass.getpass('Paste what  rclone authorize "drive"  printed on your Mac, then press Enter: ')
    conf = subprocess.run('rclone config file', shell=True, capture_output=True, text=True).stdout.strip().splitlines()[-1]
    os.makedirs(os.path.dirname(conf), exist_ok=True)
    with open(conf, 'a') as f:
        f.write(f'\n[gd]\ntype = drive\nscope = drive\ntoken = {parse_token(tok)}\n')
    os.chmod(conf, 0o600)
    del tok
sh(f'rclone about {REMOTE} || true; echo "in {REMOTE}:"; rclone lsf --dirs-only {REMOTE} | tr "\\n" " "; echo')

PULL = ['sweep9/summary.jsonl', 'sweep9/s9_h128/**', 'cache/t_9.npy', 'cache/mask_train.npy', 'mask_model/split.json',
        *[f'sweep9_teachers/{n}/best.pt' for n in ('t9_h128_s0', 't9_h128_s1', 't9_h128_s2', 't9_h128_s3',
                                                     't9_h192_drop_s0', 't9_h192_drop_s1')],
        'sweep10a/**', 'sweep10b/**', 'sweep10b_routed/**']  # sweep 10a's models join the routing; 10b: resume
PUSH = ['sweep10b/**', 'sweep10b_routed/**', 'zips/sweep10b*']
inc = lambda pats: ' '.join(f"--include '{p}'" for p in pats)
os.makedirs(OUT_DIR, exist_ok=True)
print('downloading the inputs from your Drive', flush=True)
sh(f'{RC} copy {REMOTE} {OUT_DIR} {inc(PULL)} && du -sh {OUT_DIR}')

_sync_lock = threading.Lock()
def sync_up(quiet=False):
    """Upload what this sweep made (new or changed files only) to your Drive."""
    with _sync_lock:
        r = subprocess.run(f'{RC} copy {OUT_DIR} {REMOTE} {inc(PUSH)}', shell=True, capture_output=True, text=True)
    if r.returncode:
        print('UPLOAD TO DRIVE FAILED (retried at the next upload):', r.stderr[-600:], flush=True)
    elif not quiet:
        print(f'uploaded to Drive ({REMOTE})', flush=True)
    return r.returncode == 0

def drive_writable():
    """Upload what is done, and check the Drive still takes writes (storage full / token revoked) before training."""
    sync_up()
    r = subprocess.run(f'echo ok | rclone rcat {REMOTE}/.write_test && rclone deletefile {REMOTE}/.write_test',
                       shell=True, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError('cannot write to your Drive through rclone (storage full? token revoked?): ' + r.stderr[-600:])
    print('Drive (rclone) is writable', flush=True)

def _upload_every(sec):
    while True:
        time.sleep(sec)
        sync_up(quiet=True)

drive_writable()
if not any(t.name == 'drive-upload' for t in threading.enumerate()):  # every 10 min, so a dropped runtime loses little
    threading.Thread(target=_upload_every, args=(600,), name='drive-upload', daemon=True).start()

'''

BODY = r'''sh('nvidia-smi --query-gpu=name,memory.total --format=csv; echo "CPUs: $(nproc)"; free -g | head -2')
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

# the pipeline code (pipeline/*.py from the Mac, 2026-11-04)
FILES = __FILES__
os.makedirs('/content/wnn/pipeline', exist_ok=True)
for rel, body in FILES.items():
    open(f'/content/wnn/{rel}', 'w').write(body)
sys.path.insert(0, '/content/wnn')
for m in [m for m in sys.modules if m == 'pipeline' or m.startswith('pipeline.')]:
    del sys.modules[m]  # a previous run in this session may have loaded older code
from pipeline.trainer import ExpConfig
from pipeline.features import KINDS
assert 'group' in ExpConfig.__dataclass_fields__ and 'init' in ExpConfig.__dataclass_fields__ and 'grp' in KINDS
print(len(FILES), 'pipeline files written; feature kinds', KINDS, flush=True)
sh('cd /content/wnn && python -m pipeline.fast_eval --baseline --valid-seqs 300')

CACHE = f'{OUT_DIR}/cache'
ROWW, SPLIT = '/content/wnn/mask_train.npy', f'{OUT_DIR}/mask_model/split.json'
S9 = f'{OUT_DIR}/sweep9/s9_h128/best.pt'
for need in (S9, f'{CACHE}/t_9.npy', f'{CACHE}/mask_train.npy', SPLIT):
    assert os.path.exists(need), f'missing in Drive: {need} (made by sweeps 5-9)'
sh(f'cp {CACHE}/mask_train.npy {ROWW} && ls -la {ROWW}')
T9 = '/content/wnn/t_9.npy'
if not os.path.exists(T9):
    sh(f'cp {CACHE}/t_9.npy {T9}')
sh('cd /content/wnn && python -m pipeline.groups --check')  # group detector on all validation sequences
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
    cols = ['name', 'cost', 'valid_B_scored'] + [c for c in ('B_g1', 'B_g2', 'B_g3', 'B_g4', 'group') if c in df] + ['valid_B_all', 'holdout_all']
    print(df[cols].to_string(index=False), flush=True)

# ---- Phase A: group 4 only. Fine-tunes of s9_h128 (6 passes), models from scratch (12 passes)
def ft4(name, **kw):  # like sweep 10a's ft_g4: s9_h128 on group 4, lr 5e-4, half teacher / half targets
    return {'name': name, 'init': S9, 'group': 3, 'lr': 5e-4, 'epochs': 6, 'distill': 0.5, 'teacher': T9, 'mask_weight': 0.5, **kw}
def y4(name, **kw):  # like sweep 10a's g4y_*: group 4 from scratch, true targets only, but 12 passes instead of 5
    return {'name': name, 'group': 3, 'epochs': 12, 'mask_weight': 0.5, **kw}
PHASE_A = [y4('g4y_h128_e12'), y4('g4y_h192_drop_e12', hidden=192, dropout=0.1),
           ft4('ft_g4_y', distill=0.0), ft4('ft_g4_c2000', chunk=2000),
           ft4('ft_g4_cls', states=3, cls=0.5, lr=1e-3), ft4('ft_g4_cls_y', states=3, cls=0.5, lr=1e-3, distill=0.0),
           y4('g4cls_h128_e12', states=3, cls=0.5)]
SWEEP = f'{OUT_DIR}/sweep10b'
train(PHASE_A, SWEEP, epochs=12)
sync_up()
table([e['name'] for e in PHASE_A] + ['ft_g4', 'g4y_h128', 'g4y_h192_drop', 's9_h128'], ['sweep9', 'sweep10a', 'sweep10b'])
print('Group 4 references (B_g4): s9_h128 0.4205, ft_g4 0.4192, g4y_h128 (5 passes) 0.4121, g4y_h192_drop 0.4173, '
      't9_h128_s0..s3 0.4169-0.4199, t9_h192_drop 0.4156/0.4179. B is partly the checkpoint-selection set: the routing '
      'table below also has half A (A_g4: s9_h128 0.4459)', flush=True)

# ---- Phase B: routing over sweep 10a + 10b (per group, the best on half A), routed submission, zips, speed
ROUTED = f'{OUT_DIR}/sweep10b_routed/submission'
drive_writable()
REFS = ' '.join(f'"{OUT_DIR}/sweep9_teachers/{n}/best.pt"' for n in   # shared models trained like g4y_* (5 passes, no teacher)
                ['t9_h128_s0', 't9_h128_s1', 't9_h128_s2', 't9_h128_s3', 't9_h192_drop_s0', 't9_h192_drop_s1'])
sh(f'cd /content/wnn && python -m pipeline.routed --shared {S9} --cands "{OUT_DIR}/sweep10a/*/best.pt" "{SWEEP}/*/best.pt" {REFS} --valid-split {SPLIT} '
   f'--export {ROUTED} 2>&1 | grep -vE "Warning|_generic_rnn|warnings.warn"')
os.makedirs(f'{OUT_DIR}/zips', exist_ok=True)
print(shutil.make_archive(f'{OUT_DIR}/zips/sweep10b_routed', 'zip', ROUTED), flush=True)
sync_up()  # zips and the routed submission to Drive before the speed test
print('decision: a group-4 option beats s9_h128 on BOTH A_g4 (0.4459) and B_g4 (0.4205) -> group 4 is improvable; '
      'submit sweep10b_routed.zip if "routed, detected groups" valid_B > 0.67892 (sweep 10a routed) and valid_A rose too', flush=True)
sh(f'cd /content/wnn && python -m pipeline.speed {ROUTED} {OUT_DIR}/sweep9/s9_h128/submission')  # grader ~ 0.52 x this
sync_up()
print(f"done; results in Drive: {globals().get('REMOTE', OUT_DIR)}", flush=True)
'''


RCLONE_HEADER = '''# rclone version: runs in any Colab account and reads/writes YOUR Drive (My Drive/wunder-connectome/runs) through
# rclone, so the Colab account's own Drive is not used. Once, on your Mac:  brew install rclone  then
#   rclone authorize "drive"
# sign in with YOUR Google account in the browser that opens, and copy everything it prints after "--->".
# Paste it when this cell asks (or store it as a Colab secret RCLONE_DRIVE_TOKEN: key icon on the left, notebook
# access on). It downloads only the inputs it needs (~2 GB), works on this machine's disk, and uploads its results
# every 10 minutes and at the end. The token can read your whole Drive: it stays on this runtime (deleted when the
# runtime ends); revoke it any time at myaccount.google.com/connections ("rclone").
'''


def write(name, src):
    compile(src, f"{name}_onecell.py", "exec")
    out = os.path.join(ROOT, "colab")
    open(os.path.join(out, f"{name}_onecell.py"), "w").write(src)
    nb = {"nbformat": 4, "nbformat_minor": 0,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "machine_shape": "hm", "gpuType": "A100"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
          "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": src.splitlines(keepends=True)}]}
    json.dump(nb, open(os.path.join(out, f"{name}.ipynb"), "w"), indent=1)
    print(f"{len(src)} chars, {len(FILES)} pipeline files -> colab/{name}_onecell.py, colab/{name}.ipynb")


def main():
    body = BODY.replace("__FILES__", pprint.pformat(FILES, width=120))
    write("connectome_sweep10b", HEADER + PRE + MOUNT + body)
    write("connectome_sweep10b_rclone", HEADER + RCLONE_HEADER + PRE + RCLONE + body)


if __name__ == "__main__":
    main()
