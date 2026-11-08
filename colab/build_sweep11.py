"""Write the sweep 11 one-cell Colab launchers with the current pipeline/ code embedded: colab/connectome_sweep11.ipynb
(Drive mounted in the Colab account) and colab/connectome_sweep11_rclone.ipynb (any Colab account, your Drive via
rclone). Sweep 11 = sweep 10b (can group 4 be improved?) + training on validation (WORKSPACE.md "Sweep 11 plan").

    python colab/build_sweep11.py
"""
import glob
import json
import os
import pprint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = {f"pipeline/{os.path.basename(p)}": open(p).read() for p in sorted(glob.glob(os.path.join(ROOT, "pipeline", "*.py")))}

HEADER = '''# Alpha Connectome, sweep 11: train on validation (another period than train, like the test set) + sweep 10b.
# One A100 run (~2-2.5 h), self-contained (the whole pipeline is embedded below).
# Needs from earlier sweeps in Drive: runs/sweep9/s9_h128, runs/cache/t_9.npy, runs/cache/mask_train.npy,
# runs/mask_model/split.json, runs/sweep9_teachers/t9_*, runs/sweep10a (specialists, routing).
# Part 1 = sweep 10b, can group 4 be improved? Phase A (-> runs/sweep10b):
#     g4y_h128_e12, g4y_h192_drop_e12: group 4 from scratch on the true targets, 12 passes (sweep 10a's g4y_* had 5;
#        12 x 4557 group-4 sequences ~ the 5 x 10607 sequences the all-group teachers t9_* saw)
#     ft_g4_y: s9_h128 fine-tuned on group 4 with the true targets only (no teacher)
#     ft_g4_c2000: like 10a's ft_g4, truncated backprop over 2000 rows instead of 500 (price moves are rare here)
#     ft_g4_cls, ft_g4_cls_y, g4cls_h128_e12: a joint-state head: group 4's (t0, t1) take only 40 values, tied
#        ((0,-1) 29%, (-1,0) 29%, (1,-3) 8%, ...); softmax over them + cross-entropy, prediction = the
#        |y|-weighted optimum under that distribution (fine-tunes from s9_h128 with / without the teacher; from scratch)
#   Phase B (at the very end): routing over sweep 10a + 10b models (per group the best on half A), runs/sweep10b_routed.
# Part 2 = sweep 11, validation as training data. Honest scores by cross-fitting: a model fine-tuned on validation half
#   A is scored on half B and vice versa (pipeline/crossfit.py); true mask weights (scored 1, unscored u).
#   11a fine-tunes (-> runs/sweep11_ft, each on A and on B): s9_h128, Adam const lr, 3 passes, average of the passes 2-3:
#       v_lr2_u06 (lr 2e-4, u 0.06), v_lr2_u30 (u 0.3), v_lr2_u100 (u 1), v_lr5_u06 (lr 5e-4), v_lr2_u06_e6 (6 passes,
#       average 4-6), v_lr2_u06_anc (half the loss: stay near s9_h128's own predictions); vg1..vg3: the sweep 10a group
#       specialists ft_g1..3 fine-tuned on their group's validation sequences.
#   11b controls (-> runs/sweep11_ctl, 10 passes, final weights): c1_trv (s9 recipe on train + half A/B x3, labels on
#       the validation rows), c2_v (from scratch on half A/B only, x3 -> 30 passes), c3_trv_all (train + all valid x3:
#       not scorable here, a leaderboard probe); importance weights: w05 / w025 = s9 recipe on train only, each train
#       sequence weighted by how validation-like it is ((p/(1-p))^gamma, gamma 0.5 / 0.25, cap 5, per group; p from a
#       logistic regression on sequence
#       statistics fitted on half A / half B, pipeline/shift.py); c0_tr = the same without weights (their reference).
#   11c cross-fit table (runs/sweep11_crossfit.json), also each fine-tune at 0.9 / 0.1 weights with its start (hedge).
#   11d the best fine-tune recipe (and the group recipes that beat it) retrained on all of validation (runs/sweep11_final)
#       -> zips/sweep11_routed.zip (submit), sweep11_hedge.zip (0.9 / 0.1 weights: a safer final pick),
#       sweep11_single.zip, sweep11_c3.zip; speed test.
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
        'sweep10a/**', 'sweep10b/**', 'sweep10b_routed/**',  # sweep 10a's models join the routing; 10b / 11: resume
        'cache/tv_s9.npy', 'cache/shift/**', 'sweep11_*/**', 'sweep11_crossfit.json']
PUSH = ['sweep10b/**', 'sweep10b_routed/**', 'zips/sweep10b*', 'cache/tv_s9.npy', 'cache/shift/**', 'sweep11_*/**',
        'sweep11_crossfit.json',
        'zips/sweep11*']
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

# the pipeline code (pipeline/*.py from the Mac, 2026-11-05)
FILES = __FILES__
os.makedirs('/content/wnn/pipeline', exist_ok=True)
for rel, body in FILES.items():
    open(f'/content/wnn/{rel}', 'w').write(body)
sys.path.insert(0, '/content/wnn')
for m in [m for m in sys.modules if m == 'pipeline' or m.startswith('pipeline.')]:
    del sys.modules[m]  # a previous run in this session may have loaded older code
from pipeline.trainer import ExpConfig
from pipeline.features import KINDS
assert 'valid_part' in ExpConfig.__dataclass_fields__ and 'swa_from' in ExpConfig.__dataclass_fields__ and 'grp' in KINDS
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

def train(experiments, out, epochs, flags=''):
    """Train what has no results.json in `out` yet (re-runnable). `flags`: extra Common flags (the last one wins)."""
    done = [e['name'] for e in experiments if os.path.exists(f"{out}/{e['name']}/results.json")]
    todo = [e for e in experiments if e['name'] not in done]
    print(f"\n{len(todo)} to train in {out}: {[e['name'] for e in todo]}" + (f"; already done: {done}" if done else ''), flush=True)
    if not todo:
        return
    drive_writable()
    json.dump(todo, open('experiments.json', 'w'), indent=1)
    sh(f"""set -o pipefail; cd /content/wnn && python -m pipeline.multi_train --config experiments.json --epochs {epochs} {FLAGS} {flags} --out-dir {out} 2>&1 | grep -v '"event": "batch"'""")

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

# ======== Part 2: sweep 11, validation as training data
SPEC = [f'{OUT_DIR}/sweep10a/ft_g{g + 1}/best.pt' for g in range(3)]  # the sweep 10a group specialists (10a routed g1-3)
for need in SPEC:
    assert os.path.exists(need), f'missing in Drive: {need} (sweep 10a)'
TV = '/content/wnn/tv_s9.npy'  # s9_h128's predictions on valid: the anchor of v_lr2_u06_anc
if not os.path.exists(TV):
    if os.path.exists(f'{CACHE}/tv_s9.npy'):
        sh(f'cp {CACHE}/tv_s9.npy {TV}')
    else:
        sh(f'cd /content/wnn && python -m pipeline.teacher --ckpts {S9} --data datasets/valid.parquet --out {TV} && cp {TV} {CACHE}/tv_s9.npy')
FLAGS_V = f'--pool valid --teacher-valid {TV} --evals-per-epoch 0 --skip-final-eval'

# ---- 11a: fine-tunes on validation half A / half B (pairs NAME_A, NAME_B)
def ftv(name, part, **kw):
    return {'name': f'{name}_{part}', 'init': S9, 'lr': 2e-4, 'epochs': 3, 'sched': 'const', 'select': False, 'swa_from': 2,
            'valid_part': part, 'unscored_weight': 0.06, **kw}
FT = [c for part in 'AB' for c in (
    ftv('v_lr2_u06', part), ftv('v_lr2_u30', part, unscored_weight=0.3), ftv('v_lr2_u100', part, unscored_weight=1.0),
    ftv('v_lr5_u06', part, lr=5e-4), ftv('v_lr2_u06_e6', part, epochs=6, swa_from=4),
    ftv('v_lr2_u06_anc', part, distill=0.5, teacher=T9, valid_distill=1.0),
    *[ftv(f'vg{g + 1}_lr2_u06', part, init=SPEC[g], group=g) for g in range(3)])]
SW_FT = f'{OUT_DIR}/sweep11_ft'
train(FT, SW_FT, epochs=6, flags=FLAGS_V)
sync_up()

# ---- 11b: controls from scratch with validation in the training pool (10 passes, final weights), and importance
# weights for the train sequences (how validation-like each one is; fitted on half A -> w_A, on half B -> w_B)
SHIFT = f'{CACHE}/shift'
if not os.path.exists(f'{SHIFT}/w_B_g0.25.npy'):
    sh(f'cd /content/wnn && python -m pipeline.shift --valid-split {SPLIT} --gamma 0.25 0.5 --cap 5 --out-dir {SHIFT}')
    sync_up()
else:
    print(f'importance weights already made: {SHIFT}', flush=True)
def ctl(name, **kw):
    return {'name': name, 'distill': 1.0, 'teacher': T9, 'mask_weight': 0.5, 'select': False, 'valid_repeat': 3,
            'unscored_weight': 0.3, **kw}  # s9_h128's recipe on the train rows, the labels on the validation rows
CTL = [ctl('c1_trv_A', valid_part='A'), ctl('c1_trv_B', valid_part='B'),
       ctl('c2_v_A', valid_part='A', use_train=False, distill=0.0), ctl('c2_v_B', valid_part='B', use_train=False, distill=0.0),
       ctl('c3_trv_all', valid_part='all'),
       ctl('c0_tr'), *[ctl(f'w{tag}_{H}', seq_weights=f'{SHIFT}/w_{H}_g{gm}.npy') for tag, gm in (('05', '0.5'), ('025', '0.25'))
                       for H in 'AB']]
SW_CTL = f'{OUT_DIR}/sweep11_ctl'
train(CTL, SW_CTL, epochs=10, flags='--pool train+valid --evals-per-epoch 0 --skip-final-eval')
sync_up()

# ---- 11c: cross-fitted scores (each model on the validation half it did not train on)
CF = f'{OUT_DIR}/sweep11_crossfit.json'
if not os.path.exists(CF):
    sh(f'set -o pipefail; cd /content/wnn && python -m pipeline.crossfit score --valid-split {SPLIT} --base {S9} --base-spec {" ".join(SPEC)} '
       f'--recipes "{SW_FT}/*_A/best.pt" "{SW_CTL}/*_A/best.pt" --refs {SW_CTL}/c0_tr/best.pt --hedge 0.9 --out {CF} 2>&1 | grep -vE "Warning|_generic_rnn|warnings.warn"')
    sync_up()
else:
    print(f'cross-fit table already made: {CF}', flush=True)
cf = json.load(open(CF))
print('\ncross-fitted scores (pooled scored-row WP over all of valid, each sequence from a model that never saw it):', flush=True)
for name, r in cf['scores'].items():
    print(f"  {name[:60]:60s} all {r['all']:.5f}  A {r['A']:.5f}  B {r['B']:.5f}", flush=True)
choice = cf['choice']
print('choice:', choice['best_shared'], '| route g1..g4:', choice['route'], flush=True)

# ---- 11d: the chosen recipes on all of validation -> submissions
cfg_of = {c['name'][:-2]: c for c in FT if c['name'].endswith('_A')}
FINAL = [{**cfg_of[nm], 'name': f'{nm}_all', 'valid_part': 'all'} for nm in dict.fromkeys(choice['route'])]
SW_FIN = f'{OUT_DIR}/sweep11_final'
train(FINAL, SW_FIN, epochs=6, flags=FLAGS_V)
route = [f'{SW_FIN}/{nm}_all/best.pt' for nm in choice['route']]
fallback = f"{SW_FIN}/{choice['best_shared']}_all/best.pt"
os.makedirs(f'{OUT_DIR}/zips', exist_ok=True)
SUBS = {}
for tag, alpha in (('routed', 1.0), ('hedge', 0.9)):
    sub = f'{OUT_DIR}/sweep11_{tag}/submission'
    if not os.path.exists(f'{sub}/routes.json'):
        sh(f'set -o pipefail; cd /content/wnn && python -m pipeline.crossfit export --route {" ".join(route)} --fallback {fallback} '
           f'--alpha {alpha} --out {sub} 2>&1 | grep -vE "Warning|_generic_rnn|warnings.warn"')
    SUBS[tag] = sub
SUBS['single'] = f"{SW_FIN}/{choice['best_shared']}_all/submission"
SUBS['c3'] = f'{SW_CTL}/c3_trv_all/submission'
for tag, sub in SUBS.items():
    print(shutil.make_archive(f'{OUT_DIR}/zips/sweep11_{tag}', 'zip', sub), flush=True)
sync_up()
sc = cf['scores']
ft_routed = next(v for k, v in sc.items() if k.startswith('FT routed:'))
base_routed = sc.get('base routed (spec g1-3, s9 g4)', sc['s9 (base)'])
print(f"\ndecision: cross-fitted all-valid WP, FT routed {ft_routed['all']:.5f} vs base routed {base_routed['all']:.5f} "
      f"({ft_routed['all'] - base_routed['all']:+.5f}); s9 alone {sc['s9 (base)']['all']:.5f}.\n"
      "  The leaderboard has followed about half of a validation gain (sweep 10a: +0.0007 -> +0.00033), but validation\n"
      "  training targets the period shift itself, so it may transfer better. If the gain is >= +0.002: submit\n"
      "  zips/sweep11_routed.zip (fine-tuned on all of valid; no honest score is left for it, the leaderboard is the check).\n"
      "  Keep sweep11_hedge.zip as the safer final pick. c1 vs the fine-tunes says whether to rebuild teacher + students\n"
      "  on train + valid; sweep11_c3.zip is that idea's leaderboard probe.", flush=True)
ref = sc.get('c0_tr (ref)')
if ref:
    for nm in ('w05', 'w025', 'c1_trv', 'c2_v'):
        if nm in sc:
            print(f"  {nm}: all {sc[nm]['all']:.5f} vs c0_tr (same recipe, train only, no weights) {ref['all']:.5f} "
                  f"({sc[nm]['all'] - ref['all']:+.5f})", flush=True)
sh(f"cd /content/wnn && python -m pipeline.speed {SUBS['routed']} {SUBS['hedge']} {OUT_DIR}/sweep9/s9_h128/submission")  # grader ~ 0.52 x this
sync_up()

# ======== Part 1 again: sweep 10b's routing (the group-4 question), last because it is not urgent
# ---- Phase B: routing over sweep 10a + 10b (per group, the best on half A), routed submission, zips, speed
ROUTED = f'{OUT_DIR}/sweep10b_routed/submission'
drive_writable()
if os.path.exists(f'{ROUTED}/routes.json'):
    print('sweep 10b routing already done', flush=True)
else:
    REFS = ' '.join(f'"{OUT_DIR}/sweep9_teachers/{n}/best.pt"' for n in   # shared models trained like g4y_* (5 passes, no teacher)
                    ['t9_h128_s0', 't9_h128_s1', 't9_h128_s2', 't9_h128_s3', 't9_h192_drop_s0', 't9_h192_drop_s1'])
    sh(f'set -o pipefail; cd /content/wnn && python -m pipeline.routed --shared {S9} --cands "{OUT_DIR}/sweep10a/*/best.pt" "{SWEEP}/*/best.pt" {REFS} --valid-split {SPLIT} '
       f'--export {ROUTED} 2>&1 | grep -vE "Warning|_generic_rnn|warnings.warn"')
    os.makedirs(f'{OUT_DIR}/zips', exist_ok=True)
    print(shutil.make_archive(f'{OUT_DIR}/zips/sweep10b_routed', 'zip', ROUTED), flush=True)
    sync_up()  # zips and the routed submission to Drive before the speed test
    print('decision: a group-4 option beats s9_h128 on BOTH A_g4 (0.4459) and B_g4 (0.4205) -> group 4 is improvable; '
          'submit sweep10b_routed.zip if "routed, detected groups" valid_B > 0.67892 (sweep 10a routed) and valid_A rose too', flush=True)
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
    write("connectome_sweep11", HEADER + PRE + MOUNT + body)
    write("connectome_sweep11_rclone", HEADER + RCLONE_HEADER + PRE + RCLONE + body)


if __name__ == "__main__":
    main()
