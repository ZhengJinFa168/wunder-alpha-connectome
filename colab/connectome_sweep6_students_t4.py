# Alpha Connectome, sweep 6 students on a T4 (sweep 6 phase 3, which stopped when the compute units ran out).
# Uses the cached teachers from Drive (t_w = mask-weighted 6-model teacher, t_big = sweep-4 teacher) and the pipeline
# code from connectome_sweep6.ipynb. T4 settings: no eval cache (RAM ~12.7 GB), 2 reader threads.
# Re-runnable: students whose results.json is already in Drive are skipped (group 1 finished on 2026-10-26).
# 7 students in 3 groups, in priority order; each group's checkpoints, ONNX submissions, results.json and upload-ready
# zips are saved to Drive as soon as it finishes (checkpoints also after every pass). Keep this tab open.
import os, subprocess, sys
from google.colab import drive
drive.mount('/content/drive')
OUT_DIR = '/content/drive/MyDrive/wunder-connectome/runs'
os.makedirs(OUT_DIR, exist_ok=True)

def sh(cmd):
    p = subprocess.Popen(cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end='', flush=True)
    assert p.wait() == 0, 'command failed: ' + cmd[:80]

sh('nvidia-smi --query-gpu=name,memory.total --format=csv; echo "CPUs: $(nproc)"; free -g | head -2')
DATA = r'''mkdir -p /content/wnn && cd /content/wnn
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
du -sh datasets'''
sh(DATA)
sh('pip -q install onnx onnxruntime')
os.chdir('/content/wnn')
# sweep 6's notebook carries the pipeline code; trainer.py, features.py and data.py are patched below
import json, os, glob
src = None
for cand in ['/content/drive/MyDrive/wunder_challenge/connectome_sweep6.ipynb'] + glob.glob('/content/drive/MyDrive/*/wunder_challenge/connectome_sweep6.ipynb'):
    if os.path.exists(cand): src = cand; break
assert src, 'connectome_sweep6.ipynb not found in Drive'
os.makedirs('/content/wnn/pipeline', exist_ok=True); os.makedirs('/content/wnn/analysis', exist_ok=True)
n = 0
for c in json.load(open(src))['cells']:
    s = ''.join(c['source'])
    if c['cell_type'] == 'code' and s.startswith('%%writefile /content/wnn/'):
        path, _, body = s.partition('\n')
        path = path.split(' ', 1)[1].strip()
        open(path, 'w').write(body if body.endswith('\n') or not body else body + '\n'); n += 1
print(n, 'files written from', src)
# patches on top of sweep 6's code: per-model mini-batches (trainer.py), and lower host-RAM peaks for the feature
# students on a 12.7 GB machine: features computed 16 sequences at a time, loader drops its reference to each batch,
# extras joined to the batch on the GPU
PATCHES = {'pipeline/trainer.py': [('    feats: bool = False         # also feed the 36 extra causal features (pipeline/features.py)\n', '    feats: bool = False         # also feed the 36 extra causal features (pipeline/features.py)\n    micro_batch: int | None = None  # sequences per optimizer step for this model (default: the whole data batch)\n'), ('        total_steps = self.epochs * n_batches * math.ceil(20000 / self.chunk)', '        self.micro = cfg.micro_batch or common.batch\n        total_steps = (self.epochs * n_batches * math.ceil(common.batch / self.micro)  # upper bound on steps\n                       * math.ceil(20000 / self.chunk))'), ('                h = xp = None\n                losses = []\n                for s in range(0, X.shape[1], e.chunk):\n                    x, y = Xe[:, s:s + e.chunk], Ye[:, s:s + e.chunk]\n                    p, h = e.net(x, h, xp)\n                    lo = max(0, WARMUP - s)  # skip warm-up rows\n                    if lo < x.shape[1]:\n                        pf, yf = p[:, lo:].reshape(-1, 2), y[:, lo:].reshape(-1, 2)\n                        rw = We[:, s:s + e.chunk][:, lo:].reshape(-1) if We is not None else None\n                        lam = e.cfg.distill\n                        loss = compute_loss(e.cfg.loss, pf, yf, e.cfg.alpha, rw) if lam < 1 else 0.0\n                        if lam > 0:\n                            tf = Te[:, s:s + e.chunk][:, lo:].reshape(-1, 2)\n                            loss = (1 - lam) * loss + lam * distill_loss(pf, tf, yf, e.cfg.distill_weight, rw)\n                        e.opt.zero_grad(set_to_none=True)\n                        loss.backward()\n                        torch.nn.utils.clip_grad_norm_(e.net.parameters(), common.clip_grad)\n                        e.opt.step()\n                        e.sched.step()\n                        e.update_ema()\n                        losses.append(loss.item())\n                    h = detach_state(h)\n                    xp = x[:, -1]\n                e.log(event="batch", epoch=epoch, batch=bi, loss=round(float(np.mean(losses)), 5),\n                      lr=round(e.sched.get_last_lr()[0], 6))', '                losses = []\n                for m0 in range(0, len(Xe), e.micro):  # mini-batches of whole sequences, each its own TBPTT pass\n                    Xm, Ym = Xe[m0:m0 + e.micro], Ye[m0:m0 + e.micro]\n                    Tm = Te[m0:m0 + e.micro] if Te is not None else None\n                    Wm = We[m0:m0 + e.micro] if We is not None else None\n                    h = xp = None\n                    for s in range(0, X.shape[1], e.chunk):\n                        x, y = Xm[:, s:s + e.chunk], Ym[:, s:s + e.chunk]\n                        p, h = e.net(x, h, xp)\n                        lo = max(0, WARMUP - s)  # skip warm-up rows\n                        if lo < x.shape[1]:\n                            pf, yf = p[:, lo:].reshape(-1, 2), y[:, lo:].reshape(-1, 2)\n                            rw = Wm[:, s:s + e.chunk][:, lo:].reshape(-1) if Wm is not None else None\n                            lam = e.cfg.distill\n                            loss = compute_loss(e.cfg.loss, pf, yf, e.cfg.alpha, rw) if lam < 1 else 0.0\n                            if lam > 0:\n                                tf = Tm[:, s:s + e.chunk][:, lo:].reshape(-1, 2)\n                                loss = (1 - lam) * loss + lam * distill_loss(pf, tf, yf, e.cfg.distill_weight, rw)\n                            e.opt.zero_grad(set_to_none=True)\n                            loss.backward()\n                            torch.nn.utils.clip_grad_norm_(e.net.parameters(), common.clip_grad)\n                            e.opt.step()\n                            e.sched.step()\n                            e.update_ema()\n                            losses.append(loss.detach())  # no GPU sync per step\n                        h = detach_state(h)\n                        xp = x[:, -1]\n                e.log(event="batch", epoch=epoch, batch=bi, loss=round(float(torch.stack(losses).mean()), 5),\n                      lr=round(e.sched.get_last_lr()[0], 6), steps=len(losses))'), ('            if any(e.cfg.feats for e in active):  # extras computed once per batch, shared by every model\n                X = np.concatenate([X, extra_features(X)], -1)\n            Xd = torch.from_numpy(X).to(device, non_blocking=True)  # one host->device copy per batch\n', '            if any(e.cfg.feats for e in active):  # extras computed once per batch, shared by every model, and\n                # joined on the GPU: the joined 256-seq batch would be another 3 GB array in host RAM\n                Xd = torch.cat([torch.from_numpy(X).to(device), torch.from_numpy(extra_features(X)).to(device)], -1)\n            else:\n                Xd = torch.from_numpy(X).to(device, non_blocking=True)  # one host->device copy per batch\n')], 'pipeline/features.py': [('def offline(X):\n    """Standardised extras [B,T,36] float32 for whole sequences."""\n    return ((_raw_offline(X) - MEAN) / STD).astype(np.float32)', 'def offline(X, chunk=16):\n    """Standardised extras [B,T,36] float32 for whole sequences. Computed `chunk` sequences at a time:\n    the float64 intermediates of a whole 256-sequence batch would need ~10 GB of RAM."""\n    out = np.empty(X.shape[:2] + (N_EXTRA,), np.float32)\n    for i in range(0, len(X), chunk):\n        out[i:i + chunk] = (_raw_offline(X[i:i + chunk]) - MEAN) / STD\n    return out')], 'pipeline/data.py': [('        for k in range(len(chunks)):\n            batch = fut.result()\n            if k + 1 < len(chunks):\n                fut = prefetch.submit(self.load, chunks[k + 1])\n            yield batch', '        for k in range(len(chunks)):\n            hold = [fut.result()]\n            fut = prefetch.submit(self.load, chunks[k + 1]) if k + 1 < len(chunks) else None\n            yield hold.pop()  # the generator keeps no reference, so the consumer can free the batch')]}
for f, pairs in PATCHES.items():
    t = open(f'/content/wnn/{f}').read()
    for a, b in pairs:
        if b in t: continue  # already applied
        assert t.count(a) == 1, f'{f} patch does not apply: ' + a[:60]
        t = t.replace(a, b)
    open(f'/content/wnn/{f}', 'w').write(t)
import importlib, sys; sys.path.insert(0, '/content/wnn')
for m in [m for m in sys.modules if m == 'pipeline' or m.startswith('pipeline.')]:
    del sys.modules[m]  # a previous run in this session may have loaded the old code
from pipeline.trainer import ExpConfig
assert 'micro_batch' in ExpConfig.__dataclass_fields__
print('patched:', ', '.join(PATCHES))

sh('cd /content/wnn && python -m pipeline.fast_eval --baseline --valid-seqs 100')
CACHE = f'{OUT_DIR}/cache'
ROWW, SPLIT = '/content/wnn/mask_train.npy', f'{OUT_DIR}/mask_model/split.json'
sh(f'cp {CACHE}/mask_train.npy {ROWW} && cp {CACHE}/t_w.npy /content/wnn/t_w.npy && cp {CACHE}/t_big.npy /content/wnn/t_big.npy && ls -la /content/wnn/*.npy')

def st(name, teacher='w', lam=0.5, **kw):
    return {'name': name, 'distill': 1.0, 'teacher': f'/content/wnn/t_{teacher}.npy', 'mask_weight': lam, **kw}

GROUPS = [
    [st('sw_h128'), st('sw_h128_s1', seed=1)],
    [st('sw_feat_h128_m75', lam=0.75, feats=True), st('sw_feat_h96', feats=True, hidden=96)],
    [st('sw_feat_h128', feats=True), st('sw_feat_h128_s1', feats=True, seed=1), st('sb_feat_h128', teacher='big', feats=True)],
]
SWEEP = f'{OUT_DIR}/sweep6_students'
os.makedirs(f'{OUT_DIR}/zips', exist_ok=True)
import shutil, time
import pandas as pd
t_all = time.time()
for g, group in enumerate(GROUPS, 1):
    done = [e['name'] for e in group if os.path.exists(f"{SWEEP}/{e['name']}/results.json")]
    group = [e for e in group if e['name'] not in done]
    print(f"\n===== group {g}/{len(GROUPS)}: {[e['name'] for e in group]}  ({(time.time() - t_all) / 3600:.1f} h elapsed)"
          + (f"  already done, skipped: {done}" if done else ''), flush=True)
    if not group:
        continue
    sh('free -g | head -2')  # each group trains in its own process, so the previous group's memory is already back
    json.dump(group, open('experiments.json', 'w'), indent=1)
    sh(f"""set -o pipefail; cd /content/wnn && python -m pipeline.multi_train --config experiments.json --epochs 10 --evals-per-epoch 1 --threads 2 --row-weights {ROWW} --valid-split {SPLIT} --out-dir {SWEEP} 2>&1 | grep -v '"event": "batch"'""")
    for e in group:  # upload-ready zip per finished student
        print(shutil.make_archive(f"{OUT_DIR}/zips/sweep6_{e['name']}", 'zip', f"{SWEEP}/{e['name']}/submission"), flush=True)
    rows = [json.loads(l) for l in open(f'{SWEEP}/summary.jsonl')]
    df = pd.DataFrame(rows).drop_duplicates('name', keep='last')
    print(df[['name', 'cost', 'valid_B_scored', 'valid_B_all', 'valid_scored', 'holdout_all']].sort_values('valid_B_scored', ascending=False).to_string(index=False), flush=True)
    print('references on half B: sweep 5 m50_e10 0.6769 (leaderboard 0.65444), teacher w 0.6786', flush=True)

rows = [json.loads(l) for l in open(f'{SWEEP}/summary.jsonl')]
df = pd.DataFrame(rows).drop_duplicates('name', keep='last')
top = df.sort_values('valid_B_scored', ascending=False).name.head(3).tolist()
sh('cd /content/wnn && python -m pipeline.speed ' + ' '.join(f'{SWEEP}/{n}/submission' for n in top))
print(f'all done in {(time.time() - t_all) / 3600:.1f} h')
