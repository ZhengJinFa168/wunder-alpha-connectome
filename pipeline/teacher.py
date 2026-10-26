"""Write a teacher's predictions on every training sequence, as the target for distillation.

    python -m pipeline.teacher --ensemble runs/ens/cost99/submission/ensemble.json --out teacher.npy
    python -m pipeline.teacher --ckpts runs/sweep2/s1_wmse_h128/best.pt runs/sweep2/c3_*/best.pt --out t_in2.npy
    python -m pipeline.teacher --ckpts "runs/sweep4/f*/best.pt" --oof --out t_oof.npy

Output: float16 .npy of shape [n_train_seqs, 20000, 2] (~850 MB), indexed by train.parquet row group,
read by the trainer (Common.teacher / ExpConfig.teacher).

In-sample (default): every member predicts every sequence, including the ones it trained on.
--oof: members were trained with ExpConfig.fold; sequence i is predicted only by the members whose
left-out fold is i % n_folds, so every training sequence gets honest (out-of-fold) predictions.
With --ckpts, members are rescaled to a common prediction spread (measured on the first batch).
"""
import argparse
import glob
import json
import time

import numpy as np
import torch

from .data import SeqStore
from .fast_eval import load_ckpt, pick_device, predict_batch, with_extras
from .model import EnsembleNet, needs_extra


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ensemble", help="ensemble.json written by pipeline.ensemble --export")
    src.add_argument("--ckpts", nargs="+", help="best.pt paths/globs, averaged")
    ap.add_argument("--oof", action="store_true", help="out-of-fold: use each member only on its left-out fold")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="datasets/train.parquet")
    ap.add_argument("--limit", type=int, default=None, help="only predict the first N sequences (tests)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = pick_device(args.device)

    if args.ensemble:
        spec = json.load(open(args.ensemble))
        paths, scales = spec["members"], np.asarray(spec["scales"])
    else:
        paths = [p for c in args.ckpts for p in sorted(glob.glob(c))]
        scales = None
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    nets = [load_ckpt(p).eval().to(device) for p in paths]
    folds = [ck.get("exp", {}).get("fold") for ck in cks]
    if args.oof:
        assert all(f is not None for f in folds), "--oof needs members trained with a fold"
        n_folds = cks[0].get("n_folds") or max(folds) + 1
    print(f"{len(nets)} members" + (f", folds {sorted(set(folds))}" if args.oof else ""), flush=True)

    store = SeqStore(args.data, args.threads)
    n = store.n_seqs
    out = np.lib.format.open_memmap(args.out, mode="w+", dtype=np.float16, shape=(n, 20000, 2))
    ids = list(range(args.limit or n))
    if args.limit:
        out[args.limit:] = 0
    t, done = time.time(), 0
    extra = any(needs_extra(n) for n in nets)
    for X, _, _ in store.batches(ids, 128):
        X = with_extras(X) if extra else X
        P = np.stack([predict_batch(net, X, device) for net in nets])  # [K, B, T, 2]
        if scales is None:  # common spread, fixed from the first batch
            sd = P[:, :, 99:].std(axis=(1, 2))  # [K, 2]
            scales = sd.mean(0, keepdims=True) / sd
        P = P * scales[:, None, None, :]
        if args.oof:
            batch_fold = np.arange(done, done + len(X)) % n_folds
            pred = np.zeros(P.shape[1:], np.float32)
            for f in range(n_folds):
                members = [k for k in range(len(nets)) if folds[k] == f]
                rows = batch_fold == f
                pred[rows] = P[members][:, rows].mean(0)
        else:
            pred = P.mean(0)
        out[done:done + len(X)] = pred.astype(np.float16)
        done += len(X)
        if done % 1280 == 0 or done == len(ids):
            print(f"{done}/{len(ids)} sequences, {time.time() - t:.0f}s", flush=True)
    out.flush()
    print(f"teacher ({len(nets)} models{', out-of-fold' if args.oof else ''}) -> {args.out}")


if __name__ == "__main__":
    main()
