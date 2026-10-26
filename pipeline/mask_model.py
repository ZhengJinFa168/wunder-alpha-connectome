"""Scoring-mask model: predicts which rows the metric scores (is_scored) from causal inputs only,
and turns it into per-row training weights.

    python -m pipeline.mask_model fit   --out runs/mask_model          # train on validation half A, AUC on half B
    python -m pipeline.mask_model score --model runs/mask_model --out mask_train.npy   # p(scored) for train rows

Validation is split once into halves A and B (split.json). The classifier sees only A's masks, so
models can be compared on B without the mask information leaking into the comparison.
Output of `score`: float16 .npy [n_train_seqs, 20000], p(scored) per row (0 on warm-up rows),
read by the trainer with Common.row_weights and ExpConfig.mask_weight.
"""
import argparse
import json
import os
import time

import joblib
import numpy as np

from utils import SEQUENCE_LENGTH, WARMUP

from .data import SeqStore

GROUPS = [slice(0, 11), slice(11, 22), slice(22, 33), slice(33, 44), slice(44, 48), slice(48, 52),
          slice(52, 63), slice(63, 74), slice(74, 85), slice(85, 96), slice(96, 100), slice(100, 104)]


def _trailing_mean(v, w):
    """Mean over the last min(w, t+1) rows, for every row t. v [B,T,F] -> [B,T,F]."""
    c = np.cumsum(v, 1, dtype=np.float64)
    c = np.concatenate([np.zeros_like(c[:, :1]), c], 1)
    t = np.arange(v.shape[1])
    lo = np.maximum(0, t + 1 - w)
    return ((c[:, t + 1] - c[:, lo]) / (t + 1 - lo)[None, :, None]).astype(np.float32)


def features(X):
    """Causal per-row features for rows WARMUP..T-1. X [B,T,112] -> [B, T-WARMUP, 142]."""
    d = np.abs(np.diff(X, axis=1, prepend=X[:, :1]))                     # |x_t - x_{t-1}|, 0 at t=0
    dg = np.stack([d[..., g].mean(-1) for g in GROUPS], -1)               # activity per book/trade group
    a = X[..., 106:109]                                                    # a2, a3, a4
    parts = [X, dg, _trailing_mean(a, 20), _trailing_mean(a, 200), _trailing_mean(dg, 20)]
    return np.concatenate(parts, -1)[:, WARMUP:]


def split_valid(n_seqs, seed=5):
    perm = np.random.default_rng(seed).permutation(n_seqs)
    half = n_seqs // 2
    return sorted(int(i) for i in perm[:half]), sorted(int(i) for i in perm[half:])


def predict_rows(model, X, chunk=8):
    """p(scored) [B, T] for a batch of sequences (0 on warm-up rows)."""
    out = np.zeros(X.shape[:2], np.float32)
    for i in range(0, len(X), chunk):
        f = features(X[i:i + chunk])
        out[i:i + chunk, WARMUP:] = model.predict_proba(f.reshape(-1, f.shape[-1]))[:, 1].reshape(f.shape[:2])
    return out


def fit(args):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(0)
    st = SeqStore(os.path.join(args.data_dir, "valid.parquet"), args.threads)
    A, B = split_valid(st.n_seqs)
    os.makedirs(args.out, exist_ok=True)
    json.dump({"A": A, "B": B, "seed": 5}, open(os.path.join(args.out, "split.json"), "w"))

    t = time.time()
    F, Y = [], []
    for i in range(0, len(A), 32):  # sample rows of half A for training
        X, _, M = st.load(A[i:i + 32])
        f, m = features(X), M[:, WARMUP:]
        sel = rng.random(m.shape) < args.sample
        F.append(f[sel]); Y.append(m[sel])
    F, Y = np.concatenate(F), np.concatenate(Y)
    print(f"training rows {len(Y):,} from {len(A)} sequences (half A), load {time.time() - t:.0f}s", flush=True)

    t = time.time()
    model = HistGradientBoostingClassifier(max_iter=args.trees, learning_rate=0.12, max_leaf_nodes=63,
                                           early_stopping=False, random_state=0).fit(F, Y)
    print(f"fit {time.time() - t:.0f}s", flush=True)
    joblib.dump(model, os.path.join(args.out, "model.joblib"))

    # honest check on half B (sequences the classifier never saw)
    P, M = [], []
    for i in range(0, min(len(B), args.eval_seqs), 32):
        X, _, m = st.load(B[i:i + 32])
        P.append(predict_rows(model, X)[:, WARMUP:].ravel()); M.append(m[:, WARMUP:].ravel())
    P, M = np.concatenate(P), np.concatenate(M)
    top10 = M[P >= np.quantile(P, 0.9)].sum() / M.sum()
    rep = {"auc_B": float(roc_auc_score(M, P)), "avg_precision_B": float(average_precision_score(M, P)),
           "base_rate_B": float(M.mean()), "top10pct_share_of_scored": float(top10), "eval_seqs": int(min(len(B), args.eval_seqs))}
    json.dump(rep, open(os.path.join(args.out, "report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


def score(args):
    model = joblib.load(os.path.join(args.model, "model.joblib"))
    st = SeqStore(os.path.join(args.data_dir, "train.parquet"), args.threads)
    n = args.limit or st.n_seqs
    out = np.lib.format.open_memmap(args.out, mode="w+", dtype=np.float16, shape=(st.n_seqs, SEQUENCE_LENGTH))
    if args.limit:
        out[n:] = 0
    t, done = time.time(), 0
    for X, _, _ in st.batches(list(range(n)), 64):
        out[done:done + len(X)] = predict_rows(model, X).astype(np.float16)
        done += len(X)
        if done % 640 == 0 or done == n:
            print(f"{done}/{n} sequences, {time.time() - t:.0f}s", flush=True)
    out.flush()
    print(f"mean p(scored) on required rows: {float(np.asarray(out[:n:50, WARMUP:], np.float32).mean()):.4f} -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "score"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", help="folder written by fit (for score)")
    ap.add_argument("--data-dir", default="datasets")
    ap.add_argument("--sample", type=float, default=0.1, help="fraction of half-A rows used for fitting")
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--eval-seqs", type=int, default=400)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N train sequences (tests)")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    fit(args) if args.cmd == "fit" else score(args)


if __name__ == "__main__":
    main()
