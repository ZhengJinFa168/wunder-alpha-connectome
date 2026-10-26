"""Linear readout on a frozen random EMA filter bank (the "pure NumPy baseline" someone posted):
s_t = (1 - a_j) s_{t-1} + a_j (x_t @ proj_j) for 1024 units with half-lives 1..3000 steps, then
weighted ridge (weights |clipped target|, like the metric) on [s_t, 1]. Ridge strength picked on
validation half A, reported on half B (scored rows), same protocol as the GRU sweeps.

    python -m analysis.reservoir_ridge [--units 1024] [--train-seqs 1000]
"""
import argparse
import json
import time

import numpy as np

from utils import WARMUP, weighted_pearson
from pipeline.data import SeqStore, split_train


def bank(X, proj, alphas):
    """X [B,T,112] -> states [B,T,U] (float32), exact causal EMAs started at 0."""
    Z = X @ proj                                                       # [B,T,U] float32
    S = np.empty_like(Z)
    s = np.zeros((Z.shape[0], Z.shape[2]), np.float32)
    a = alphas[None].astype(np.float32)
    for t in range(Z.shape[1]):                                         # 20k vector updates per batch
        s += a * (Z[:, t] - s)
        S[:, t] = s
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", type=int, default=1024)
    ap.add_argument("--train-seqs", type=int, default=1000)
    ap.add_argument("--every", type=int, default=4, help="use every k-th training row")
    ap.add_argument("--split", default=None, help="split.json (default: recomputed like mask_model)")
    args = ap.parse_args()
    rng = np.random.RandomState(0)
    alphas = 1.0 - np.exp(-np.log(2.0) / np.geomspace(1.0, 3000.0, args.units))
    proj = (rng.randn(112, args.units) / np.sqrt(112)).astype(np.float32)

    train, valid = SeqStore("datasets/train.parquet"), SeqStore("datasets/valid.parquet")
    tr_ids, _ = split_train(train.n_seqs, 500)
    tr_ids = list(np.random.default_rng(0).choice(tr_ids, args.train_seqs, replace=False))
    if args.split:
        sp = json.load(open(args.split)); A, B = sp["A"], sp["B"]
    else:
        from pipeline.mask_model import split_valid
        A, B = split_valid(valid.n_seqs)

    U = args.units + 1
    G = np.zeros((U, U)); r = np.zeros((U, 2))
    t0 = time.time()
    for i in range(0, len(tr_ids), 8):
        X, Y, _ = train.load(tr_ids[i:i + 8])
        S = bank(X, proj, alphas)[:, WARMUP::args.every].reshape(-1, args.units)
        y = np.clip(Y[:, WARMUP::args.every].reshape(-1, 2), -2, 2)
        F = np.c_[S, np.ones(len(S), np.float32)].astype(np.float64)
        w = np.abs(y).mean(1)                                           # one weight per row (both targets)
        G += F.T @ (F * w[:, None]); r += F.T @ (y * w[:, None])
        if i % 200 == 0:
            print(f"  {i + 8}/{len(tr_ids)} train seqs, {time.time() - t0:.0f}s", flush=True)

    def preds(ids, beta):
        P, Yall, Mall = [], [], []
        for i in range(0, len(ids), 8):
            X, Y, M = valid.load(ids[i:i + 8])
            S = bank(X, proj, alphas)
            P.append(S @ beta[:-1] + beta[-1]); Yall.append(Y); Mall.append(M)
        return np.concatenate(P), np.concatenate(Yall), np.concatenate(Mall)

    def wp(P, Y, M):
        return float(np.mean([weighted_pearson(Y[..., j][M], P[..., j][M]) for j in range(2)]))

    scale = np.trace(G) / U
    best = None
    for lam in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        beta = np.linalg.solve(G + lam * scale * np.eye(U), r)
        s = wp(*preds(A[:300], beta))
        print(f"  ridge {lam:g}: half A (300 seqs) WP {s:.4f}", flush=True)
        if best is None or s > best[0]:
            best = (s, lam, beta)
    P, Y, M = preds(B, best[2])
    print(f"\nreservoir + ridge ({args.units} units, {args.train_seqs} train seqs, ridge {best[1]:g}):")
    print(f"  half B scored WP {wp(P, Y, M):.4f}   all rows {wp(P, Y, np.broadcast_to(np.arange(20000) >= WARMUP, M.shape)):.4f}")
    np.save("runs/reservoir_B_preds.npy", P.astype(np.float16))
    print(f"total {time.time() - t0:.0f}s; half-B predictions saved to runs/reservoir_B_preds.npy")


if __name__ == "__main__":
    main()
