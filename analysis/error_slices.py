"""Where does the model score well or badly? Weighted Pearson (metric conventions) on scored rows,
sliced by position, sequence volatility, target size, prediction size, a3 regime, i1 activity;
plus calibration and the between-sequence (level) part of the global metric.

    python -m analysis.error_slices [--ckpt runs/gru128_wmse/best.pt] [--seqs 300]
"""
import argparse
import json

import numpy as np
import torch

from utils import WARMUP, weighted_pearson
from pipeline.data import SeqStore
from pipeline.fast_eval import load_ckpt, pick_device, with_extras
from pipeline.model import needs_extra


@torch.no_grad()
def predict(net, X, dev, reset_every=None):
    if needs_extra(net):  # models with the extra causal features need them appended
        X = with_extras(X)
    out = np.empty((len(X), X.shape[1], 2), np.float32)
    h, step = None, reset_every or 2000
    for s in range(0, X.shape[1], step):
        if reset_every:
            h = None
        p, h = net(torch.from_numpy(X[:, s:s + step]).to(dev), h)
        out[:, s:s + step] = p.cpu().numpy()
    return out


def wp(P, Y, mask):
    return float(np.mean([weighted_pearson(Y[..., j][mask], P[..., j][mask]) for j in range(2)]))


def by_bins(name, values, P, Y, M, n=5, per_seq=False):
    """WP on scored rows within quantile bins of `values` ([S] per sequence or [S,T] per row)."""
    v = values if not per_seq else np.broadcast_to(values[:, None], M.shape)
    edges = np.quantile(v[M], np.linspace(0, 1, n + 1))
    print(f"\n{name}")
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        mk = M & (v >= lo) & ((v <= hi) if i == n - 1 else (v < hi))
        print(f"  {lo:+8.3f} .. {hi:+8.3f}: WP {wp(P, Y, mk):.4f}   ({mk.sum():,} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/gru128_wmse/best.pt")
    ap.add_argument("--seqs", type=int, default=300)
    ap.add_argument("--valid-split", help="split.json: sample only validation half B")
    args = ap.parse_args()
    dev = pick_device()
    net = load_ckpt(args.ckpt).eval().to(dev)
    st = SeqStore("datasets/valid.parquet")
    pool = json.load(open(args.valid_split))["B"] if args.valid_split else list(range(st.n_seqs))
    ids = pool[::max(1, len(pool) // args.seqs)][:args.seqs]
    Xs, Ys, Ms, P, P10k = [], [], [], [], []
    for i in range(0, len(ids), 50):
        X, Y, M = st.load(ids[i:i + 50])
        P.append(predict(net, X, dev)); P10k.append(predict(net, X, dev, 10000))
        Xs.append(X[..., [107, 96, 97, 98, 99, 100, 101, 102, 103]].copy()); Ys.append(Y); Ms.append(M)
    F, Y, M, P, P10k = (np.concatenate(a) for a in (Xs, Ys, Ms, P, P10k))
    print(f"{len(ids)} validation sequences, {M.sum():,} scored rows")
    print(f"WP scored: normal {wp(P, Y, M):.4f}   reset every 10,000 steps {wp(P10k, Y, M):.4f}")

    # per-sequence properties
    need = np.arange(Y.shape[1]) >= WARMUP
    vol = Y[:, need, 0].std(1)
    frac = M[:, need].mean(1)
    by_bins("Sequence volatility (std of t0 over the sequence)", vol, P, Y, M, per_seq=True)
    by_bins("Share of the sequence's rows that are scored", frac, P, Y, M, per_seq=True)
    by_bins("Target size |t0| (row)", np.abs(Y[..., 0]), P, Y, M)
    by_bins("Prediction size |p0| (row): is the model right when it is confident?", np.abs(P[..., 0]), P, Y, M)
    by_bins("a3 level (row)", F[..., 0], P, Y, M)
    act = np.abs(np.diff(F[..., 1:], axis=1, prepend=F[:, :1, 1:])).mean(-1)
    by_bins("i1 trade activity (|change| of i1 trade features, row)", act, P, Y, M)

    # calibration: mean target per prediction decile (scored rows, t0)
    p0, y0 = P[..., 0][M], np.clip(Y[..., 0][M], -2, 2)
    q = np.quantile(p0, np.linspace(0, 1, 11))
    print("\nCalibration, t0 (scored rows): mean prediction -> mean target, per prediction decile")
    print("  " + "  ".join(f"{p0[(p0 >= a) & (p0 <= b)].mean():+.2f}->{y0[(p0 >= a) & (p0 <= b)].mean():+.2f}"
                           for a, b in zip(q[:-1], q[1:])))
    slope = np.polyfit(p0, y0, 1)[0]
    print(f"  least-squares slope target~prediction: {slope:.2f} (1.0 = well scaled)")

    # global metric = within-sequence + between-sequence parts
    for j, name in enumerate(("t0", "t1")):
        ms = np.array([Y[s, M[s], j].mean() for s in range(len(Y))])
        mp = np.array([P[s, M[s], j].mean() for s in range(len(Y))])
        Yd, Pd = Y.copy(), P.copy()
        for s in range(len(Y)):
            Yd[s, :, j] -= ms[s]; Pd[s, :, j] -= mp[s]
        within = float(weighted_pearson(Yd[..., j][M], Pd[..., j][M]))
        print(f"\n{name}: corr of per-sequence MEAN prediction vs mean target {np.corrcoef(mp, ms)[0, 1]:.3f}"
              f" (std of means: target {ms.std():.3f}, prediction {mp.std():.3f});"
              f" WP within sequences (means removed) {within:.4f}")
    per_seq = np.array([wp(P[s:s + 1], Y[s:s + 1], M[s:s + 1]) for s in range(len(Y))])
    print(f"\nPer-sequence WP: median {np.median(per_seq):.3f}, 10th pct {np.percentile(per_seq, 10):.3f}, "
          f"worst {per_seq.min():.3f}, share below 0.4: {(per_seq < 0.4).mean():.1%}")
    print(f"corr(per-seq WP, volatility) {np.corrcoef(per_seq, vol)[0, 1]:+.3f}; "
          f"corr(per-seq WP, scored share) {np.corrcoef(per_seq, frac)[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
