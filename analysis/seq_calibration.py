"""Can a sequence's best prediction scale/level be predicted from its own early context?
Per sequence: context = rows 0..K (feature means/stds, activity, the GRU's prediction mean/std).
Label = weighted least-squares slope/offset of target on GRU prediction over rows >= K (scored).
Fit ridge on validation half A, apply to half B, score rows >= K of half B.

    python -m analysis.seq_calibration [--ckpt ...] [--K 2000] [--seqs 600]
"""
import argparse

import numpy as np

from utils import WARMUP, weighted_pearson
from pipeline.data import SeqStore
from pipeline.fast_eval import load_ckpt, pick_device, predict_batch
from pipeline.mask_model import split_valid


def context(X, P, K):
    x = X[:, :K]
    d = np.abs(np.diff(x, axis=1)).mean(1)                                   # [S,112] activity
    p = P[:, WARMUP:K]
    return np.concatenate([x.mean(1), x.std(1), d, p.mean(1), p.std(1), np.abs(p).mean(1)], 1)


def collect(st, net, dev, ids):
    C, P, Y, M = [], [], [], []
    for i in range(0, len(ids), 50):
        X, y, m = st.load(ids[i:i + 50])
        p = predict_batch(net, X, dev)
        C.append(X); P.append(p); Y.append(y); M.append(m)
    return np.concatenate(C), np.concatenate(P), np.concatenate(Y), np.concatenate(M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/gru128_wmse/best.pt")
    ap.add_argument("--K", type=int, default=2000)
    ap.add_argument("--seqs", type=int, default=600)
    args = ap.parse_args()
    st, dev = SeqStore("datasets/valid.parquet"), pick_device()
    net = load_ckpt(args.ckpt).eval().to(dev)
    A, B = split_valid(st.n_seqs)
    K = args.K
    late = np.arange(20000) >= K
    out = {}
    for name, ids in (("A", A[:args.seqs]), ("B", B[:args.seqs])):
        X, P, Y, M = collect(st, net, dev, ids)
        ctx = context(X, P, K)
        lab = []
        for s in range(len(P)):
            row = []
            for j in range(2):
                sel = M[s] & late
                y, p = np.clip(Y[s, sel, j], -2, 2), P[s, sel, j]
                w = np.abs(y) + 1e-6
                b = np.polyfit(p, y, 1, w=np.sqrt(w)) if sel.sum() > 20 else np.array([1.0, 0.0])
                row += [b[0], b[1]]
            lab.append(row)
        out[name] = dict(ctx=ctx, lab=np.array(lab), P=P, Y=Y, M=M & late[None])
        del X

    mu, sd = out["A"]["ctx"].mean(0), out["A"]["ctx"].std(0) + 1e-6
    ZA, ZB = (out["A"]["ctx"] - mu) / sd, (out["B"]["ctx"] - mu) / sd
    LA = out["A"]["lab"]
    lab_mu = LA.mean(0)

    def wp(P, Y, M):
        return float(np.mean([weighted_pearson(Y[..., j][M], P[..., j][M]) for j in range(2)]))

    def apply(P, lab):
        Q = P.copy()
        for j in range(2):
            Q[..., j] = lab[:, 2 * j, None] * P[..., j] + lab[:, 2 * j + 1, None]
        return Q

    b = out["B"]
    print(f"half B, rows >= {K}: GRU as is {wp(b['P'], b['Y'], b['M']):.4f}   "
          f"oracle per-sequence slope+offset {wp(apply(b['P'], b['lab']), b['Y'], b['M']):.4f}")
    for lam in (1.0, 10.0, 100.0, 1000.0):
        Wt = np.linalg.solve(ZA.T @ ZA + lam * np.eye(ZA.shape[1]), ZA.T @ (LA - lab_mu))
        pred = ZB @ Wt + lab_mu
        r = [np.corrcoef(pred[:, k], b["lab"][:, k])[0, 1] for k in range(4)]
        print(f"  ridge {lam:6g}: predicted calibration -> WP {wp(apply(b['P'], pred), b['Y'], b['M']):.4f}"
              f"   (corr with true slope t0 {r[0]:+.2f}, offset t0 {r[1]:+.2f}, slope t1 {r[2]:+.2f}, offset t1 {r[3]:+.2f})")


if __name__ == "__main__":
    main()
