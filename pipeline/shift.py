"""Importance weights for the train sequences: how much each one looks like the validation period.

    python -m pipeline.shift --valid-split split.json --gamma 0.25 0.5 --cap 5 --out-dir runs/cache/shift

Per sequence: mean, log std and log mean |row-to-row change| of each of the 112 inputs (inputs only). Per instrument
group, a logistic regression (balanced classes, L2) separates train sequences from validation half H (H = A, B). Every
train sequence gets an out-of-fold p (5 folds over train; half H always in the fit). Weight = (p / (1 - p))^gamma,
normalised to mean 1 within the group (so the group mix stays the train mix), capped at `cap`, renormalised.
w_A_g{gamma}.npy uses only half A (honest to score on half B) and w_B_g{gamma}.npy only half B. Honest AUC: out-of-fold train
sequences vs the other half's sequences, from a model fitted on all train + half H.
Outputs: w_{A,B}_g{gamma}.npy (float32 [n_train], 1 for sequences without stats), seqstats.npz (cached), report.json.
"""
import argparse
import json
import os
import time

import numpy as np

from .data import SeqStore
from .groups import label as group_label


def seq_stats(store, ids, threads_batch=64):
    """-> stats [len(ids), 336], group [len(ids)]."""
    S, G = np.zeros((len(ids), 336), np.float32), np.zeros(len(ids), int)
    t = time.time()
    for b0, (X, Y, _) in zip(range(0, len(ids), threads_batch), store.batches(list(ids), threads_batch)):
        n = len(X)
        S[b0:b0 + n, :112] = X.mean(1)
        S[b0:b0 + n, 112:224] = np.log(X.std(1) + 1e-6)
        S[b0:b0 + n, 224:] = np.log(np.abs(np.diff(X, axis=1)).mean(1) + 1e-6)
        G[b0:b0 + n] = group_label(Y)
        if (b0 // threads_batch) % 20 == 19:
            print(f"  {b0 + n}/{len(ids)} sequences, {time.time() - t:.0f}s", flush=True)
    return S, G


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser()
    ap.add_argument("--valid-split", required=True)
    ap.add_argument("--gamma", type=float, nargs="+", default=[0.5])
    ap.add_argument("--cap", type=float, default=5.0)
    ap.add_argument("--C", type=float, default=0.05, help="inverse L2 strength")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-limit", type=int, default=None, help="tests: stats for the first N train sequences only")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    split = json.load(open(args.valid_split))
    train, valid = SeqStore("datasets/train.parquet", args.threads), SeqStore("datasets/valid.parquet", args.threads)
    tr_ids = np.arange(args.train_limit or train.n_seqs)
    va_ids = np.sort(np.asarray(split["A"] + split["B"]))
    cache = os.path.join(args.out_dir, "seqstats.npz")
    if os.path.exists(cache) and len(np.load(cache)["tr_ids"]) == len(tr_ids):
        z = np.load(cache)
        St, Gt, Sv, Gv = z["St"], z["Gt"], z["Sv"], z["Gv"]
        print(f"sequence statistics: cached ({cache})", flush=True)
    else:
        print(f"sequence statistics: {len(tr_ids)} train + {len(va_ids)} validation sequences", flush=True)
        St, Gt = seq_stats(train, tr_ids)
        Sv, Gv = seq_stats(valid, va_ids)
        np.savez(cache, St=St, Gt=Gt, Sv=Sv, Gv=Gv, tr_ids=tr_ids, va_ids=va_ids)
    half = {v: h for h in "AB" for v in split[h]}
    hv = np.array([half[v] for v in va_ids])
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 5, len(tr_ids))

    def fit(Xt, Xv):
        X = np.concatenate([Xt, Xv])
        y = np.r_[np.zeros(len(Xt)), np.ones(len(Xv))]
        sc = StandardScaler().fit(X)
        m = LogisticRegression(C=args.C, class_weight="balanced", max_iter=3000).fit(sc.transform(X), y)
        return lambda Z: m.predict_proba(sc.transform(Z))[:, 1]

    report = {}
    for H, other in (("A", "B"), ("B", "A")):
        w = {gm: np.ones(train.n_seqs, np.float32) for gm in args.gamma}
        report[H] = {}
        for g in range(4):
            it, iv, io = np.flatnonzero(Gt == g), np.flatnonzero((Gv == g) & (hv == H)), np.flatnonzero((Gv == g) & (hv == other))
            if len(it) < 10 or len(iv) < 5:
                print(f"half {H} group {g + 1}: too few sequences ({len(it)} train, {len(iv)} valid), weights 1", flush=True)
                continue
            p = np.zeros(len(it))
            for k in range(5):  # out-of-fold p for the train sequences
                tr_k, te_k = it[fold[it] != k], it[fold[it] == k]
                if len(te_k):
                    p[fold[it] == k] = fit(St[tr_k], Sv[iv])(St[te_k])
            auc = float("nan")
            if len(io):
                po = fit(St[it], Sv[iv])(Sv[io])
                auc = roc_auc_score(np.r_[np.zeros(len(p)), np.ones(len(po))], np.r_[p, po])
            p = np.clip(p, 1e-6, 1 - 1e-6)
            report[H][f"g{g + 1}"] = {"auc_other_half": auc, "n_train": len(it), "n_valid": len(iv)}
            for gm in args.gamma:
                wg = (p / (1 - p)) ** gm
                for _ in range(3):  # normalise, cap, repeat
                    wg = np.minimum(wg / wg.mean(), args.cap)
                wg = wg / wg.mean()
                w[gm][tr_ids[it]] = wg
                ess = wg.sum() ** 2 / (wg ** 2).sum() / len(wg)
                q = np.quantile(wg, [0.1, 0.5, 0.9])
                report[H][f"g{g + 1}"][f"gamma {gm:g}"] = {"ess": ess, "w_q10_50_90": q.tolist()}
                print(f"half {H} group {g + 1}: AUC train vs half {other} {auc:.3f}; gamma {gm:g}: weights q10/50/90 "
                      f"{q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}, max {wg.max():.2f}, effective sample {ess:.2f} of {len(it)}", flush=True)
        for gm in args.gamma:
            np.save(os.path.join(args.out_dir, f"w_{H}_g{gm:g}.npy"), w[gm])
    json.dump({"gamma": args.gamma, "cap": args.cap, "C": args.C, **report},
              open(os.path.join(args.out_dir, "report.json"), "w"), indent=1)
    print(f"weights -> {args.out_dir}/w_{{A,B}}_g{{{','.join(f'{g:g}' for g in args.gamma)}}}.npy", flush=True)


if __name__ == "__main__":
    main()
