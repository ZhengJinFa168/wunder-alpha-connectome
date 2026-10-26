"""Can the scoring mask (is_scored) be predicted from causal inputs? Trains a gradient-boosted
classifier on some validation sequences and tests on others (split by sequence).

    python analysis/mask_classifier.py
"""
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

from pipeline.data import SeqStore

T0 = 100  # first row with a previous required row (deltas)
GROUPS = [slice(0, 11), slice(11, 22), slice(22, 33), slice(33, 44), slice(44, 48), slice(48, 52),
          slice(52, 63), slice(63, 74), slice(74, 85), slice(85, 96), slice(96, 100), slice(100, 104)]
NAMES = ([f"f{j}" for j in range(104)] + [f"a{j}" for j in range(8)] +
         ["d_i0bidp", "d_i0askp", "d_i0bidv", "d_i0askv", "d_i0tdp", "d_i0tdv",
          "d_i1bidp", "d_i1askp", "d_i1bidv", "d_i1askv", "d_i1tdp", "d_i1tdv"] + ["h20_a2", "h20_a3", "h20_a4"])


def features(X):
    """Causal per-row features for a batch of sequences X [B,T,112] -> [B,T-T0,127]."""
    x = X[:, T0:]
    d = np.abs(X[:, T0:] - X[:, T0 - 1:-1])
    dg = np.stack([d[..., g].mean(-1) for g in GROUPS], -1)
    v = X[:, T0 - 20:, 106:109]  # trailing 20-row mean of a2..a4 (rows t-19..t)
    c = np.cumsum(np.pad(v, ((0, 0), (1, 0), (0, 0))), 1)
    h = (c[:, 20:] - c[:, :-20])[:, :x.shape[1]] / 20
    return np.concatenate([x, dg, h], -1).astype(np.float32)


def collect(store, ids, keep, rng):
    F, Yv = [], []
    for i in range(0, len(ids), 25):
        X, _, M = store.load(ids[i:i + 25])
        f, m = features(X), M[:, T0:]
        sel = rng.random(m.shape) < keep
        F.append(f[sel]); Yv.append(m[sel])
    return np.concatenate(F), np.concatenate(Yv)


def main():
    rng = np.random.default_rng(0)
    st = SeqStore("datasets/valid.parquet")
    perm = rng.permutation(st.n_seqs)
    tr_ids, te_ids = list(perm[:200]), list(perm[200:260])
    t = time.time()
    Xtr, ytr = collect(st, tr_ids, 0.25, rng)
    Xte, yte = collect(st, te_ids, 1.0, rng)
    print(f"train rows {len(ytr):,} (200 seqs, 25% sample)  test rows {len(yte):,} (60 other seqs)  "
          f"load {time.time() - t:.0f}s  base rate {yte.mean():.3f}", flush=True)

    clf = HGB(max_iter=300, learning_rate=0.08, max_leaf_nodes=63, random_state=0).fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    print(f"ALL FEATURES      AUC {roc_auc_score(yte, p):.3f}   avg precision {average_precision_score(yte, p):.3f}"
          f"  (random: AUC 0.5, AP {yte.mean():.3f})", flush=True)
    for name, cols in [("current 112 only", list(range(112))), ("a0-a7 only", list(range(104, 112))),
                       ("a2,a3,a4 only", [106, 107, 108]), ("book/trade only", list(range(104)) + list(range(112, 124)))]:
        c = HGB(max_iter=200, learning_rate=0.08, max_leaf_nodes=63, random_state=0).fit(Xtr[:, cols], ytr)
        print(f"  {name:18s} AUC {roc_auc_score(yte, c.predict_proba(Xte[:, cols])[:, 1]):.3f}", flush=True)

    ev = rng.random(len(yte)) < 0.04
    pi = permutation_importance(clf, Xte[ev], yte[ev], scoring="roc_auc", n_repeats=2, random_state=0)
    top = np.argsort(-pi.importances_mean)[:12]
    print("top features (AUC drop when shuffled):", [(NAMES[j], round(float(pi.importances_mean[j]), 3)) for j in top])
    # how concentrated is the mask among the classifier's top-scored rows?
    for q in (0.1, 0.25):
        thr = np.quantile(p, 1 - q)
        print(f"  top {int(q * 100)}% of rows by classifier contain {yte[p >= thr].sum() / yte.sum():.1%} of scored rows")


if __name__ == "__main__":
    main()
