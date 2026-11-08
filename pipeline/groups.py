"""Instrument groups (tick sizes): labels from the targets, and fitting the input-based detector.

    python -m pipeline.groups --fit        # fit the detector on train, check it on valid, write it into features.py
    python -m pipeline.groups --check      # accuracy of the detector in features.py on valid

Every target of a sequence is a whole multiple of its instrument's tick (features.TICKS), so the smallest
non-zero |target| names the group (training and evaluation only; at inference features.detect / the "grp"
feature work from the inputs).
"""
import argparse
import os
import re

import numpy as np
import pyarrow.parquet as pq

from utils import FEATURE_COLUMNS

from . import features
from .features import DET_ROWS, TICKS

HERE = os.path.dirname(os.path.abspath(__file__))


# Joint (t0, t1) target states of group 4, in ticks, clipped to +-4 (4 ticks = 2.63 > CLIP, i.e. the metric's clip at
# +-2): all 40 pairs that occur in train (every 4th group-4 sequence, rows 99+, 2026-11-05), most frequent first.
# The two targets are tied: (0,-1) 29%, (-1,0) 29%, (-1,-1) 9.6% (rarely scored), (1,-3) 8.3%, (-3,1) 8.0%, ...
STATE_TICKS = {3: [(0, -1), (-1, 0), (-1, -1), (1, -3), (-3, 1), (-3, 2), (2, -3), (-3, 3), (3, -3), (-3, 4), (4, -3),
                   (0, -2), (-2, 0), (1, -4), (-4, 1), (2, -4), (-4, 2), (-4, 4), (-4, 3), (3, -4), (4, -4), (-2, -1),
                   (-1, -2), (-3, 0), (0, -3), (-2, -2), (-1, -3), (-3, -1), (0, -4), (-2, -3), (-4, 0), (-1, -4),
                   (-3, -2), (-2, -4), (-4, -4), (-4, -1), (-4, -3), (-4, -2), (-3, -3), (-3, -4)]}


def state_values(group):
    """[K, 2] target values (clipped like the metric) of a group's joint states (for GRUNet(states=...))."""
    v = np.clip(np.asarray(STATE_TICKS[group], np.float64) * TICKS[group], -2.0, 2.0)
    return np.round(v, 6).tolist()


def label(Y):
    """Y [B,T,2] (numpy) -> group index [B] (0..3)."""
    a = np.abs(Y[..., 0]).astype(np.float64)
    tick = np.where(a > 1e-6, a, np.inf).min(1)
    return np.abs(tick[:, None] - np.asarray(TICKS)[None]).argmin(1)


def warmup_stats(path, ids=None):
    """(mean, std) [N,112] of the first DET_ROWS rows, as the detector computes them, and the group labels [N]."""
    pf = pq.ParquetFile(path)
    ids = range(pf.num_row_groups) if ids is None else ids
    X, G = [], []
    for g in ids:
        t = pf.read_row_group(int(g), columns=list(FEATURE_COLUMNS) + ["t0", "t1"]).to_pandas().to_numpy(np.float32)
        X.append(t[:DET_ROWS, :112].copy())  # a copy, so the whole sequence can be freed
        G.append(label(t[None, :, 112:])[0])
    X = np.stack(X).astype(np.float64)
    x0 = X[:, :1]
    xs = X - x0
    m = xs.mean(1)
    std = np.sqrt(np.maximum((xs * xs).mean(1) - m * m, 0.0))
    return m + x0[:, 0], std, np.array(G)


def accuracy(mean, std, G):
    P = features._det_probs(mean, std)
    pred = P.argmax(1)
    cm = np.zeros((4, 4), int)
    np.add.at(cm, (G, pred), 1)
    return (pred == G).mean(), P.max(1), pred, cm


def fit(C=0.1):
    from sklearn.linear_model import LogisticRegression

    mt, st, gt = warmup_stats("datasets/train.parquet")
    Z = np.concatenate([mt, st], 1)
    mu, sc = Z.mean(0), Z.std(0)
    sc = np.where(sc > 1e-6, sc, 1.0)
    clf = LogisticRegression(C=C, max_iter=10000).fit(np.clip((Z - mu) / sc, -10, 10), gt)
    fmt = lambda a: np.array2string(np.asarray(a), precision=8, separator=", ", threshold=10 ** 6, max_line_width=116)
    block = ("# DETECTOR BEGIN\n"
             f"# fitted on {len(gt)} train sequences, C={C}\n"
             f"DET_MEAN = np.array({fmt(mu)})\nDET_SCALE = np.array({fmt(sc)})\n"
             f"DET_COEF = np.array({fmt(clf.coef_)})\nDET_INTERCEPT = np.array({fmt(clf.intercept_)})\n"
             "# DETECTOR END")
    path = os.path.join(HERE, "features.py")
    src = open(path).read()
    src = re.sub(r"# DETECTOR BEGIN.*?# DETECTOR END", lambda _: block, src, flags=re.S)
    open(path, "w").write(src)
    print(f"detector written to {path} ({len(gt)} train sequences, train groups {np.bincount(gt)})")
    exec(compile(block, "detector", "exec"), vars(features))  # use the new constants for the check below
    a, conf, _, _ = accuracy(mt, st, gt)
    print(f"train accuracy {a:.4f}")


def check():
    mv, sv, gv = warmup_stats("datasets/valid.parquet")
    a, conf, pred, cm = accuracy(mv, sv, gv)
    wrong = pred != gv
    print(f"valid accuracy {a:.4f} ({int(wrong.sum())} of {len(gv)} wrong); confidence of the wrong ones: "
          f"{np.round(np.sort(conf[wrong]), 3).tolist()}; correct ones below 0.99: {int((conf[~wrong] < 0.99).sum())}")
    print(cm)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--C", type=float, default=0.1)
    args = ap.parse_args()
    if args.fit:
        fit(args.C)
    if args.fit or args.check:
        check()
