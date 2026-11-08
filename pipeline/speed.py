"""Time submission folders row by row on one CPU thread, next to the official baseline.

    python -m pipeline.speed runs/NAME/submission runs/ens/submission
    python -m pipeline.speed --per-group --json speed.json runs/NAME/submission   # one sequence of each group (routed
                                                                                  # submissions run a model per group)

The test set is 39.4M rows in 60 min (~91 us/row on the grader's CPU). The baseline is known to
fit, so the ratio to the baseline is the number to watch; absolute times depend on this machine.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import importlib.util
import json
import sys
import time

import numpy as np
import pyarrow.parquet as pq

from utils import FEATURE_COLUMNS, WARMUP, DataPoint

TEST_ROWS = 39_400_000
GROUP_MIX = np.array([398, 347, 431, 697]) / 1873  # validation sequences per group (the test mix is unknown)


def load_model(folder, i):
    folder = os.path.abspath(folder)
    sys.path.insert(0, folder)
    spec = importlib.util.spec_from_file_location(f"solution_{i}", os.path.join(folder, "solution.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PredictionModel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--per-group", action="store_true", help="time the first validation sequence of each group")
    ap.add_argument("--json", help="write {folder: us/row (per-group: {g1..g4, mix})} here")
    args = ap.parse_args()

    pf = pq.ParquetFile("datasets/valid.parquet")
    seqs = [0]
    if args.per_group:
        from pipeline.groups import label
        found = {}
        for s in range(pf.num_row_groups):
            y = pf.read_row_group(s, columns=["t0", "t1"]).to_pandas().to_numpy()[None]
            found.setdefault(int(label(y)[0]), s)
            if len(found) == 4:
                break
        seqs = [found[g] for g in range(4)]
    Xs = []
    for s in seqs:
        tbl = pf.read_row_group(s, columns=list(FEATURE_COLUMNS))
        Xs.append(np.column_stack([tbl[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)[:args.rows])

    results, out = [], {}
    for i, folder in enumerate(["baseline"] + args.folders):
        model = load_model(folder, i)
        per = []
        for s, X in zip(seqs, Xs):
            t = time.perf_counter()
            for step in range(len(X)):
                model.predict(DataPoint(s, step, step >= WARMUP, X[step]))
            per.append((time.perf_counter() - t) / len(X) * 1e6)
        us = float(np.dot(per, GROUP_MIX)) if args.per_group else per[0]
        results.append((folder, us))
        base = results[0][1]
        extra = ("  per group " + " / ".join(f"{u:.1f}" for u in per)) if args.per_group else ""
        print(f"{folder:50s} {us:7.1f} us/row  {us / base:5.2f}x baseline  "
              f"~{us * TEST_ROWS / 60e6:5.1f} min for the test set here{extra}", flush=True)
        out[folder] = {**{f"g{g + 1}": u for g, u in enumerate(per)}, "mix": us} if args.per_group else us
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
