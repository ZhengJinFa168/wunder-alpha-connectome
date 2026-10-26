"""Time submission folders row by row on one CPU thread, next to the official baseline.

    python -m pipeline.speed runs/NAME/submission runs/ens/submission

The test set is 39.4M rows in 60 min (~91 us/row on the grader's CPU). The baseline is known to
fit, so the ratio to the baseline is the number to watch; absolute times depend on this machine.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import importlib.util
import sys
import time

import numpy as np
import pyarrow.parquet as pq

from utils import FEATURE_COLUMNS, WARMUP, DataPoint

TEST_ROWS = 39_400_000


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
    args = ap.parse_args()

    tbl = pq.ParquetFile("datasets/valid.parquet").read_row_group(0, columns=list(FEATURE_COLUMNS))
    X = np.column_stack([tbl[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)[:args.rows]

    results = []
    for i, folder in enumerate(["baseline"] + args.folders):
        model = load_model(folder, i)
        t = time.perf_counter()
        for step in range(len(X)):
            model.predict(DataPoint(0, step, step >= WARMUP, X[step]))
        us = (time.perf_counter() - t) / len(X) * 1e6
        results.append((folder, us))
        base = results[0][1]
        print(f"{folder:50s} {us:7.1f} us/row  {us / base:5.2f}x baseline  "
              f"~{us * TEST_ROWS / 60e6:5.1f} min for the test set here", flush=True)


if __name__ == "__main__":
    main()
