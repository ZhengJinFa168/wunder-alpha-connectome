"""Score a solution folder on valid.parquet with the official scorer from utils.py,
optionally on a subset of sequences for quick iteration.

    python evaluate.py my_solution --max-seqs 50
    python evaluate.py baseline                    # full validation set (slow)
"""
import argparse
import importlib.util
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from utils import (FEATURE_COLUMNS, SEQUENCE_LENGTH, TARGET_COLUMNS, DataPoint,
                   GlobalAccumulator, validate_sequence)


def load_model(solution_dir: str):
    solution_dir = os.path.abspath(solution_dir)
    sys.path.insert(0, solution_dir)
    spec = importlib.util.spec_from_file_location("solution", os.path.join(solution_dir, "solution.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PredictionModel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("solution_dir")
    ap.add_argument("--data", default=os.path.join(ROOT, "datasets", "valid.parquet"))
    ap.add_argument("--max-seqs", type=int, default=None)
    args = ap.parse_args()

    model = load_model(args.solution_dir)
    pf = pq.ParquetFile(args.data)
    n_groups = pf.num_row_groups if args.max_seqs is None else min(args.max_seqs, pf.num_row_groups)

    acc = GlobalAccumulator()
    t_model = 0.0
    for g in tqdm(range(n_groups), desc="sequences"):
        table = pf.read_row_group(g, use_threads=False)
        seq, need = validate_sequence(table)
        features = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        targets = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
        preds = np.full((SEQUENCE_LENGTH, 2), np.nan, dtype=np.float32)
        t0 = time.perf_counter()
        for step in range(SEQUENCE_LENGTH):
            value = model.predict(DataPoint(seq, step, bool(need[step]), features[step]))
            if not need[step]:
                if value is not None:
                    raise ValueError(f"expected None on warm-up step {step}")
            else:
                value = np.asarray(value, dtype=np.float32)
                if value.shape != (2,) or not np.isfinite(value).all():
                    raise ValueError(f"bad output at seq {seq} step {step}: {value}")
                preds[step] = value
        t_model += time.perf_counter() - t0
        acc.add(targets, preds, table["is_scored"].to_numpy().astype(bool) & need)

    res = acc.result()
    n_rows = n_groups * SEQUENCE_LENGTH
    us_row = t_model / n_rows * 1e6
    print(f"\nseqs={n_groups}  scored rows={res['selected_rows']:,}")
    print(f"WP t0={res['t0']:.6f}  t1={res['t1']:.6f}  score={res['weighted_pearson']:.6f}")
    print(f"model time {us_row:.1f} us/row -> ~{us_row * 1873 * SEQUENCE_LENGTH / 60e6:.0f} min "
          f"for a valid-sized test set on this machine (limit 60 min on 1 vCPU)")


if __name__ == "__main__":
    main()
