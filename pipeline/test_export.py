"""Check that exported ONNX step graphs + solution.py reproduce PyTorch predictions row by row.

    python -m pipeline.test_export            # random GRU / diff-GRU / LSTM / minGRU / TCN / mixed ensembles
"""
import importlib.util
import os
import shutil
import tempfile

import numpy as np
import pyarrow.parquet as pq
import torch

from utils import FEATURE_COLUMNS, WARMUP, DataPoint

from .fast_eval import predict_batch
from .features import N_EXTRA
from .model import EnsembleNet, GRUNet, MinGRUNet, TCNNet, export_onnx

HERE = os.path.dirname(os.path.abspath(__file__))


def run_rows(net, X):
    d = tempfile.mkdtemp()
    export_onnx(net, os.path.join(d, "model.onnx"))
    shutil.copy(os.path.join(HERE, "solution_template.py"), os.path.join(d, "solution.py"))
    shutil.copy(os.path.join(HERE, "features.py"), os.path.join(d, "wnn_features.py"))
    spec = importlib.util.spec_from_file_location("sol_test", os.path.join(d, "solution.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.PredictionModel()
    out = np.zeros((len(X), 2), np.float32)
    for seq in (0, 1):  # two sequences: also checks the state reset
        for t in range(len(X)):
            p = m.predict(DataPoint(seq, t, t >= WARMUP, X[t]))
            if p is not None:
                out[t] = p
    shutil.rmtree(d)
    return out


def main():
    torch.manual_seed(0)
    tbl = pq.ParquetFile("datasets/valid.parquet").read_row_group(0, columns=list(FEATURE_COLUMNS))
    X = np.column_stack([tbl[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)[:3000]
    a, b, c = GRUNet(96), GRUNet(64, use_diff=True), GRUNet(80, layers=3, cell="lstm")
    d, e = GRUNet(64, n_extra=N_EXTRA), GRUNet(48, use_diff=True, n_extra=N_EXTRA)
    # minGRU: 3000 rows cross several scan blocks and chunks; TCN 3**6 = 729-row receptive field crosses chunks
    f, g = MinGRUNet(64, layers=3), MinGRUNet(48, layers=2, use_diff=True, n_extra=N_EXTRA)
    t1, t2 = TCNNet(32, layers=6, kernel=3), TCNNet(24, layers=8, kernel=2, use_diff=True, n_extra=N_EXTRA)
    nets = {"gru": a, "gru_diff": b, "lstm_3layer": c, "gru_feats": d, "gru_diff_feats": e,
            "mingru": f, "mingru_diff_feats": g, "tcn_k3": t1, "tcn_k2_diff_feats": t2,
            "ensemble": EnsembleNet([a, b, c], np.array([[1.0, 1.1], [0.9, 1.2], [1.3, 0.7]])),
            "ensemble_feats": EnsembleNet([a, d, e], np.array([[1.0, 1.0], [0.8, 1.2], [1.1, 0.9]])),
            "ensemble_mixed": EnsembleNet([a, f, t1, t2], np.array([[1.0, 1.0], [0.8, 1.2], [1.1, 0.9], [0.9, 1.0]]))}
    for name, net in nets.items():
        net.eval()
        ref = predict_batch(net, X[None], torch.device("cpu"), chunk=700)[0]
        got = run_rows(net, X)
        err = np.abs(ref[WARMUP:] - got[WARMUP:]).max()
        print(f"{name:12s} max |torch - onnx| = {err:.2e}  {'OK' if err < 1e-4 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
