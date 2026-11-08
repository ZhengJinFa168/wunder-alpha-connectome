"""Check that exported ONNX step graphs + solution.py reproduce PyTorch predictions row by row.

    python -m pipeline.test_export            # random GRU / diff-GRU / LSTM / minGRU / TCN / feature kinds / aux head /
                                              # group-conditioned GRU / ensembles / routed submission
"""
import importlib.util
import os
import shutil
import json
import tempfile

import numpy as np
import pyarrow.parquet as pq
import torch

from utils import FEATURE_COLUMNS, WARMUP, DataPoint

from .fast_eval import predict_batch
from .features import KINDS, N_EXTRA, detect
from .groups import state_values
from .model import EnsembleNet, GRUNet, GroupGRUNet, MinGRUNet, TCNNet, build_net, export_onnx

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


def run_routed(nets, route, fallback, threshold, X):
    """Routed submission (pipeline/solution_routed.py) row by row over two copies of X."""
    d = tempfile.mkdtemp()
    files = []
    for k, net in enumerate(nets):
        files.append(f"model_{k}.onnx")
        export_onnx(net, os.path.join(d, files[-1]))
    json.dump({"models": files, "route": route, "fallback": fallback, "threshold": threshold},
              open(os.path.join(d, "routes.json"), "w"))
    shutil.copy(os.path.join(HERE, "solution_routed.py"), os.path.join(d, "solution.py"))
    shutil.copy(os.path.join(HERE, "solution_template.py"), os.path.join(d, "wnn_runner.py"))
    shutil.copy(os.path.join(HERE, "features.py"), os.path.join(d, "wnn_features.py"))
    spec = importlib.util.spec_from_file_location("sol_routed_test", os.path.join(d, "solution.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.PredictionModel()
    out = np.zeros((len(X), 2), np.float32)
    for seq in (0, 1):
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
    n1, n2 = GRUNet(64, n_extra=KINDS["norm"], feat_kind="norm"), GRUNet(48, n_extra=KINDS["act+norm"], feat_kind="act+norm")
    n3 = GRUNet(56, use_diff=True, n_extra=KINDS["norm"], feat_kind="norm", n_aux=7)  # aux head: not in the export
    nets.update({"gru_norm": n1, "gru_act+norm": n2, "gru_diff_norm_aux": n3,
                 "ensemble_norm": EnsembleNet([a, n1, n3], np.array([[1.0, 1.0], [0.8, 1.2], [1.1, 0.9]]))})
    base = GRUNet(64)  # group-conditioned GRU: starts as exactly the plain GRU it is copied from
    gg = GroupGRUNet(64).from_gru(base.state_dict()).eval()
    same = np.abs(predict_batch(gg, X[None], torch.device("cpu"))[0] - predict_batch(base.eval(), X[None], torch.device("cpu"))[0]).max()
    print(f"ggru from_gru vs gru: max diff {same:.2e}  {'OK' if same < 1e-5 else 'MISMATCH'}")
    gg2 = GroupGRUNet(48, layers=3)
    with torch.no_grad():
        gg2.film_scale.normal_(0, 0.3), gg2.film_shift.normal_(0, 0.3)
    nets.update({"ggru": gg2, "ensemble_ggru": EnsembleNet([n1, gg2], np.array([[1.0, 1.0], [0.8, 1.2]]))})
    st = GRUNet(64, states=state_values(3))  # joint-state head (group 4): softmax over the states -> prediction
    with torch.no_grad():
        st.head.logits.weight.normal_(0, 0.3)
    st2 = build_net(st.config)  # rebuilt from its saved config, as fast_eval / routed load it
    st2.load_state_dict(st.state_dict())
    same = np.abs(predict_batch(st2.eval(), X[None], torch.device("cpu"))[0] - predict_batch(st.eval(), X[None], torch.device("cpu"))[0]).max()
    print(f"state head rebuilt from config: max diff {same:.2e}  {'OK' if same < 1e-6 else 'MISMATCH'}")
    nets.update({"gru_states": st, "ensemble_states": EnsembleNet([a, st], np.ones((2, 2)))})
    nets = {k: v for k, v in nets.items() if k != "ensemble_ggru"}  # members with different kinds: not exportable
    for name, net in nets.items():
        net.eval()
        ref = predict_batch(net, X[None], torch.device("cpu"), chunk=700)[0]
        got = run_rows(net, X)
        err = np.abs(ref[WARMUP:] - got[WARMUP:]).max()
        print(f"{name:18s} max |torch - onnx| = {err:.2e}  {'OK' if err < 1e-4 else 'MISMATCH'}")
    # routed submission: 3 models, the detector's group picks one; threshold above 1 -> always the fallback
    g = int(detect(X[None]).argmax())
    members = [a, n1, gg2, EnsembleNet([a, gg2], np.ones((2, 2))), st]  # last: a blend as routed.py exports it
    refs = [predict_batch(net.eval(), X[None], torch.device("cpu"))[0] for net in members]
    for route, fallback, thr in (([0, 1, 2, 4], 0, 0.0), ([4, 4, 4, 4], 0, 0.0), ([0, 1, 2, 1], 0, 0.0), ([3, 3, 3, 3], 1, 0.0), ([1, 2, 1, 2], 0, 1.01)):
        want = refs[fallback] if thr > 1 else refs[route[g]]
        got = run_routed(members, route, fallback, thr, X)
        err = np.abs(want[WARMUP:] - got[WARMUP:]).max()
        print(f"routed {route} fb {fallback} thr {thr}: group {g + 1}, max |torch - onnx| = {err:.2e}  "
              f"{'OK' if err < 1e-4 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
