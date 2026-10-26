"""Generic stateful ONNX step runner. Copied as solution.py next to model.onnx by the pipeline.
Works for single models (input "h") and ensembles (inputs "h_0", "h_1", ...)."""
from pathlib import Path
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import onnxruntime as ort


class PredictionModel:
    def __init__(self):
        o = ort.SessionOptions()
        o.intra_op_num_threads = 1
        o.inter_op_num_threads = 1
        o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        o.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(Path(__file__).resolve().parent / "model.onnx"),
                                            sess_options=o, providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        # hidden-state inputs, in graph order; outputs[1:] are their updates in the same order
        self.h_names = [i.name for i in inputs if i.name not in ("x", "x_prev")]
        self.h_init = [np.zeros(i.shape, dtype=np.float32) for i in inputs if i.name in self.h_names]
        self.use_prev = any(i.name == "x_prev" for i in inputs)  # dropped when no model uses diff features
        self.seq_ix = None

    def predict(self, data_point):
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self.h = [h.copy() for h in self.h_init]
            self.x_prev = None
        x = np.asarray(data_point.state, dtype=np.float32).reshape(1, 1, 112)
        if not np.isfinite(x).all():  # a NaN would otherwise stay in the hidden state for the whole sequence
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        feeds = dict(zip(self.h_names, self.h))
        feeds["x"] = x
        if self.use_prev:
            feeds["x_prev"] = x[0] if self.x_prev is None else self.x_prev
        out = self.session.run(None, feeds)
        self.h = out[1:]
        self.x_prev = x[0]
        if not data_point.need_prediction:
            return None
        pred = out[0].reshape(2)
        if not np.isfinite(pred).all():
            pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        return pred
