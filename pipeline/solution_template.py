"""Generic stateful ONNX step runner. Copied as solution.py next to model.onnx by the pipeline.
Works for single models and ensembles: every graph input other than x / x_prev is a recurrent
state, and graph outputs[1:] are their updates in the same order. The routed submission (solution_routed.py)
reuses this file as wnn_runner.py: one PredictionModel(model_file) per model, driven through reset() / step()."""
from pathlib import Path
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import onnxruntime as ort


class PredictionModel:
    def __init__(self, model_file="model.onnx"):
        o = ort.SessionOptions()
        o.intra_op_num_threads = 1
        o.inter_op_num_threads = 1
        o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        o.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(Path(__file__).resolve().parent / model_file),
                                            sess_options=o, providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        self.state_names = [i.name for i in inputs if i.name not in ("x", "x_prev")]
        self.state_init = [np.zeros(i.shape, dtype=np.float32) for i in inputs if i.name in self.state_names]
        self.use_prev = any(i.name == "x_prev" for i in inputs)  # absent when no model uses diff features
        self.out_names = [o.name for o in self.session.get_outputs()]
        x_width = next(i.shape[-1] for i in inputs if i.name == "x")
        self.features = None
        if x_width > 112:  # the model also takes the extra causal features (wnn_features.py, shipped alongside)
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import wnn_features
            kinds = getattr(wnn_features, "KINDS", {"act": wnn_features.N_EXTRA})
            kind = next((k for k, w in kinds.items() if 112 + w == x_width), None)
            assert kind, "wnn_features.py does not match model.onnx"
            self.features = wnn_features.Online(kind) if hasattr(wnn_features, "KINDS") else wnn_features.Online()
        self.seq_ix = None

    def reset(self):
        self.feeds = dict(zip(self.state_names, [s.copy() for s in self.state_init]))
        self.x_prev = None
        if self.features is not None:
            self.features.reset()

    def predict(self, data_point):
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self.reset()
        pred = self.step(data_point.state)
        if not data_point.need_prediction:
            return None
        return pred

    def step(self, row):
        """One row of the current sequence -> prediction [2]."""
        feeds = self.feeds
        state = np.asarray(row, dtype=np.float32)
        x = state.reshape(1, 1, 112)
        if self.use_prev:
            feeds["x_prev"] = x[0] if self.x_prev is None else self.x_prev
            self.x_prev = x[0]
        if self.features is not None:
            x = np.concatenate((state, self.features(state))).reshape(1, 1, -1)
        feeds["x"] = x
        out = self.session.run(self.out_names, feeds)
        for name, value in zip(self.state_names, out[1:]):
            feeds[name] = value
        pred = out[0].reshape(2)
        if not np.isfinite(pred).all():  # never seen; keep the submission valid and restart the state
            self.reset()
            pred = np.zeros(2, dtype=np.float32)
        return pred
