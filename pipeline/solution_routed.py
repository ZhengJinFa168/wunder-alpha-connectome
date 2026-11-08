"""Routed submission: one model per instrument group. Copied as solution.py by pipeline/routed.py, next to
routes.json, the model files, wnn_runner.py (= solution_template.py) and wnn_features.py.

During the warm-up rows (no prediction needed) the group detector (wnn_features "grp") sees the rows and they are
buffered. At the first row that needs a prediction the detector picks the group's model (or the fallback model when
it is less sure than routes.json's threshold), the buffered rows are replayed through that model only, and from then
on only that model runs."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import wnn_features
from wnn_runner import PredictionModel as Runner


class PredictionModel:
    def __init__(self):
        cfg = json.load(open(HERE / "routes.json"))
        self.runners = [Runner(str(HERE / f)) for f in cfg["models"]]  # absolute: wnn_runner may come from another folder
        self.route, self.fallback, self.threshold = cfg["route"], cfg["fallback"], cfg["threshold"]
        self.detector = wnn_features.Online("grp")
        self.seq_ix = None

    def predict(self, data_point):
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            for r in self.runners:
                r.reset()
            self.detector.reset()
            self.active, self.probs, self.buffer = None, None, []
        if self.active is None:
            self.probs = self.detector(data_point.state)
            if not data_point.need_prediction:
                self.buffer.append(np.array(data_point.state, dtype=np.float32))  # a copy: the caller may reuse its array
                return None
            g = int(np.argmax(self.probs))
            self.active = self.runners[self.route[g] if self.probs[g] >= self.threshold else self.fallback]
            for row in self.buffer:  # catch the chosen model up on the warm-up rows
                self.active.step(row)
            self.buffer = []
        pred = self.active.step(data_point.state)
        return pred if data_point.need_prediction else None
