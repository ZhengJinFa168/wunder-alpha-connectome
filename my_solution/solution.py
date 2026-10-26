"""Submission entry point. Everything in this folder gets zipped by make_submission.sh.

Starting point: a stateful EMA baseline so the pipeline works end to end.
Replace with your own model (e.g. a GRU exported to ONNX, like the official baseline).
"""
import os

import numpy as np

try:
    from utils import DataPoint  # provided by the scorer / starter pack
except ImportError:  # keeps the file importable on its own
    DataPoint = object

HERE = os.path.dirname(os.path.abspath(__file__))  # load weights relative to this file


class PredictionModel:
    def __init__(self):
        self.seq_ix = None
        self.alpha = 0.05
        self.ema = None

    def _reset(self):
        self.ema = None

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self._reset()

        # Update state on every row, including warm-up rows.
        x = data_point.state.astype(np.float32)
        self.ema = x if self.ema is None else self.ema + self.alpha * (x - self.ema)

        if not data_point.need_prediction:
            return None

        # Placeholder signal: i0 order-book volume imbalance (bid vols 22-32 vs ask vols 33-43).
        bid, ask = x[22:33].sum(), x[33:44].sum()
        imb = (bid - ask) / (abs(bid) + abs(ask) + 1e-6)
        pred = np.array([imb, imb], dtype=np.float32)
        return np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
