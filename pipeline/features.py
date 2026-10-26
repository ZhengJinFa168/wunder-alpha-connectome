"""Extra causal input features, computed identically offline (training, whole sequences) and
online (inference, one row at a time):

* activity of each instrument's prices / volumes / trades: mean |x_t - x_{t-1}| over the group
  (0 on the first row of a sequence), now and as EMAs;
* EMAs of the extra features a2..a5 (a3 = the strongest scoring-mask signal).

EMA_k(t) = EMA_k(t-1) + (x_t - EMA_k(t-1)) / span_k, started at x_0. All values standardised with
fixed constants (MEAN / STD, measured once on training data). Width: N_EXTRA = 36.
This file is copied next to solution.py as wnn_features.py, so it must only need numpy at inference.
"""
import numpy as np

GROUPS = [(0, 22), (22, 44), (44, 52), (52, 74), (74, 96), (96, 104)]  # i0 p, i0 v, i0 trades, i1 p, i1 v, i1 trades
LEVEL_COLS = [106, 107, 108, 109]                                      # a2, a3, a4, a5
SPANS = [20, 200, 2000]
N_EXTRA = len(GROUPS) * (1 + len(SPANS)) + len(LEVEL_COLS) * len(SPANS)  # 36

# standardisation constants: `python -m pipeline.features` on 256 training sequences
MEAN = [
    0.02899, 0.080176, 0.344639, 0.013477, 0.046691, 0.409072,
    0.028905, 0.079919, 0.344147, 0.013465, 0.046752, 0.408771,
    0.028743, 0.079457, 0.341971, 0.013384, 0.046461, 0.406134,
    0.026218, 0.07248, 0.31095, 0.012185, 0.04226, 0.36902,
    -0.197401, -0.235902, 0.050653, -0.37099, -0.197616, -0.236261,
    0.050583, -0.3714, -0.199425, -0.240088, 0.051767, -0.375679]
STD = [
    0.061435, 0.144017, 0.647244, 0.045508, 0.138009, 0.751662,
    0.035306, 0.083746, 0.311466, 0.019796, 0.072021, 0.43189,
    0.028954, 0.070935, 0.235276, 0.013908, 0.058679, 0.335773,
    0.02191, 0.058177, 0.200191, 0.010707, 0.048553, 0.27719,
    0.911831, 0.966895, 0.3914, 1.13453, 0.729865, 0.841434,
    0.244381, 0.879383, 0.598998, 0.688904, 0.264152, 0.809816]


def _raw_offline(X):
    """X [B,T,112] float32 -> unstandardised extras [B,T,36] float64 (whole sequences, exact EMA)."""
    from scipy.signal import lfilter

    X = X.astype(np.float64)
    d = np.abs(np.diff(X, axis=1, prepend=X[:, :1]))
    act = np.stack([d[..., a:b].mean(-1) for a, b in GROUPS], -1)      # [B,T,6]
    lev = X[..., LEVEL_COLS]                                           # [B,T,4]
    out = [act]
    for v in (act, lev):
        for span in SPANS:
            a = 1.0 / span
            zi = (1 - a) * v[:, :1]                                     # [B,1,F]: so that EMA(0) = x_0
            y, _ = lfilter([a], [1, -(1 - a)], v, axis=1, zi=zi)
            out.append(y)
    # order: act, act_ema20, act_ema200, act_ema2000, lev_ema20, lev_ema200, lev_ema2000
    return np.concatenate(out, -1)


def offline(X, chunk=16):
    """Standardised extras [B,T,36] float32 for whole sequences. Computed `chunk` sequences at a time:
    the float64 intermediates of a whole 256-sequence batch would need ~10 GB of RAM."""
    out = np.empty(X.shape[:2] + (N_EXTRA,), np.float32)
    for i in range(0, len(X), chunk):
        out[i:i + chunk] = (_raw_offline(X[i:i + chunk]) - MEAN) / STD
    return out


class Online:
    """Row-by-row version of `offline`; reset() at every new sequence. Kept to a handful of numpy
    calls per row: one matrix product for all group activities, one array for all EMAs."""

    def __init__(self):
        n_g, n_l = len(GROUPS), len(LEVEL_COLS)
        self.G = np.zeros((112, n_g))                                   # |dx| @ G = per-group mean
        for k, (a, b) in enumerate(GROUPS):
            self.G[a:b, k] = 1.0 / (b - a)
        self.alpha = np.array([1.0 / s for s in SPANS])[:, None]       # [3,1]
        self.mean = np.asarray(MEAN, dtype=np.float64)
        self.inv_std = 1.0 / np.asarray(STD, dtype=np.float64)
        self.n_g = n_g
        self.raw = np.empty(N_EXTRA)
        self.lev_idx = np.array(LEVEL_COLS)
        self.reset()

    def reset(self):
        self.prev = None

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        n_g = self.n_g
        if self.prev is None:
            act = np.zeros(n_g)
            self.ema = np.empty((len(SPANS), n_g + len(LEVEL_COLS)))     # [3, 6 activities + 4 levels]
            self.ema[:, :n_g] = 0.0
            self.ema[:, n_g:] = x[self.lev_idx]
        else:
            act = np.abs(x - self.prev) @ self.G
            v = np.concatenate((act, x[self.lev_idx]))
            self.ema += (v - self.ema) * self.alpha
        self.prev = x
        raw = self.raw
        raw[:n_g] = act
        raw[n_g:n_g * 4] = self.ema[:, :n_g].ravel()
        raw[n_g * 4:] = self.ema[:, n_g:].ravel()
        return ((raw - self.mean) * self.inv_std).astype(np.float32)


def compute_stats(n_seqs=256, data="datasets/train.parquet"):
    from .data import SeqStore

    st = SeqStore(data)
    ids = list(range(0, st.n_seqs, st.n_seqs // n_seqs))[:n_seqs]
    R = []
    for i in range(0, len(ids), 32):
        X, _, _ = st.load(ids[i:i + 32])
        R.append(_raw_offline(X)[:, 99:].reshape(-1, N_EXTRA)[::7])
    R = np.concatenate(R)
    return R.mean(0), R.std(0) + 1e-6


if __name__ == "__main__":
    m, s = compute_stats()
    np.set_printoptions(precision=6, suppress=True, linewidth=150)
    print("MEAN = " + repr([round(float(v), 6) for v in m]))
    print("STD = " + repr([round(float(v), 6) for v in s]))
