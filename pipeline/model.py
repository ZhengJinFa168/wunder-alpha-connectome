"""Stateful sequence models (GRU/LSTM, minGRU, causal TCN), metric-aligned losses, ONNX step export and
baseline import. Every model has forward(x, h, x_prev) -> (pred, new_state) for chunked training/evaluation;
minGRU and TCN also have stream_step() for the one-row ONNX graph (see research/architectures_cnn_rnn_transformer.md)."""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

N_FEATURES = 112
CLIP = 2.0


def _model_inputs(x, x_prev, use_diff, n_extra):
    """The 112 features, then their row-to-row diffs (use_diff), then the extra causal features (n_extra)."""
    base = x[..., :N_FEATURES]
    parts = [base]
    if use_diff:
        first = base[:, :1] if x_prev is None else x_prev[:, None, :N_FEATURES]
        prev = torch.cat([first, base[:, :-1]], 1)
        parts.append(base - prev)
    if n_extra:
        assert x.shape[-1] >= N_FEATURES + n_extra, "model needs the extra features as input"
        parts.append(x[..., N_FEATURES:N_FEATURES + n_extra])
    return torch.cat(parts, -1) if len(parts) > 1 else base


class GRUNet(nn.Module):
    """Stacked GRU (or LSTM with cell="lstm") + linear head. The recurrent module keeps the
    attribute name `gru` for either cell so older checkpoints load unchanged."""

    def __init__(self, hidden=128, layers=2, use_diff=False, dropout=0.0, cell="gru", n_extra=0):
        super().__init__()
        self.config = dict(hidden=hidden, layers=layers, use_diff=use_diff, dropout=dropout, cell=cell, n_extra=n_extra)
        self.use_diff = use_diff
        self.cell = cell
        self.n_extra = n_extra  # extra causal features (pipeline/features.py), appended after the 112 inputs
        in_dim = N_FEATURES * (2 if use_diff else 1) + n_extra
        rnn = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.gru = rnn(in_dim, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x, h=None, x_prev=None):
        """x [B,T,112 (+ extras)]; h [L,B,H] or None; x_prev [B,>=112] = last row of the previous
        chunk (None at sequence start, where the first diff is zero). Input order to the GRU:
        the 112 features, then their diffs (use_diff), then the extras (n_extra)."""
        base = x[..., :N_FEATURES]
        parts = [base]
        if self.use_diff:
            first = base[:, :1] if x_prev is None else x_prev[:, None, :N_FEATURES]
            prev = torch.cat([first, base[:, :-1]], 1)
            parts.append(base - prev)
        if self.n_extra:
            assert x.shape[-1] >= N_FEATURES + self.n_extra, "model needs the extra features as input"
            parts.append(x[..., N_FEATURES:N_FEATURES + self.n_extra])
        out, h = self.gru(torch.cat(parts, -1) if len(parts) > 1 else base, h)
        return self.head(out), h


def _g(x):
    """minGRU candidate activation: x + 0.5 for x >= 0, sigmoid(x) below, so the candidate is always positive."""
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))


def _log_g(x):
    return torch.where(x >= 0, torch.log(F.relu(x) + 0.5), -F.softplus(-x))


class MinGRUNet(nn.Module):
    """Stack of minGRU layers (Feng et al. 2024, "Were RNNs All We Needed?") + linear head.

    Per layer: z_t = sigmoid(W_z u_t), h_t = (1 - z_t) * h_{t-1} + z_t * g(W_h u_t). Gate and candidate depend
    only on the layer input u_t (not on h_{t-1}), so training runs a parallel log-space scan over each chunk
    (no step-by-step loop; long chunks are cheap), while inference stays an O(1) update per row with ~1/3 of a GRU
    layer's multiply-adds. The nonlinear state interactions a GRU has come from stacking layers instead.
    Gate biases start at log-spaced memory lengths of 2..2000 rows (chrono-style), so some units remember long
    from the first step. State: one [B, H] tensor per layer."""
    BLOCK = 128  # scan block: keeps cumulative log-gates small enough for float32 (state is carried across blocks)

    def __init__(self, hidden=192, layers=3, use_diff=False, dropout=0.0, n_extra=0):
        super().__init__()
        self.config = dict(arch="mingru", hidden=hidden, layers=layers, use_diff=use_diff, dropout=dropout,
                           n_extra=n_extra)
        self.use_diff, self.n_extra, self.hidden = use_diff, n_extra, hidden
        in_dim = N_FEATURES * (2 if use_diff else 1) + n_extra
        self.proj = nn.ModuleList([nn.Linear(in_dim if l == 0 else hidden, 2 * hidden) for l in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 2)
        with torch.no_grad():  # chrono init: 1 / z = memory length tau, log-uniform in [2, 2000]
            for p in self.proj:
                tau = torch.exp(torch.empty(hidden).uniform_(math.log(2.0), math.log(2000.0)))
                p.bias[:hidden] = -torch.log(tau - 1)

    @classmethod
    def _scan(cls, k, c, h0):
        """h_t = a_t h_{t-1} + b_t with a = 1 - sigmoid(k), b = sigmoid(k) g(c), in log space per block:
        h_t = A_t h0 + sum_{j<=t} (A_t / A_j) b_j, A_t = prod_{i<=t} a_i. k, c [B,T,H]; h0 [B,H] or None (zeros)."""
        log_a = -F.softplus(k)
        log_b = -F.softplus(-k) + _log_g(c)
        outs = []
        for s in range(0, k.shape[1], cls.BLOCK):
            a_star = log_a[:, s:s + cls.BLOCK].cumsum(1)
            h = torch.exp(a_star + torch.logcumsumexp(log_b[:, s:s + cls.BLOCK] - a_star, 1))
            if h0 is not None:
                h = h + torch.exp(a_star) * h0[:, None]
            outs.append(h)
            h0 = h[:, -1]
        return torch.cat(outs, 1)

    def forward(self, x, h=None, x_prev=None):
        """x [B,T,112 (+ extras)]; h list of [B,H] per layer or None; x_prev as in GRUNet."""
        z = _model_inputs(x, x_prev, self.use_diff, self.n_extra)
        new = []
        for l, proj in enumerate(self.proj):
            k, c = proj(z).chunk(2, -1)
            z = self._scan(k, c, None if h is None else h[l])
            new.append(z[:, -1])
            if l < len(self.proj) - 1:
                z = self.drop(z)
        return self.head(z), new

    def stream_states(self):
        return [(f"h{l}", (1, self.hidden)) for l in range(len(self.proj))]

    def stream_step(self, z, states):
        """One row: z [1,1,in] (already built inputs), states list of [1,H] -> (pred [1,1,2], new states)."""
        z, new = z[:, 0], []
        for proj, h in zip(self.proj, states):
            k, c = proj(z).chunk(2, -1)
            z = h + torch.sigmoid(k) * (_g(c) - h)
            new.append(z)
        return self.head(z)[:, None], new


class TCNNet(nn.Module):
    """Causal dilated temporal convolution network (Bai et al. 2018, WaveNet-style) + linear head.

    Input projection to `hidden` channels, then `layers` pre-norm residual blocks
    z <- z + W_mix relu(conv_{kernel, dilation}(LayerNorm(z))), dilation = kernel**l, so the receptive field
    is kernel**layers rows (2**11 = 2048, 3**7 = 2187). Each output sees a fixed window of past rows; unlike an
    RNN nothing older can leak in. State between chunks: the last (kernel - 1) * dilation normalised inputs of
    each block ([B, pad_l, C]), zeros before the sequence start, so chunked training, evaluation and row-by-row
    inference see exactly the same context."""

    def __init__(self, hidden=64, layers=11, use_diff=False, dropout=0.0, n_extra=0, kernel=2):
        super().__init__()
        assert kernel >= 2, "TCN kernel must be >= 2"
        self.config = dict(arch="tcn", hidden=hidden, layers=layers, use_diff=use_diff, dropout=dropout,
                           n_extra=n_extra, kernel=kernel)
        self.use_diff, self.n_extra, self.hidden, self.kernel = use_diff, n_extra, hidden, kernel
        self.dilations = [kernel ** l for l in range(layers)]
        self.pads = [(kernel - 1) * d for d in self.dilations]
        in_dim = N_FEATURES * (2 if use_diff else 1) + n_extra
        self.inp = nn.Linear(in_dim, hidden)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.convs = nn.ModuleList([nn.Conv1d(hidden, hidden, kernel, dilation=d) for d in self.dilations])
        self.mix = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 2)

    @property
    def receptive_field(self):
        return 1 + sum(self.pads)

    def forward(self, x, h=None, x_prev=None):
        """x [B,T,112 (+ extras)]; h list of [B, pad_l, C] per block or None; x_prev as in GRUNet."""
        z = self.inp(_model_inputs(x, x_prev, self.use_diff, self.n_extra))
        B, new = z.shape[0], []
        for l, (norm, conv, mix, pad) in enumerate(zip(self.norms, self.convs, self.mix, self.pads)):
            u = norm(z)
            past = h[l] if h is not None else u.new_zeros(B, pad, u.shape[-1])
            full = torch.cat([past, u], 1)  # [B, pad + T, C]
            new.append(full[:, -pad:])
            y = conv(full.transpose(1, 2)).transpose(1, 2)  # [B, T, C]
            z = z + mix(self.drop(F.relu(y)))
        return self.head(self.out_norm(z)), new

    def stream_states(self):
        return [(f"b{l}", (1, pad, self.hidden)) for l, pad in enumerate(self.pads)]

    def stream_step(self, z, states):
        """One row: z [1,1,in], states list of [1, pad_l, C] -> (pred [1,1,2], new states). Buffers are shifted
        with Concat + Slice (simple, but copies every buffer once per row: see the notes on ring buffers)."""
        z, new = self.inp(z), []
        for norm, conv, mix, d, buf in zip(self.norms, self.convs, self.mix, self.dilations, states):
            full = torch.cat([buf, norm(z)], 1)  # [1, pad + 1, C]
            new.append(full[:, 1:])
            taps = full[:, ::d]  # the kernel's inputs: rows t - (k-1)d, ..., t - d, t
            y = F.conv1d(taps.transpose(1, 2), conv.weight, conv.bias).transpose(1, 2)  # [1, 1, C]
            z = z + mix(F.relu(y))
        return self.head(self.out_norm(z)), new


ARCHS = {"gru": GRUNet, "mingru": MinGRUNet, "tcn": TCNNet}


def build_net(config):
    """Model from a saved config. GRU/LSTM configs predate the `arch` key, so it defaults to gru."""
    cfg = dict(config)
    return ARCHS[cfg.pop("arch", "gru")](**cfg)


def needs_extra(net):
    members = list(net.nets) if isinstance(net, EnsembleNet) else [net]
    return max(getattr(m, "n_extra", 0) for m in members)


def detach_state(h):
    """Detach a recurrent state: tensor (GRU), (h, c) tuple (LSTM) or list of those (ensemble)."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    return type(h)(detach_state(x) for x in h)


def macs(config) -> int:
    """Multiply-adds per row (inference cost model). Ensembles: sum over members."""
    if "members" in config:
        return sum(macs(m) for m in config["members"])
    arch = config.get("arch", "gru")
    H, n_in = config["hidden"], N_FEATURES * (2 if config["use_diff"] else 1)
    if arch == "mingru":  # one [in -> 2H] projection per layer
        return sum(2 * H * (n_in if l == 0 else H) for l in range(config["layers"])) + 2 * H
    if arch == "tcn":  # input projection, per block a k-tap conv + 1x1 mix (LayerNorms ignored)
        return n_in * H + config["layers"] * (config["kernel"] + 1) * H * H + 2 * H
    gates = 4 if config.get("cell", "gru") == "lstm" else 3
    total = 0
    for _ in range(config["layers"]):
        total += gates * H * (n_in + H)
        n_in = H
    return total + 2 * H


BASELINE_MACS = macs(dict(hidden=128, layers=2, use_diff=False))  # 190,720


def _rw(w, rw):
    """Multiply metric weights [N,2] by optional per-row weights [N]."""
    return w if rw is None else w * rw[:, None]


def wmse_loss(p, y, rw=None):
    """Weighted MSE with the metric's clipping and |y| weights (times optional row weights). p, y: [N, 2]."""
    y = y.clamp(-CLIP, CLIP)
    w = _rw(y.abs(), rw)
    return ((w * (p - y) ** 2).sum(0) / (w.sum(0) + 1e-8)).mean()


def distill_loss(p, teacher, y, weight="y", rw=None):
    """MSE to the teacher's predictions. Row weights: "y" = |clipped target| (like the metric),
    "none" = uniform, "teacher" = |teacher prediction| (no dependence on the noisy target)."""
    if weight == "y":
        w = y.clamp(-CLIP, CLIP).abs()
    elif weight == "teacher":
        w = teacher.abs()
    elif weight == "none":
        w = torch.ones_like(teacher)
    else:
        raise ValueError(weight)
    w = _rw(w, rw)
    return ((w * (p - teacher) ** 2).sum(0) / (w.sum(0) + 1e-8)).mean()


def wpearson(p, y, rw=None):
    """Per-target weighted Pearson (pooled over the rows given), metric conventions. -> [2]."""
    y = y.clamp(-CLIP, CLIP)
    w = _rw(y.abs(), rw)
    W = w.sum(0) + 1e-8
    yc = y - (w * y).sum(0) / W
    pc = p - (w * p).sum(0) / W
    cov = (w * yc * pc).sum(0)
    return cov / torch.sqrt((w * yc * yc).sum(0) * (w * pc * pc).sum(0) + 1e-12)


def compute_loss(kind, p, y, alpha=0.1, rw=None):
    if kind == "wmse":
        return wmse_loss(p, y, rw)
    if kind == "pearson":
        return 1 - wpearson(p, y, rw).mean()
    if kind == "mix":  # correlation objective, small wmse term anchors the scale for clipping
        return 1 - wpearson(p, y, rw).mean() + alpha * wmse_loss(p, y, rw)
    raise ValueError(kind)


class EnsembleNet(nn.Module):
    """Average of several GRUNets. Each member's output is multiplied by `scales[k]` (per target)
    first, so members with different prediction spreads count equally. h is a list, one per member."""

    def __init__(self, nets, scales):
        super().__init__()
        self.nets = nn.ModuleList(nets)
        self.register_buffer("scales", torch.as_tensor(np.asarray(scales), dtype=torch.float32))  # [K, 2]
        self.use_diff = any(n.use_diff for n in nets)
        self.config = {"members": [n.config for n in nets], "scales": np.asarray(scales).tolist()}

    def forward(self, x, h=None, x_prev=None):
        h = h if h is not None else [None] * len(self.nets)
        preds, new_h = [], []
        for k, (net, hk) in enumerate(zip(self.nets, h)):
            p, hn = net(x, hk, x_prev if net.use_diff else None)
            preds.append(p * self.scales[k])
            new_h.append(hn)
        return torch.stack(preds).mean(0), new_h


def _split_layers(net):
    """Single-layer copies of a stacked GRU/LSTM, so the ONNX graph gets one state input per layer
    (as in the baseline) instead of slicing/concatenating a stacked state on every row."""
    rnn, mods = net.gru, []
    for l in range(rnn.num_layers):
        m = type(rnn)(rnn.input_size if l == 0 else rnn.hidden_size, rnn.hidden_size, 1, batch_first=True)
        for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            getattr(m, f"{n}_l0").data.copy_(getattr(rnn, f"{n}_l{l}").data)
        mods.append(m)
    return nn.ModuleList(mods)


def _member_states(m):
    """(name suffix, shape) of each streaming state of one model, in graph order."""
    if isinstance(m, GRUNet):
        H, lstm = m.config["hidden"], m.cell == "lstm"
        return [s for l in range(m.config["layers"])
                for s in [(f"h{l}", (1, 1, H))] + ([(f"c{l}", (1, 1, H))] if lstm else [])]
    return m.stream_states()


class _Step(nn.Module):
    """One-row step graph: (x [1,1,112], x_prev [1,112], *states) -> (pred [1,1,2], *new states).
    States are flat, per member: GRU one h [1,1,H] per layer, LSTM h and c per layer, minGRU one h [1,H] per
    layer, TCN one input buffer [1, pad, C] per block."""

    def __init__(self, net):
        super().__init__()
        self.ensemble = isinstance(net, EnsembleNet)
        members = list(net.nets) if self.ensemble else [net]
        self.rnn = [isinstance(m, GRUNet) for m in members]
        self.layers = nn.ModuleList([_split_layers(m) if r else nn.ModuleList() for m, r in zip(members, self.rnn)])
        self.heads = nn.ModuleList([m.head if r else nn.Identity() for m, r in zip(members, self.rnn)])
        self.streams = nn.ModuleList([nn.Identity() if r else m for m, r in zip(members, self.rnn)])
        self.n_states = [len(_member_states(m)) for m in members]
        self.diff = [m.use_diff for m in members]
        self.n_extra = [getattr(m, "n_extra", 0) for m in members]
        self.lstm = [r and m.cell == "lstm" for m, r in zip(members, self.rnn)]
        self.register_buffer("scales", net.scales if self.ensemble else torch.ones(1, 2))

    def forward(self, x, x_prev, *states):
        states, new, preds, i = list(states), [], [], 0
        for k, layers in enumerate(self.layers):
            base = x[..., :N_FEATURES]
            parts = [base] + ([base - x_prev[:, None]] if self.diff[k] else []) + \
                    ([x[..., N_FEATURES:N_FEATURES + self.n_extra[k]]] if self.n_extra[k] else [])
            z = torch.cat(parts, -1) if len(parts) > 1 else base
            if not self.rnn[k]:  # minGRU / TCN: the model's own one-row update
                p, ns = self.streams[k].stream_step(z, states[i:i + self.n_states[k]])
                new += ns
                i += self.n_states[k]
            else:
                for layer in layers:
                    if self.lstm[k]:
                        z, (h, c) = layer(z, (states[i], states[i + 1]))
                        new += [h, c]
                        i += 2
                    else:
                        z, h = layer(z, states[i])
                        new.append(h)
                        i += 1
                p = self.heads[k](z)
            preds.append(p * self.scales[k] if self.ensemble else p)
        pred = preds[0] if len(preds) == 1 else torch.stack(preds).mean(0)
        return (pred, *new)


def state_names(net):
    members = list(net.nets) if isinstance(net, EnsembleNet) else [net]
    return [f"m{k}_{s}" for k, m in enumerate(members) for s, _ in _member_states(m)]


def export_onnx(net, path: str):
    """Export a model (GRUNet, MinGRUNet, TCNNet) or EnsembleNet as a one-row step graph. Inputs: x, x_prev
    (dropped from the graph when no member uses diff features), then the states per member named m{k}_<state>
    (h{l} / c{l} for GRU/LSTM and minGRU, b{l} for TCN buffers); outputs: pred, then the new states in the same
    order. All states start at zero (solution.py zero-fills them from the graph's input shapes)."""
    net = net.eval().cpu()
    names = state_names(net)
    members = list(net.nets) if isinstance(net, EnsembleNet) else [net]
    states = [torch.zeros(*shape) for m in members for _, shape in _member_states(m)]
    args = (torch.zeros(1, 1, N_FEATURES + needs_extra(net)), torch.zeros(1, N_FEATURES), *states)
    with torch.no_grad():
        torch.onnx.export(_Step(net), args, path, input_names=["x", "x_prev", *names],
                          output_names=["pred", *[n + "_out" for n in names]], opset_version=17, dynamo=False)


def load_baseline(onnx_path: str) -> GRUNet:
    """Load the organisers' 2-layer GRU ONNX weights into a GRUNet (for checks / warm starts)."""
    import onnx
    from onnx import numpy_helper

    g = onnx.load(onnx_path).graph
    init = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    grus = [n for n in g.node if n.op_type == "GRU"]
    matmul = next(n for n in g.node if n.op_type == "MatMul")
    H = 128
    net = GRUNet(hidden=H, layers=len(grus))

    def reorder(a):  # ONNX gates z,r,h -> PyTorch r,z,n
        return np.concatenate([a[H:2 * H], a[:H], a[2 * H:]], 0)

    sd = {}
    for l, n in enumerate(grus):
        W, R, B = (init[n.input[i]][0] for i in (1, 2, 3))
        sd[f"gru.weight_ih_l{l}"] = reorder(W)
        sd[f"gru.weight_hh_l{l}"] = reorder(R)
        sd[f"gru.bias_ih_l{l}"] = reorder(B[:3 * H])
        sd[f"gru.bias_hh_l{l}"] = reorder(B[3 * H:])
    sd["head.weight"] = init[matmul.input[1]].T
    sd["head.bias"] = init["reg_head.bias"]
    net.load_state_dict({k: torch.from_numpy(np.array(v, dtype=np.float32)) for k, v in sd.items()})
    return net
