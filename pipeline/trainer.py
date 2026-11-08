"""Shared training loop: one data stream feeds any number of experiments.

Every batch of whole sequences is read from parquet once and then used to train each model in
turn (truncated BPTT, hidden state carried between `chunk`-step pieces, loss on steps >= 99).
Evaluation is shared the same way: one pass over the eval data scores all models.

Optional per model: EMA of the weights (evaluated and saved instead of the raw weights),
distillation towards a teacher's predictions on the training set (see pipeline/teacher.py), its own
number of passes, and a data fold to leave out (out-of-fold teachers: sequences with
id % n_folds == fold are skipped by that model), extra input features of a kind (pipeline/features.py),
auxiliary targets (aux_targets) learned by a training-only head, one instrument group only (`group`: the model
trains on that group's sequences and is selected on that group's validation rows), and a starting checkpoint
(`init`).

Validation sequences can be trained on too (Common.pool "valid" or "train+valid"; ExpConfig valid_part = which half of
split.json, or all): their rows are weighted by the true mask (scored 1, unscored `unscored_weight`), and such models
are not selected on validation (select=False: the final weights, or the average of the end-of-epoch weights from
epoch `swa_from` on, are saved).
"""
import copy
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from utils import WARMUP

from .data import VOFF, PoolStore, SeqStore, split_train, take_rows, valid_key
from .groups import label as group_label, state_values
from .features import offline as extra_features, width as feat_width
from .fast_eval import evaluate_many
from .model import BASELINE_MACS, CLIP, GRUNet, build_net, compute_loss, detach_state, distill_loss, export_onnx, load_baseline, macs

HERE = os.path.dirname(os.path.abspath(__file__))
AUX_CHANGES = [(0, 10), (0, 50), (0, 200), (0, 1000), (1, 50), (1, 200)]  # (instrument, rows ahead)
AUX_VOL = 100  # plus: mean |change| of i0's average price over the next AUX_VOL rows
N_AUX = len(AUX_CHANGES) + 1


def aux_targets(X):
    """Auxiliary targets from the raw inputs X [B,T,>=112] (torch): future change of each instrument's average
    price column (i0: 0-21, i1: 52-73) at several horizons, and i0's near-future activity. -> A, V [B,T,N_AUX]:
    targets scaled to unit spread over the batch (clipped to +-5) and where they exist (V, rows with enough future)."""
    B, T = X.shape[:2]
    mids = [X[..., 0:22].mean(-1), X[..., 52:74].mean(-1)]
    A, V = X.new_zeros(B, T, N_AUX), torch.zeros(B, T, N_AUX, dtype=torch.bool, device=X.device)
    for k, (i, h) in enumerate(AUX_CHANGES):
        A[:, :T - h, k] = mids[i][:, h:] - mids[i][:, :-h]
        V[:, :T - h, k] = True
    step = (mids[0][:, 1:] - mids[0][:, :-1]).abs()  # step[t] = |m[t+1] - m[t]|
    c = torch.cat([step.new_zeros(B, 1), step.cumsum(1)], 1)  # c[t] = sum of step[:t]
    n = T - AUX_VOL
    A[:, :n, -1] = (c[:, AUX_VOL:AUX_VOL + n] - c[:, :n]) / AUX_VOL
    V[:, :n, -1] = True
    for k in range(N_AUX):
        a = A[..., k][V[..., k]]
        A[..., k] = ((A[..., k] - a.mean()) / (a.std() + 1e-8)).clamp(-5, 5) * V[..., k]
    return A, V


@dataclass
class ExpConfig:
    """Per-model settings. Everything about the data stream is shared (see Common)."""
    name: str
    hidden: int = 128
    layers: int = 2
    diff: bool = False
    dropout: float = 0.0
    init_baseline: bool = False
    loss: str = "wmse"          # wmse | pearson | mix
    alpha: float = 0.1          # wmse weight in mix
    lr: float = 2e-3
    wd: float = 1e-4
    seed: int = 0
    cell: str = "gru"           # gru | lstm
    chunk: int | None = None    # TBPTT length for this model (default: Common.chunk)
    ema: float = 0.0            # >0: keep an EMA of the weights with this decay; eval/save the EMA
    distill: float = 0.0        # weight of the teacher loss (0 = off, 1 = teacher only); needs a teacher
    distill_weight: str = "y"   # teacher-loss row weights: y (|clipped target|, like the metric) | none | teacher
    teacher: str | None = None  # this model's teacher .npy (default: Common.teacher)
    epochs: int | None = None   # passes for this model (default: Common.epochs)
    fold: int | None = None     # leave out training sequences with id % Common.n_folds == fold
    mask_weight: float = 0.0    # lambda: row weight (1 - lambda) + lambda * p^mask_power / mean(p^mask_power)
    mask_power: float = 1.0     # p = predicted p(scored) from Common.row_weights (pipeline/mask_model.py)
    feats: bool | str = False   # extra causal input features: a kind from features.KINDS (act | norm | act+norm);
                                # True = act (the 36 activity / EMA features)
    aux: float = 0.0            # >0: weight of the auxiliary-target loss (aux_targets; GRU only, not exported)
    micro_batch: int | None = None  # sequences per optimizer step for this model (default: the whole data batch)
    arch: str = "gru"           # gru (cell gru|lstm) | mingru | tcn (see pipeline/model.py)
    kernel: int = 2             # tcn only: kernel size; dilations kernel**l, receptive field kernel**layers rows
    group: int | None = None    # 0..3: train only on this instrument group's sequences (pipeline/groups.py)
    init: str | None = None     # start from this best.pt (same shape; a plain GRU also starts an arch="ggru" model)
    states: int | None = None   # 0..3: joint-state output head over that group's (t0, t1) states (groups.STATE_TICKS)
    cls: float = 0.0            # >0: weight of the state cross-entropy (|y|-weighted) added to the loss; needs states
    use_train: bool = True      # train on the train.parquet sequences of the pool
    valid_part: str | None = None  # A | B | all: also train on these validation sequences (split.json halves)
    valid_repeat: int = 1       # each pass sees this model's validation sequences this many times
    unscored_weight: float = 1.0  # validation rows: weight of unscored rows (scored rows 1), normalised to mean 1
    valid_distill: float = 0.0  # validation rows: the teacher target is (1 - v) * clip(y) + v * Common.teacher_valid
    select: bool = True         # False: no checkpoint selection, keep the final (or averaged) weights
    swa_from: int | None = None # with select=False: average the end-of-epoch weights of passes swa_from.. (1-based)
    sched: str = "onecycle"     # onecycle | const
    seq_weights: str | None = None  # .npy [n_train]: a weight per train sequence (pipeline/shift.py), times the row weights
    train_part: str = "all"     # all | head (output layer only) | top (output layer + last GRU layer): the rest is frozen
    valid_frac: float = 1.0     # <1: train on this random share of the validation sequences (bagging, for weight soups)
    bag_seed: int = 0           # which random share (valid_frac)


@dataclass
class Common:
    epochs: int = 4
    batch: int = 256            # sequences per batch
    chunk: int = 500            # TBPTT length
    clip_grad: float = 1.0
    train_seqs: int | None = None
    holdout: int = 500
    valid_seqs: int = 400
    evals_per_epoch: int = 2
    seed: int = 0               # data order + holdout split
    device: str = "auto"
    threads: int = 4            # parquet reader threads
    cache_eval: bool = False    # keep eval data in RAM (~9 MB per sequence)
    cache_train: bool = False   # keep training sequences in RAM after first read
    data_dir: str = "datasets"
    out_dir: str = "runs"
    teacher: str | None = None  # teacher predictions on train.parquet (.npy, [n_seqs, 20000, 2]) for distillation
    n_folds: int = 4            # for ExpConfig.fold
    row_weights: str | None = None  # p(scored) per train row (.npy [n_seqs, 20000]) for ExpConfig.mask_weight
    valid_split: str | None = None  # split.json from mask_model: evaluate and select on validation half B
    pool: str = "train"         # training stream: train | valid | train+valid (validation sequences: ExpConfig.valid_part)
    teacher_valid: str | None = None  # teacher predictions on valid.parquet (.npy [1873, 20000, 2]), ExpConfig.valid_distill
    skip_final_eval: bool = False  # no full validation / holdout scoring at the end (export only)
    pool_valid_seqs: int | None = None  # tests: only the first N sequences of each validation half join the pool
    extra: dict = field(default_factory=dict)


class Experiment:
    def __init__(self, cfg: ExpConfig, common: Common, device, n_batches):
        self.cfg = cfg
        self.chunk = cfg.chunk or common.chunk
        self.epochs = cfg.epochs or common.epochs
        self.teacher_path = (cfg.teacher or common.teacher) if cfg.distill > 0 else None
        assert cfg.distill == 0 or self.teacher_path, f"{cfg.name}: distill > 0 needs a teacher"
        self.micro = cfg.micro_batch or common.batch
        total_steps = (self.epochs * n_batches * math.ceil(common.batch / self.micro)  # upper bound on steps
                       * math.ceil(20000 / self.chunk))
        self.kind = "act" if cfg.feats is True else (cfg.feats or None)
        n_extra = feat_width(self.kind)
        assert cfg.aux == 0 or cfg.arch == "gru", "auxiliary targets: GRU only"
        assert cfg.states is None or (cfg.arch == "gru" and not cfg.init_baseline), "state head: plain GRU only"
        assert cfg.cls == 0 or cfg.states is not None, "cls needs a state head (states)"
        assert cfg.valid_part in (None, "A", "B", "all"), cfg.valid_part
        assert cfg.valid_part is None or ("valid" in common.pool and common.valid_split), \
            f"{cfg.name}: valid_part needs --pool valid / train+valid and --valid-split"
        assert cfg.valid_part is None or not cfg.select, f"{cfg.name}: trained on validation -> select=False"
        assert cfg.use_train and "train" in common.pool or cfg.valid_part, f"{cfg.name}: nothing to train on"
        assert cfg.valid_distill == 0 or (cfg.distill > 0 and common.teacher_valid), "valid_distill: distill + teacher_valid"
        assert cfg.swa_from is None or not cfg.select, "swa_from needs select=False"
        assert cfg.train_part in ("all", "head", "top"), cfg.train_part
        assert cfg.train_part == "all" or (cfg.init and cfg.arch == "gru"), f"{cfg.name}: train_part needs a GRU init"
        assert 0 < cfg.valid_frac <= 1 and (cfg.valid_frac == 1 or cfg.valid_part), f"{cfg.name}: valid_frac"
        torch.manual_seed(cfg.seed)
        if cfg.arch == "ggru":
            assert self.kind == "grp", "arch ggru takes the group probabilities: feats='grp'"
        if cfg.init_baseline:
            assert cfg.arch == "gru" and not cfg.diff and cfg.hidden == 128 and cfg.layers == 2, "baseline shape is fixed"
            self.net = load_baseline("baseline/baseline.onnx")
        elif cfg.arch == "gru":
            self.net = GRUNet(cfg.hidden, cfg.layers, cfg.diff, cfg.dropout, cfg.cell, n_extra=n_extra,
                              feat_kind=self.kind, n_aux=N_AUX if cfg.aux > 0 else 0,
                              states=state_values(cfg.states) if cfg.states is not None else None)
        else:
            self.net = build_net(dict(arch=cfg.arch, hidden=cfg.hidden, layers=cfg.layers, use_diff=cfg.diff,
                                      dropout=cfg.dropout, n_extra=n_extra, feat_kind=self.kind,
                                      **({"kernel": cfg.kernel} if cfg.arch == "tcn" else {})))
        if cfg.init:
            ck = torch.load(cfg.init, map_location="cpu", weights_only=False)
            if cfg.arch == "ggru" and ck["config"].get("arch", "gru") == "gru":
                self.net.from_gru(ck["state_dict"])
            elif cfg.states is not None and "states" not in ck["config"]:  # new state head on a plain model's body
                sd = {k: v for k, v in ck["state_dict"].items() if not k.startswith("head.")}
                missing, unexpected = self.net.load_state_dict(sd, strict=False)
                assert not unexpected and all(k.startswith("head.") for k in missing), (missing, unexpected)
            else:
                self.net.load_state_dict(ck["state_dict"])
        self.net.to(device)
        self.ema_net = copy.deepcopy(self.net) if cfg.ema > 0 else None
        self.n_updates = 0
        self.total_steps = total_steps
        if cfg.train_part != "all":  # freeze everything below the trained part
            top = f"_l{self.net.config['layers'] - 1}"
            for n, p in self.net.named_parameters():
                p.requires_grad_(n.startswith("head.") or (cfg.train_part == "top" and n.startswith("gru.")
                                                             and (n.endswith(top) or n.endswith(top + "_reverse"))))
        trainable = [p for p in self.net.parameters() if p.requires_grad]
        self.n_trainable = sum(p.numel() for p in trainable)
        self.opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.wd)
        self.in_bag = None  # valid_frac < 1: bool per validation sequence (set by train_many)
        if cfg.sched == "const":
            self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda step: 1.0)
        else:
            assert cfg.sched == "onecycle", cfg.sched
            self.sched = torch.optim.lr_scheduler.OneCycleLR(self.opt, max_lr=cfg.lr, total_steps=total_steps,
                                                             pct_start=max(0.05, 3 / total_steps))
        self.swa, self.n_swa = None, 0
        self.seq_w = None
        if cfg.seq_weights:
            self.seq_w = np.load(cfg.seq_weights).astype(np.float32)
            assert self.seq_w.ndim == 1 and (self.seq_w >= 0).all(), f"{cfg.seq_weights}: shape {self.seq_w.shape}"
        self.dir = os.path.join(common.out_dir, cfg.name)
        os.makedirs(self.dir, exist_ok=True)
        self.log_f = open(os.path.join(self.dir, "log.jsonl"), "a")
        self.best = -1.0
        self.params = sum(p.numel() for p in self.net.parameters())
        self.cost = macs(self.net.config) / BASELINE_MACS

    @property
    def eval_net(self):
        return self.ema_net if self.ema_net is not None else self.net

    @torch.no_grad()
    def update_ema(self):
        self.n_updates += 1
        if self.ema_net is None:
            return
        decay = min(self.cfg.ema, (1 + self.n_updates) / (10 + self.n_updates))  # warm-up
        for pe, p in zip(self.ema_net.parameters(), self.net.parameters()):
            pe.lerp_(p, 1 - decay)

    @torch.no_grad()
    def add_swa(self):
        sd = self.eval_net.state_dict()
        if self.swa is None:
            self.swa = {k: v.detach().cpu().double() if v.is_floating_point() else v.detach().cpu().clone()
                        for k, v in sd.items()}  # float64 on the CPU (MPS has no float64)
        else:
            for k, v in sd.items():
                if v.is_floating_point():
                    self.swa[k] += v.detach().cpu().double()
        self.n_swa += 1

    def save(self, n_folds):
        torch.save({"config": self.net.config, "state_dict": self.eval_net.state_dict(), "exp": asdict(self.cfg),
                    "n_folds": n_folds}, os.path.join(self.dir, "best.pt"))

    def log(self, **kw):
        line = json.dumps(kw)
        self.log_f.write(line + "\n")
        self.log_f.flush()
        print(f"[{self.cfg.name}] {line}", flush=True)


def _store(common, name):
    return SeqStore(os.path.join(common.data_dir, f"{name}.parquet"), n_threads=common.threads)


def train_many(configs: list[ExpConfig], common: Common):
    from .fast_eval import pick_device

    device = pick_device(common.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    rng = np.random.default_rng(common.seed)
    train, valid = _store(common, "train"), _store(common, "valid")
    assert common.pool in ("train", "valid", "train+valid"), common.pool
    pool = PoolStore(train, _store(common, "valid")) if "valid" in common.pool else train
    if common.cache_train:
        train.enable_cache()
    if common.cache_eval:
        valid.enable_cache()
    train_ids, hold_ids = split_train(train.n_seqs, common.holdout, common.seed)
    if common.train_seqs:
        train_ids = rng.choice(train_ids, common.train_seqs, replace=False)
    split = json.load(open(common.valid_split)) if common.valid_split else None
    valid_B = split["B"] if split else None
    valid_A = split["A"] if split else None
    valid_ids = (valid_B or list(range(valid.n_seqs)))[:common.valid_seqs]
    pool_ids = list(train_ids) if "train" in common.pool else []
    if "valid" in common.pool:
        reps = max([c.valid_repeat for c in configs if c.valid_part] or [1])
        vs = sorted(valid_A[:common.pool_valid_seqs] + valid_B[:common.pool_valid_seqs])  # halves A + B = all of valid
        pool_ids += [valid_key(v, c) for c in range(reps) for v in vs]
        half_of = {v: "A" for v in valid_A}
        half_of.update({v: "B" for v in valid_B})
        mtab = pq.read_table(os.path.join(common.data_dir, "valid_mask.parquet"), columns=["step_in_seq", "is_scored"])
        need = mtab["step_in_seq"].to_numpy() >= WARMUP
        scored_share = float(mtab["is_scored"].to_numpy()[need].astype(np.float64).mean())  # of the required rows
        print(f"pool {common.pool}: {len(pool_ids)} sequence passes per epoch; validation scored share {scored_share:.4f}",
              flush=True)
    hold_store = train
    if common.cache_eval and not common.cache_train:  # cache holdout without caching all of train
        hold_store = _store(common, "train")
        hold_store.enable_cache()

    n_batches = math.ceil(len(pool_ids) / common.batch)
    exps = [Experiment(c, common, device, n_batches) for c in configs]
    for e in exps:
        if e.cfg.valid_frac < 1:
            e.in_bag = np.random.default_rng(10_000 + e.cfg.bag_seed).random(valid.n_seqs) < e.cfg.valid_frac
    teachers = {}  # path -> memmap, loaded once and shared by the models that use it
    for path in sorted({e.teacher_path for e in exps if e.teacher_path}):
        teachers[path] = np.load(path, mmap_mode="r")
        assert teachers[path].shape[:2] == (train.n_seqs, 20000), f"teacher {path}: shape {teachers[path].shape}"
    teacher_valid = None
    if any(e.cfg.valid_distill > 0 for e in exps):
        teacher_valid = np.load(common.teacher_valid, mmap_mode="r")
        assert teacher_valid.shape[:2] == (valid.n_seqs, 20000), f"teacher_valid: shape {teacher_valid.shape}"
    row_w, p_norm = None, {}
    if any(e.cfg.mask_weight > 0 for e in exps):
        assert common.row_weights, "mask_weight > 0 needs --row-weights (see pipeline/mask_model.py)"
        row_w = np.load(common.row_weights, mmap_mode="r")
        assert row_w.shape == (train.n_seqs, 20000), f"row weights shape {row_w.shape}"
        sample = np.asarray(row_w[np.sort(np.asarray(train_ids))[::20], 99:], dtype=np.float32)  # train rows only
        for pw in {e.cfg.mask_power for e in exps if e.cfg.mask_weight > 0}:
            p_norm[pw] = float((sample ** pw).mean())  # so the average row weight stays about 1
            print(f"row weights: mean p^{pw:g} = {p_norm[pw]:.4f}", flush=True)
    n_epochs = max(e.epochs for e in exps)
    t0 = time.time()
    for e in exps:
        e.log(event="start", config=asdict(e.cfg), common=asdict(common), device=str(device), params=e.params,
              cost=round(e.cost, 3), train_seqs=len(train_ids), holdout=len(hold_ids), steps=e.total_steps,
              trainable=e.n_trainable, bag=int(e.in_bag.sum()) if e.in_bag is not None else None,
              pool=common.pool, pool_seqs=len(pool_ids))
    eval_at = set(np.linspace(0, n_batches, common.evals_per_epoch + 1, dtype=int)[1:] - 1)

    def run_eval(epoch, bi, group):
        nets = [e.eval_net for e in group]
        vs = evaluate_many(nets, valid, valid_ids, device)
        hs = evaluate_many(nets, hold_store, hold_ids, device)
        for e, v, h in zip(group, vs, hs):
            # with a valid split: select on half B's scored rows (the mask model never saw them);
            # otherwise don't select on the visible valid mask alone
            sel = v["scored"] if valid_B else (v["scored"] + h["all"]) / 2
            if e.cfg.group is not None and v[f"scored_g{e.cfg.group}"] == v[f"scored_g{e.cfg.group}"]:
                sel = v[f"scored_g{e.cfg.group}"]  # a group model is selected on its group's rows (if any)
            is_best = e.cfg.select and sel > e.best
            if is_best:
                e.best = sel
                e.save(common.n_folds)
            e.log(event="eval", epoch=epoch, batch=bi, valid=round(v["scored"], 6), valid_all=round(v["all"], 6),
                  valid_groups=[round(v[f"scored_g{g}"], 4) for g in range(4)],
                  holdout=round(h["all"], 6), select=round(sel, 6), best=is_best, time=round(time.time() - t0, 1))
            e.net.train()

    for epoch in range(n_epochs):
        active = [e for e in exps if epoch < e.epochs]
        for e in active:
            e.net.train()
        ids = np.asarray(rng.permutation(np.asarray(pool_ids, np.int64)))
        for bi, (X, Y, M) in enumerate(pool.batches(list(ids), common.batch)):
            tb = time.time()
            batch_ids = ids[bi * common.batch:(bi + 1) * common.batch]
            groups = group_label(Y)
            is_tr = batch_ids < VOFF
            vseq = batch_ids % VOFF  # validation sequence of a validation key
            vcopy = batch_ids // VOFF - 1
            vhalf = np.array([half_of[v] if not t else "" for v, t in zip(vseq, is_tr)]) if not is_tr.all() else None
            Xd = torch.from_numpy(X).to(device, non_blocking=True)  # one host->device copy per batch
            # extras once per batch and kind, shared by every model of that kind, joined on the GPU (the joined
            # 256-seq batch would be another 3+ GB array in host RAM)
            Xk = {None: Xd, **{k: torch.cat([Xd, torch.from_numpy(extra_features(X, k)).to(device)], -1)
                               for k in sorted({e.kind for e in active} - {None})}}
            Ad = Vd = None
            if any(e.cfg.aux > 0 for e in active):
                Ad, Vd = aux_targets(Xd)
            Yd = torch.from_numpy(Y).to(device, non_blocking=True)
            def per_train_seq(arr):  # rows of a train-indexed array for this batch (0 for validation sequences)
                out = np.zeros((len(batch_ids),) + arr.shape[1:], np.float32)
                if is_tr.any():
                    out[is_tr] = take_rows(arr, batch_ids[is_tr])
                return torch.from_numpy(out).to(device)
            Td = {path: per_train_seq(t) for path, t in teachers.items() if any(e.teacher_path == path for e in active)}
            Pd = None
            if row_w is not None and any(e.cfg.mask_weight > 0 for e in active):
                Pd = per_train_seq(row_w)
            Md = TVd = None
            if vhalf is not None:
                Md = torch.from_numpy(M).to(device)
                if teacher_valid is not None and any(e.cfg.valid_distill > 0 for e in active):
                    tv = np.zeros((len(batch_ids), 20000, 2), np.float32)
                    tv[~is_tr] = take_rows(teacher_valid, vseq[~is_tr])
                    TVd = torch.from_numpy(tv).to(device)
            for e in active:
                Xe, Ye, Te = Xk[e.kind], Yd, Td.get(e.teacher_path)
                Ae, Ve = (Ad, Vd) if e.cfg.aux > 0 else (None, None)
                We = None
                if e.cfg.mask_weight > 0:
                    lam, pw = e.cfg.mask_weight, e.cfg.mask_power
                    We = (1 - lam) + lam * Pd.clamp(min=0) ** pw / p_norm[pw]
                keep = is_tr.copy() if e.cfg.use_train else np.zeros(len(batch_ids), bool)
                if e.cfg.fold is not None:  # out-of-fold model: drop its fold's sequences from the batch
                    keep &= ~is_tr | (batch_ids % common.n_folds != e.cfg.fold)
                if e.cfg.valid_part and vhalf is not None:  # its validation sequences (and copies)
                    mine = ~is_tr & (vcopy < e.cfg.valid_repeat) & ((vhalf == e.cfg.valid_part) | (e.cfg.valid_part == "all"))
                    if e.in_bag is not None:  # bagging: only this model's share of the validation sequences
                        mine[~is_tr] &= e.in_bag[vseq[~is_tr]]
                    keep |= mine
                vrows = ~is_tr & keep
                if vrows.any():  # validation rows: true-mask weights; teacher target blended with the labels
                    u = e.cfg.unscored_weight
                    wv = (Md.float() + u * (~Md).float()) / (scored_share + u * (1 - scored_share))
                    vr = torch.as_tensor(np.flatnonzero(vrows), device=device)
                    We = torch.ones_like(Md, dtype=torch.float32) if We is None else We.clone()
                    We[vr] = wv[vr]
                    if Te is not None:
                        v = e.cfg.valid_distill
                        tv = Yd[vr].clamp(-CLIP, CLIP) if v == 0 else (1 - v) * Yd[vr].clamp(-CLIP, CLIP) + v * TVd[vr]
                        Te = Te.clone()
                        Te[vr] = tv
                if e.seq_w is not None and is_tr.any():  # importance weight per train sequence
                    ws = np.ones(len(batch_ids), np.float32)
                    ws[is_tr] = e.seq_w[batch_ids[is_tr]]
                    We = (We if We is not None else torch.ones(Yd.shape[:2], device=device)) * torch.from_numpy(ws).to(device)[:, None]
                if e.cfg.group is not None:  # group model: only its group's sequences
                    keep &= groups == e.cfg.group
                if not keep.any():
                    continue
                if not keep.all():
                    rows = torch.as_tensor(np.flatnonzero(keep), device=device)
                    Xe, Ye = Xe[rows], Yd[rows]
                    Ae, Ve = (Ae[rows], Ve[rows]) if Ae is not None else (None, None)
                    Te = Te[rows] if Te is not None else None
                    We = We[rows] if We is not None else None
                losses = []
                for m0 in range(0, len(Xe), e.micro):  # mini-batches of whole sequences, each its own TBPTT pass
                    Xm, Ym = Xe[m0:m0 + e.micro], Ye[m0:m0 + e.micro]
                    Tm = Te[m0:m0 + e.micro] if Te is not None else None
                    Wm = We[m0:m0 + e.micro] if We is not None else None
                    Am, Vm = (Ae[m0:m0 + e.micro], Ve[m0:m0 + e.micro]) if Ae is not None else (None, None)
                    h = xp = None
                    for s in range(0, X.shape[1], e.chunk):
                        x, y = Xm[:, s:s + e.chunk], Ym[:, s:s + e.chunk]
                        p, h = e.net(x, h, xp)
                        lo = max(0, WARMUP - s)  # skip warm-up rows
                        if lo < x.shape[1]:
                            pf, yf = p[:, lo:].reshape(-1, 2), y[:, lo:].reshape(-1, 2)
                            rw = Wm[:, s:s + e.chunk][:, lo:].reshape(-1) if Wm is not None else None
                            lam = e.cfg.distill
                            loss = compute_loss(e.cfg.loss, pf, yf, e.cfg.alpha, rw) if lam < 1 else 0.0
                            if lam > 0:
                                tf = Tm[:, s:s + e.chunk][:, lo:].reshape(-1, 2)
                                loss = (1 - lam) * loss + lam * distill_loss(pf, tf, yf, e.cfg.distill_weight, rw)
                            if e.cfg.cls > 0:  # cross-entropy of the joint target state, |y|-weighted like the metric
                                hd = e.net.head
                                lg = hd.last_logits[:, lo:].reshape(-1, hd.values.shape[0])
                                wc = yf.clamp(-CLIP, CLIP).abs().sum(1) * (rw if rw is not None else 1.0)
                                ce = F.cross_entropy(lg, hd.state_index(yf), reduction="none")
                                loss = loss + e.cfg.cls * (wc * ce).sum() / wc.sum().clamp(min=1e-8)
                            if Am is not None:  # auxiliary targets on the same rows, from the training-only head
                                pa = e.net.aux_head(e.net.last_out[:, lo:])
                                ta, va = Am[:, s:s + e.chunk][:, lo:], Vm[:, s:s + e.chunk][:, lo:]
                                loss = loss + e.cfg.aux * ((pa - ta) ** 2 * va).sum() / va.sum().clamp(min=1)
                            e.opt.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(e.net.parameters(), common.clip_grad)
                            e.opt.step()
                            e.sched.step()
                            e.update_ema()
                            losses.append(loss.detach())  # no GPU sync per step
                        h = detach_state(h)
                        xp = x[:, -1]
                e.log(event="batch", epoch=epoch, batch=bi, loss=round(float(torch.stack(losses).mean()), 5),
                      lr=round(e.sched.get_last_lr()[0], 6), steps=len(losses))
            print(f"batch {bi + 1}/{n_batches} epoch {epoch + 1}/{n_epochs}: {time.time() - tb:.1f}s "
                  f"for {len(active)} models, elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
            del Xd, Xk, Yd, Td, Pd, Ad, Vd, Md, TVd
            if bi in eval_at:
                run_eval(epoch, bi, active)
        for e in active:  # end of pass: weight averaging (select=False models)
            if e.cfg.swa_from is not None and epoch + 1 >= e.cfg.swa_from:
                e.add_swa()

    for e in exps:  # select=False: the final weights, or their average over the last passes
        if e.cfg.select:
            continue
        if e.n_swa:
            sd = e.eval_net.state_dict()
            e.eval_net.load_state_dict({k: (e.swa[k] / e.n_swa).to(v.dtype) if v.is_floating_point() else v
                                        for k, v in sd.items()})
        e.save(common.n_folds)
        e.log(event="final", swa_passes=e.n_swa)

    # final: best checkpoints -> full validation + holdout, ONNX submission folders, summary table
    for e in exps:
        ck = torch.load(os.path.join(e.dir, "best.pt"), map_location="cpu", weights_only=False)
        e.net.load_state_dict(ck["state_dict"])
        e.net.to(device)
    nets = [e.net for e in exps]
    if common.skip_final_eval:  # scored elsewhere (pipeline/crossfit.py)
        for e in exps:
            sub = os.path.join(e.dir, "submission")
            os.makedirs(sub, exist_ok=True)
            export_onnx(e.net, os.path.join(sub, "model.onnx"))
            shutil.copy(os.path.join(HERE, "solution_template.py"), os.path.join(sub, "solution.py"))
            shutil.copy(os.path.join(HERE, "features.py"), os.path.join(sub, "wnn_features.py"))
            json.dump({"final_eval": "skipped"}, open(os.path.join(e.dir, "results.json"), "w"))
            e.log(event="done", final_eval="skipped")
        with open(os.path.join(common.out_dir, "summary.jsonl"), "a") as f:
            for e in exps:
                f.write(json.dumps({"name": e.cfg.name, "params": e.params, "cost": round(e.cost, 2)}) + "\n")
        return []
    full_valid = evaluate_many(nets, _store(common, "valid"), range(valid.n_seqs), device)  # uncached: 17 GB
    full_hold = evaluate_many(nets, hold_store, hold_ids, device)
    full_B = evaluate_many(nets, _store(common, "valid"), valid_B, device) if valid_B else [None] * len(exps)
    rows = []
    for e, v, h, b in zip(exps, full_valid, full_hold, full_B):
        sub = os.path.join(e.dir, "submission")
        os.makedirs(sub, exist_ok=True)
        export_onnx(e.net, os.path.join(sub, "model.onnx"))
        shutil.copy(os.path.join(HERE, "solution_template.py"), os.path.join(sub, "solution.py"))
        shutil.copy(os.path.join(HERE, "features.py"), os.path.join(sub, "wnn_features.py"))
        res = {"valid_full": v, "holdout": h, **({"valid_B": b} if b else {})}
        json.dump(res, open(os.path.join(e.dir, "results.json"), "w"), indent=1)
        e.log(event="done", **res)
        rows.append({"name": e.cfg.name, "params": e.params, "cost": round(e.cost, 2), "valid_scored": round(v["scored"], 6),
                     "valid_all": round(v["all"], 6), "holdout_all": round(h["all"], 6),
                     "t0": round(v["t0"], 4), "t1": round(v["t1"], 4),
                     **({"valid_B_scored": round(b["scored"], 6), "valid_B_all": round(b["all"], 6),
                         **{f"B_g{g + 1}": round(b[f"scored_g{g}"], 4) for g in range(4)}} if b else {}),
                     **({"group": e.cfg.group + 1} if e.cfg.group is not None else {})})
    rows.sort(key=lambda r: -r["valid_B_scored"] if "valid_B_scored" in r else -(r["valid_scored"] + r["holdout_all"]))
    with open(os.path.join(common.out_dir, "summary.jsonl"), "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("\n" + " | ".join(rows[0].keys()))
    for r in rows:
        print(" | ".join(str(v) for v in r.values()))
    return rows
