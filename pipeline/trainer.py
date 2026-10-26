"""Shared training loop: one data stream feeds any number of experiments.

Every batch of whole sequences is read from parquet once and then used to train each model in
turn (truncated BPTT, hidden state carried between `chunk`-step pieces, loss on steps >= 99).
Evaluation is shared the same way: one pass over the eval data scores all models.

Optional per model: EMA of the weights (evaluated and saved instead of the raw weights),
distillation towards a teacher's predictions on the training set (see pipeline/teacher.py), its own
number of passes, and a data fold to leave out (out-of-fold teachers: sequences with
id % n_folds == fold are skipped by that model).
"""
import copy
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from utils import WARMUP

from .data import SeqStore, split_train
from .features import N_EXTRA, offline as extra_features
from .fast_eval import evaluate_many
from .model import BASELINE_MACS, GRUNet, build_net, compute_loss, detach_state, distill_loss, export_onnx, load_baseline, macs

HERE = os.path.dirname(os.path.abspath(__file__))


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
    feats: bool = False         # also feed the 36 extra causal features (pipeline/features.py)
    micro_batch: int | None = None  # sequences per optimizer step for this model (default: the whole data batch)
    arch: str = "gru"           # gru (cell gru|lstm) | mingru | tcn (see pipeline/model.py)
    kernel: int = 2             # tcn only: kernel size; dilations kernel**l, receptive field kernel**layers rows


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
        torch.manual_seed(cfg.seed)
        if cfg.init_baseline:
            assert cfg.arch == "gru" and not cfg.diff and cfg.hidden == 128 and cfg.layers == 2, "baseline shape is fixed"
            self.net = load_baseline("baseline/baseline.onnx")
        elif cfg.arch == "gru":
            self.net = GRUNet(cfg.hidden, cfg.layers, cfg.diff, cfg.dropout, cfg.cell,
                              n_extra=N_EXTRA if cfg.feats else 0)
        else:
            self.net = build_net(dict(arch=cfg.arch, hidden=cfg.hidden, layers=cfg.layers, use_diff=cfg.diff,
                                      dropout=cfg.dropout, n_extra=N_EXTRA if cfg.feats else 0,
                                      **({"kernel": cfg.kernel} if cfg.arch == "tcn" else {})))
        self.net.to(device)
        self.ema_net = copy.deepcopy(self.net) if cfg.ema > 0 else None
        self.n_updates = 0
        self.total_steps = total_steps
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        self.sched = torch.optim.lr_scheduler.OneCycleLR(self.opt, max_lr=cfg.lr, total_steps=total_steps,
                                                         pct_start=max(0.05, 3 / total_steps))
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
    if common.cache_train:
        train.enable_cache()
    if common.cache_eval:
        valid.enable_cache()
    train_ids, hold_ids = split_train(train.n_seqs, common.holdout, common.seed)
    if common.train_seqs:
        train_ids = rng.choice(train_ids, common.train_seqs, replace=False)
    valid_B = json.load(open(common.valid_split))["B"] if common.valid_split else None
    valid_ids = (valid_B or list(range(valid.n_seqs)))[:common.valid_seqs]
    hold_store = train
    if common.cache_eval and not common.cache_train:  # cache holdout without caching all of train
        hold_store = _store(common, "train")
        hold_store.enable_cache()

    n_batches = math.ceil(len(train_ids) / common.batch)
    exps = [Experiment(c, common, device, n_batches) for c in configs]
    teachers = {}  # path -> memmap, loaded once and shared by the models that use it
    for path in sorted({e.teacher_path for e in exps if e.teacher_path}):
        teachers[path] = np.load(path, mmap_mode="r")
        assert teachers[path].shape[:2] == (train.n_seqs, 20000), f"teacher {path}: shape {teachers[path].shape}"
    row_w, p_norm = None, {}
    if any(e.cfg.mask_weight > 0 for e in exps):
        assert common.row_weights, "mask_weight > 0 needs --row-weights (see pipeline/mask_model.py)"
        row_w = np.load(common.row_weights, mmap_mode="r")
        assert row_w.shape == (train.n_seqs, 20000), f"row weights shape {row_w.shape}"
        sample = np.asarray(row_w[np.sort(np.asarray(train_ids))[::20], 99:], dtype=np.float32)
        for pw in {e.cfg.mask_power for e in exps if e.cfg.mask_weight > 0}:
            p_norm[pw] = float((sample ** pw).mean())  # so the average row weight stays about 1
            print(f"row weights: mean p^{pw:g} = {p_norm[pw]:.4f}", flush=True)
    n_epochs = max(e.epochs for e in exps)
    t0 = time.time()
    for e in exps:
        e.log(event="start", config=asdict(e.cfg), common=asdict(common), device=str(device), params=e.params,
              cost=round(e.cost, 3), train_seqs=len(train_ids), holdout=len(hold_ids), steps=e.total_steps)
    eval_at = set(np.linspace(0, n_batches, common.evals_per_epoch + 1, dtype=int)[1:] - 1)

    def run_eval(epoch, bi, group):
        nets = [e.eval_net for e in group]
        vs = evaluate_many(nets, valid, valid_ids, device)
        hs = evaluate_many(nets, hold_store, hold_ids, device)
        for e, v, h in zip(group, vs, hs):
            # with a valid split: select on half B's scored rows (the mask model never saw them);
            # otherwise don't select on the visible valid mask alone
            sel = v["scored"] if valid_B else (v["scored"] + h["all"]) / 2
            is_best = sel > e.best
            if is_best:
                e.best = sel
                torch.save({"config": e.net.config, "state_dict": e.eval_net.state_dict(), "exp": asdict(e.cfg),
                            "n_folds": common.n_folds},
                           os.path.join(e.dir, "best.pt"))
            e.log(event="eval", epoch=epoch, batch=bi, valid=round(v["scored"], 6), valid_all=round(v["all"], 6),
                  holdout=round(h["all"], 6), select=round(sel, 6), best=is_best, time=round(time.time() - t0, 1))
            e.net.train()

    for epoch in range(n_epochs):
        active = [e for e in exps if epoch < e.epochs]
        for e in active:
            e.net.train()
        ids = np.asarray(rng.permutation(train_ids))
        for bi, (X, Y, _) in enumerate(train.batches(list(ids), common.batch)):
            tb = time.time()
            batch_ids = ids[bi * common.batch:(bi + 1) * common.batch]
            if any(e.cfg.feats for e in active):  # extras computed once per batch, shared by every model, and
                # joined on the GPU: the joined 256-seq batch would be another 3 GB array in host RAM
                Xd = torch.cat([torch.from_numpy(X).to(device), torch.from_numpy(extra_features(X)).to(device)], -1)
            else:
                Xd = torch.from_numpy(X).to(device, non_blocking=True)  # one host->device copy per batch
            Yd = torch.from_numpy(Y).to(device, non_blocking=True)
            order = torch.as_tensor(np.argsort(np.argsort(batch_ids)), device=device)  # memmap reads in file order
            Td = {path: torch.from_numpy(np.asarray(t[np.sort(batch_ids)], dtype=np.float32)).to(device)[order]
                  for path, t in teachers.items() if any(e.teacher_path == path for e in active)}
            Pd = None
            if row_w is not None and any(e.cfg.mask_weight > 0 for e in active):
                Pd = torch.from_numpy(np.asarray(row_w[np.sort(batch_ids)], dtype=np.float32)).to(device)[order]
            for e in active:
                Xe, Ye, Te = Xd, Yd, Td.get(e.teacher_path)
                We = None
                if e.cfg.mask_weight > 0:
                    lam, pw = e.cfg.mask_weight, e.cfg.mask_power
                    We = (1 - lam) + lam * Pd.clamp(min=0) ** pw / p_norm[pw]
                if e.cfg.fold is not None:  # out-of-fold model: drop its fold's sequences from the batch
                    rows = torch.as_tensor(np.flatnonzero(batch_ids % common.n_folds != e.cfg.fold), device=device)
                    Xe, Ye = Xd[rows], Yd[rows]
                    Te = Te[rows] if Te is not None else None
                    We = We[rows] if We is not None else None
                losses = []
                for m0 in range(0, len(Xe), e.micro):  # mini-batches of whole sequences, each its own TBPTT pass
                    Xm, Ym = Xe[m0:m0 + e.micro], Ye[m0:m0 + e.micro]
                    Tm = Te[m0:m0 + e.micro] if Te is not None else None
                    Wm = We[m0:m0 + e.micro] if We is not None else None
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
            del Xd, Yd, Td, Pd
            if bi in eval_at:
                run_eval(epoch, bi, active)

    # final: best checkpoints -> full validation + holdout, ONNX submission folders, summary table
    for e in exps:
        ck = torch.load(os.path.join(e.dir, "best.pt"), map_location="cpu", weights_only=False)
        e.net.load_state_dict(ck["state_dict"])
        e.net.to(device)
    nets = [e.net for e in exps]
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
                     **({"valid_B_scored": round(b["scored"], 6), "valid_B_all": round(b["all"], 6)} if b else {})})
    rows.sort(key=lambda r: -r["valid_B_scored"] if "valid_B_scored" in r else -(r["valid_scored"] + r["holdout_all"]))
    with open(os.path.join(common.out_dir, "summary.jsonl"), "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("\n" + " | ".join(rows[0].keys()))
    for r in rows:
        print(" | ".join(str(v) for v in r.values()))
    return rows
