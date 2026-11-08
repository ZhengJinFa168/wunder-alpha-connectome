"""Batched evaluation: runs whole sequences through the network at once (seconds instead of
the ~15 min row-by-row loop) and scores them with the official GlobalAccumulator.

    python -m pipeline.fast_eval --baseline                 # should reproduce 0.617052
    python -m pipeline.fast_eval --ckpt runs/NAME/best.pt --holdout 500
"""
import argparse
import json
import time

import numpy as np
import torch

from utils import SEQUENCE_LENGTH, WARMUP, GlobalAccumulator

from .data import SeqStore, split_train
from .groups import label as group_label
from .model import build_net, feat_kind, load_baseline

NEED = np.arange(SEQUENCE_LENGTH) >= WARMUP


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def with_extras(X, kind="act"):
    """Append the extra causal features of `kind` (pipeline/features.py) to raw inputs [B,T,112]."""
    from .features import offline
    return np.concatenate([X, offline(X[..., :112], kind)], -1)


def net_inputs(net, X, cache):
    """Raw batch X [B,T,112] -> what `net` takes; `cache` (a dict per batch) shares the extras between models."""
    kind = feat_kind(net)
    if kind and kind not in cache:
        cache[kind] = with_extras(X, kind)
    return cache[kind] if kind else X


@torch.no_grad()
def predict_batch(net, X, device, chunk=2000):
    if X.shape[-1] == 112:
        X = net_inputs(net, X, {})
    B, T, _ = X.shape
    out = np.empty((B, T, 2), np.float32)
    h = xp = None
    for s in range(0, T, chunk):
        x = torch.from_numpy(X[:, s:s + chunk]).to(device)
        p, h = net(x, h, xp)
        xp = x[:, -1]
        out[:, s:s + chunk] = p.float().cpu().numpy()
    return out


def evaluate_many(nets, store: SeqStore, ids, device, batch_size=128):
    """Score several models in one pass over the data. Per model:
    {'scored': WP on scored rows, 'all': WP on all required rows, ...} (equal without a mask)."""
    for net in nets:
        net.eval()
    accs = [(GlobalAccumulator(), GlobalAccumulator()) for _ in nets]
    acc_g = [[GlobalAccumulator() for _ in range(4)] for _ in nets]  # scored rows per instrument group
    for X, Y, M in store.batches(list(ids), batch_size):
        cache = {}  # extras once per batch and kind, shared by every model that uses them
        groups = group_label(Y)
        for net, (acc, acc_all), ag in zip(nets, accs, acc_g):
            P = predict_batch(net, net_inputs(net, X, cache), device)
            for b in range(len(X)):
                acc.add(Y[b], P[b], M[b])
                ag[groups[b]].add(Y[b], P[b], M[b])
                if store.has_mask:
                    acc_all.add(Y[b], P[b], NEED)
    out = []
    for (acc, acc_all), ag in zip(accs, acc_g):
        r = acc.result()
        res = {"scored": r["weighted_pearson"], "t0": r["t0"], "t1": r["t1"], "n_seqs": r["blocks"]}
        for g, a in enumerate(ag):  # nan when the eval set has no sequence of that group
            res[f"scored_g{g}"] = a.result()["weighted_pearson"] if a.selected_rows else float("nan")
        res["all"] = acc_all.result()["weighted_pearson"] if store.has_mask else res["scored"]
        out.append(res)
    return out


def evaluate(net, store: SeqStore, ids, device, batch_size=128):
    return evaluate_many([net], store, ids, device, batch_size)[0]


def load_ckpt(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = build_net(ck["config"])
    net.load_state_dict(ck["state_dict"])
    return net


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--baseline", action="store_true")
    g.add_argument("--ckpt")
    ap.add_argument("--valid-seqs", type=int, default=None, help="first N validation seqs (default all)")
    ap.add_argument("--holdout", type=int, default=0, help="also score the N-seq train holdout")
    ap.add_argument("--valid-split", help="split.json from mask_model: score validation half B instead")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    net = (load_baseline("baseline/baseline.onnx") if args.baseline else load_ckpt(args.ckpt)).to(device)
    out = {}
    t = time.time()
    valid = SeqStore("datasets/valid.parquet")
    ids = json.load(open(args.valid_split))["B"] if args.valid_split else list(range(valid.n_seqs))
    out["valid_B" if args.valid_split else "valid"] = evaluate(net, valid, ids[:args.valid_seqs or len(ids)], device)
    if args.holdout:
        train = SeqStore("datasets/train.parquet")
        _, hold = split_train(train.n_seqs, args.holdout)
        out["holdout"] = evaluate(net, train, hold, device)
    out["seconds"] = round(time.time() - t, 1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
