"""Score models alone and averaged, pick an ensemble on the train holdout, optionally export it.

    python -m pipeline.ensemble --ckpts "runs/*/best.pt" --budgets 1 1.5 2 --export runs/ensembles

Every model is run once over the holdout and validation data (shared loading); combinations are then
scored from the stored predictions. Members are rescaled to a common prediction spread before
averaging (scales measured on the holdout). Selection uses the holdout only, so the validation
numbers stay an honest check of the chosen combination.

Cost = multiply-adds per row relative to the baseline (models.macs). Real run time also has a
fixed per-row overhead, so an ensemble's measured time ratio is lower than its cost; check
candidates with pipeline.speed. One greedy ensemble is built and exported per --budgets value.
"""
import argparse
import glob
import json
import os
import shutil
import time

import numpy as np

from utils import SEQUENCE_LENGTH, WARMUP, weighted_pearson

from .data import SeqStore, split_train
from .fast_eval import load_ckpt, pick_device, predict_batch, with_extras
from .model import BASELINE_MACS, EnsembleNet, export_onnx, load_baseline, macs, needs_extra

HERE = os.path.dirname(os.path.abspath(__file__))
NEED = np.arange(SEQUENCE_LENGTH) >= WARMUP


def load_member(path):
    if path == "baseline":
        return "baseline", load_baseline("baseline/baseline.onnx")
    return os.path.basename(os.path.dirname(path)), load_ckpt(path)


def collect(nets, store, ids, device, masks):
    """-> Y {mask: [N,2]}, P {mask: [K,N,2]} for masks 'all' (required rows) and/or 'scored'."""
    ys = {m: [] for m in masks}
    ps = {m: [[] for _ in nets] for m in masks}
    extra = any(needs_extra(n) for n in nets)
    for X, Y, M in store.batches(list(ids), 128):
        X = with_extras(X) if extra else X
        sel = {"all": np.broadcast_to(NEED, M.shape), "scored": M}
        for m in masks:
            ys[m].append(Y[sel[m]])
        for k, net in enumerate(nets):
            P = predict_batch(net, X, device)
            for m in masks:
                ps[m][k].append(P[sel[m]])
    return ({m: np.concatenate(ys[m]) for m in masks},
            {m: np.stack([np.concatenate(p) for p in ps[m]]) for m in masks})


def wp(y, p):
    return float(np.mean([weighted_pearson(y[:, j], p[:, j]) for j in range(2)]))


def combine(P, scales, idx):
    return (P[idx] * scales[idx][:, None, :]).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="best.pt paths/globs, or 'baseline'")
    ap.add_argument("--holdout", type=int, default=500)
    ap.add_argument("--valid-seqs", type=int, default=None, help="first N validation seqs (default all)")
    ap.add_argument("--max-members", type=int, default=5)
    ap.add_argument("--budgets", type=float, nargs="+", default=[99.0],
                    help="max total cost (x baseline multiply-adds); one greedy ensemble per budget")
    ap.add_argument("--export", help="root folder: writes <root>/cost<budget>/submission per budget")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--valid-split", help="split.json from mask_model: report on validation half B")
    ap.add_argument("--select-on", choices=["holdout", "validA"], default="holdout",
                    help="what greedy selection maximises: holdout all-rows WP, or validation half A scored rows "
                         "(needs --valid-split; right for mask-weighted models, keeps half B honest)")
    args = ap.parse_args()
    assert args.select_on == "holdout" or args.valid_split, "--select-on validA needs --valid-split"

    paths = []
    for c in args.ckpts:
        paths += [c] if c == "baseline" else sorted(glob.glob(c))
    names, nets = zip(*[load_member(p) for p in paths])
    device = pick_device(args.device)
    nets = [n.eval().to(device) for n in nets]
    print(f"{len(nets)} models: {', '.join(names)}", flush=True)

    t = time.time()
    train = SeqStore("datasets/train.parquet", args.threads)
    valid = SeqStore("datasets/valid.parquet", args.threads)
    _, hold_ids = split_train(train.n_seqs, args.holdout)
    Yh, Ph = collect(nets, train, hold_ids, device, ["all"])
    if args.valid_split:
        split = json.load(open(args.valid_split))
        Ya, Pa = collect(nets, valid, split["A"], device, ["scored"])
        Yv, Pv = collect(nets, valid, split["B"], device, ["scored", "all"])  # report on half B
    else:
        Yv, Pv = collect(nets, valid, range(args.valid_seqs or valid.n_seqs), device, ["scored", "all"])
    print(f"predictions collected in {time.time() - t:.0f}s", flush=True)
    SEL = args.select_on

    sd = Ph["all"].std(axis=1)  # [K, 2]
    scales = sd.mean(0, keepdims=True) / sd

    def sel_score(idx):
        if SEL == "validA":
            return wp(Ya["scored"], combine(Pa["scored"], scales, idx))
        return wp(Yh["all"], combine(Ph["all"], scales, idx))

    def score(idx):  # valid_* = half B when --valid-split is given
        out = {"holdout": wp(Yh["all"], combine(Ph["all"], scales, idx)),
               "valid_scored": wp(Yv["scored"], combine(Pv["scored"], scales, idx)),
               "valid_all": wp(Yv["all"], combine(Pv["all"], scales, idx))}
        if args.valid_split:
            out["validA"] = wp(Ya["scored"], combine(Pa["scored"], scales, idx))
        return out

    cost = np.array([macs(n.config) / BASELINE_MACS for n in nets])
    rows = []
    singles = [(k, score([k])) for k in range(len(nets))]
    for k, sc in singles:
        rows.append({"combo": names[k], "n": 1, "cost": cost[k], **sc})
    order = [k for k, sc in sorted(singles, key=lambda ks: -ks[1][SEL])]
    for top in range(2, min(len(nets), args.max_members) + 1):
        rows.append({"combo": f"top{top} by {SEL}", "n": top, "cost": cost[order[:top]].sum(),
                     "members": [names[k] for k in order[:top]], **score(order[:top])})

    def greedy(budget):
        """Forward selection on SEL (no repeats) within a total cost budget."""
        ok = [k for k in order if cost[k] <= budget]
        if not ok:
            return []
        chosen, best = [ok[0]], singles[ok[0]][1][SEL]
        while len(chosen) < args.max_members:
            cand = [(k, sel_score(chosen + [k])) for k in range(len(nets))
                    if k not in chosen and cost[chosen + [k]].sum() <= budget]
            if not cand:
                break
            k, sc = max(cand, key=lambda c: c[1])
            if sc <= best:
                break
            chosen.append(k)
            best = sc
        return chosen

    picks = {}
    for b in args.budgets:
        chosen = greedy(b)
        if chosen:
            picks[b] = chosen
            rows.append({"combo": f"greedy, cost <= {b:g}", "n": len(chosen), "cost": cost[chosen].sum(),
                         "members": [names[k] for k in chosen], **score(chosen)})

    vl = "B" if args.valid_split else ""
    # diversity: members that disagree with the best single model are the ones an average (or a teacher) gains from
    best_k = order[0]
    print(f"\nprediction correlation with {names[best_k]} (valid{vl} scored rows, mean of t0/t1):")
    for k in order:
        r = np.mean([np.corrcoef(Pv["scored"][k][:, j], Pv["scored"][best_k][:, j])[0, 1] for j in range(2)])
        print(f"  {names[k][:40]:40s} {r:.4f}")
    print(f"\n{'combo':30s} {'n':>2s} {'cost':>5s} {'holdout':>8s} {'valid'+vl+'_sc':>9s} {'valid'+vl+'_all':>10s}"
          + (f" {'validA_sc':>9s}" if args.valid_split else ""))
    for r in rows:
        print(f"{r['combo'][:30]:30s} {r['n']:2d} {r['cost']:5.2f} {r['holdout']:8.5f} {r['valid_scored']:9.5f} "
              f"{r['valid_all']:10.5f}" + (f" {r['validA']:9.5f}" if "validA" in r else "")
              + (f"   {'+'.join(r['members'])}" if "members" in r else ""))

    if args.export:
        for b, chosen in picks.items():
            out = os.path.join(args.export, f"cost{b:g}", "submission")
            os.makedirs(out, exist_ok=True)
            net = nets[chosen[0]] if len(chosen) == 1 else EnsembleNet([nets[k] for k in chosen], scales[chosen])
            export_onnx(net, os.path.join(out, "model.onnx"))
            shutil.copy(os.path.join(HERE, "solution_template.py"), os.path.join(out, "solution.py"))
            shutil.copy(os.path.join(HERE, "features.py"), os.path.join(out, "wnn_features.py"))
            row = next(r for r in rows if r["combo"] == f"greedy, cost <= {b:g}")
            json.dump({"members": [paths[k] for k in chosen], "scales": scales[chosen].tolist(),
                       "scores": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}},
                      open(os.path.join(out, "ensemble.json"), "w"), indent=1)
            print(f"exported cost<={b:g}: {len(chosen)} model(s) -> {out}")

if __name__ == "__main__":
    main()
