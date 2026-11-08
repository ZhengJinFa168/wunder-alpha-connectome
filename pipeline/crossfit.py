"""Honest scores for models trained on validation: cross-fitting over the two halves of split.json.

    python -m pipeline.crossfit score --valid-split split.json --base runs/sweep9/s9_h128/best.pt \
        --base-spec runs/sweep10a/ft_g1/best.pt runs/sweep10a/ft_g2/best.pt runs/sweep10a/ft_g3/best.pt \
        --recipes "runs/sweep11_ft/*_A/best.pt" --hedge 0.9 --out crossfit.json
    python -m pipeline.crossfit export --route g1.pt g2.pt g3.pt g4.pt --fallback g4.pt --alpha 0.9 --out DIR

score: a recipe is a pair of models NAME_A / NAME_B (ExpConfig valid_part A / B; the glob names the _A ones). The
model trained on half A predicts half B and vice versa, so every validation sequence gets an honest prediction:
  A, B   = WP on half A (from the B model) / half B (from the A model), scored rows;
  all    = pooled WP over all of valid with those cross-fitted predictions; g1..g4 the same per instrument group.
The base models (not trained on valid) predict everything. Recipes trained on one group (ExpConfig group) only count
for that group. --hedge a: also each recipe's weight interpolation (1 - a) * init + a * model (its ExpConfig init).
Routed combinations use the true groups (the detector gives the same scores, sweep 10a).

export: a routed submission (pipeline/solution_routed.py) from given models per group, optionally each one's weights
interpolated towards its init (--alpha < 1).
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np
import torch

from .data import SeqStore
from .fast_eval import load_ckpt, net_inputs, pick_device, predict_batch
from .groups import label as group_label
from .model import export_onnx
from .routed import interpolate, wp

HERE = os.path.dirname(os.path.abspath(__file__))


def exp_of(path):
    return torch.load(path, map_location="cpu", weights_only=False).get("exp") or {}


def hedged(path, alpha):
    """The model at `path`, or its weights interpolated towards its init: (1 - alpha) * init + alpha * model."""
    net = load_ckpt(path).eval()
    if alpha >= 1:
        return net
    init = exp_of(path).get("init")
    assert init and os.path.exists(init), f"{path}: no init checkpoint to interpolate towards ({init})"
    mid = interpolate(load_ckpt(init).eval(), net, alpha)
    assert mid is not None, f"{path}: shape differs from its init"
    return mid.eval()


def score(args):
    split = json.load(open(args.valid_split))
    store = SeqStore("datasets/valid.parquet", args.threads)
    n = store.n_seqs
    half = np.full(n, -1)
    half[split["A"]] = 0
    half[split["B"]] = 1
    device = pick_device(args.device)

    # models: (name, net, which half it predicts: None = all, group or None)
    models = [("s9 (base)", load_ckpt(args.base).eval(), None, None)]
    for g, p in enumerate(args.base_spec or []):
        models.append((f"spec_g{g + 1} (base)", load_ckpt(p).eval(), None, g))
    refs = [p for c in (args.refs or []) for p in sorted(glob.glob(c))]
    for p in refs:  # more models not trained on valid (e.g. the controls' reference)
        models.append((os.path.basename(os.path.dirname(p)) + " (ref)", load_ckpt(p).eval(), None, None))
    recipes = []  # name, group, index of the A model, index of the B model
    a_paths = [p for c in args.recipes for p in sorted(glob.glob(c))]
    for pa in a_paths:
        name = os.path.basename(os.path.dirname(pa))
        assert name.endswith("_A"), f"{pa}: recipes are named NAME_A / NAME_B"
        pb = os.path.join(os.path.dirname(os.path.dirname(pa)), name[:-2] + "_B", "best.pt")
        if not os.path.exists(pb):
            print(f"skipping {name[:-2]}: no {pb}", flush=True)
            continue
        ea, eb = exp_of(pa), exp_of(pb)
        assert ea.get("valid_part", "A") in ("A", None) and eb.get("valid_part", "B") in ("B", None), \
            f"{name}: valid_part {ea.get('valid_part')} / {eb.get('valid_part')}"  # None: uses the half otherwise (seq_weights)
        g = ea.get("group")
        variants = [(name[:-2], 1.0)] + ([(f"{name[:-2]}~{args.hedge:g}", args.hedge)] if args.hedge and ea.get("init") else [])
        for rname, alpha in variants:
            ka = len(models)
            models.append((rname + "_A", hedged(pa, alpha), 1, g))  # trained on A -> predicts B
            models.append((rname + "_B", hedged(pb, alpha), 0, g))
            recipes.append({"name": rname, "group": g, "A_model": ka, "B_model": ka + 1, "alpha": alpha,
                            "kind": "control" if not ea.get("init") else "fine-tune",
                            "paths": [pa, pb]})
    nets = [m[1].to(device) for m in models]
    print(f"{len(recipes)} recipes, {len(models)} models", flush=True)

    Y, G = [None] * n, np.zeros(n, int)
    P = [dict() for _ in models]  # model -> seq -> scored-row predictions
    ids = np.sort(np.flatnonzero(half >= 0))  # the two halves (= all of valid)
    for b0, (X, Yb, M) in zip(range(0, len(ids), 128), store.batches(list(ids), 128)):
        seqs = ids[b0:b0 + len(X)]
        G[seqs] = group_label(Yb)
        for b, s in enumerate(seqs):
            Y[s] = Yb[b][M[b]]
        cache = {}
        for k, (name, _, pred_half, g) in enumerate(models):
            sel = np.ones(len(seqs), bool)
            if pred_half is not None:
                sel &= half[seqs] == pred_half
            if g is not None:
                sel &= G[seqs] == g
            if not sel.any():
                continue
            Pb = predict_batch(nets[k], net_inputs(nets[k], X, cache)[sel], device)
            for j, b in enumerate(np.flatnonzero(sel)):  # Pb holds only the selected sequences
                P[k][seqs[b]] = Pb[j][M[b]]
        if (b0 // 128) % 4 == 3 or b0 + len(X) == len(ids):
            print(f"{b0 + len(X)}/{len(ids)} sequences", flush=True)

    def pooled(assign, seqs):
        """assign: seq -> model index. Pooled scored-row WP over seqs."""
        seqs = [s for s in seqs if len(Y[s])]
        if not seqs:
            return float("nan")
        return wp(np.concatenate([Y[s] for s in seqs]), np.concatenate([P[assign(s)][s] for s in seqs]))

    A, B = np.flatnonzero(half == 0), np.flatnonzero(half == 1)
    ALL = np.concatenate([A, B])
    by_g = {g: ALL[G[ALL] == g] for g in range(4)}

    def row(assign, groups=range(4)):
        """Scores of a per-sequence assignment; restricted to `groups` (for one-group recipes)."""
        sub = lambda seqs: [s for s in seqs if G[s] in groups]
        return {"A": pooled(assign, sub(A)), "B": pooled(assign, sub(B)), "all": pooled(assign, sub(ALL)),
                **{f"g{g + 1}": pooled(assign, by_g[g]) if g in groups else float("nan") for g in range(4)}}

    res = {"s9 (base)": row(lambda s: 0)}
    spec_idx = {models[k][3]: k for k in range(1, 1 + len(args.base_spec or []))}
    if spec_idx:
        res["base routed (spec g1-3, s9 g4)"] = row(lambda s: spec_idx.get(G[s], 0))
    for k, (name, _, _, _) in enumerate(models):
        if name.endswith(" (ref)"):
            res[name] = row(lambda s, k=k: k)
    cross = {}
    for r in recipes:
        assign = lambda s, r=r: r["B_model"] if half[s] == 0 else r["A_model"]
        cross[r["name"]] = assign
        res[r["name"]] = row(assign, range(4) if r["group"] is None else [r["group"]])

    # best shared fine-tune recipe (all groups, no hedge) on the cross-fitted pooled score; then per group 1-3 the
    # better of it and the group recipes, on that group's cross-fitted score
    shared = [r for r in recipes if r["group"] is None and r["alpha"] == 1 and r["kind"] == "fine-tune"]
    choice = {}
    if shared:
        best = max(shared, key=lambda r: res[r["name"]]["all"])
        route = [best["name"]] * 4
        for g in range(3):
            opts = [r["name"] for r in recipes if r["group"] == g and r["alpha"] == 1] + [best["name"]]
            route[g] = max(opts, key=lambda nm: np.nan_to_num(res[nm][f"g{g + 1}"], nan=-1))
        res["FT routed: " + " / ".join(route)] = row(lambda s: cross[route[G[s]]](s))
        hedge_route = [f"{nm}~{args.hedge:g}" for nm in route] if args.hedge else None
        if hedge_route and all(nm in cross for nm in hedge_route):
            res[f"FT routed, hedge {args.hedge:g}"] = row(lambda s: cross[hedge_route[G[s]]](s))
        choice = {"best_shared": best["name"], "route": route, "hedge_route": hedge_route,
                  "recipes": {r["name"]: r["paths"] for r in recipes}}

    base = res["s9 (base)"]
    print(f"\ncross-fitted scored-row WP (A: from the model trained on B, B: from the model trained on A; "
          f"all = pooled over all valid). d_all = all - s9 base")
    print(f"{'model / recipe':44s} {'A':>8s} {'B':>8s} {'all':>8s} {'d_all':>8s} " + " ".join(f"{'g' + str(g + 1):>7s}" for g in range(4)))
    for name, r in res.items():
        d = r["all"] - base["all"] if r["all"] == r["all"] else float("nan")
        print(f"{name[:44]:44s} {r['A']:8.5f} {r['B']:8.5f} {r['all']:8.5f} {d:+8.5f} " +
              " ".join(f"{r[f'g{g + 1}']:7.4f}" for g in range(4)), flush=True)
    if choice:
        print(f"\nchoice: best shared fine-tune {choice['best_shared']}; route g1..g4 {choice['route']}", flush=True)
    if args.out:
        json.dump({"scores": res, "choice": choice}, open(args.out, "w"), indent=1, default=float)


def export(args):
    paths = list(dict.fromkeys(args.route + [args.fallback]))  # unique, in order
    os.makedirs(args.out, exist_ok=True)
    files = []
    for k, p in enumerate(paths):
        files.append(f"model_{k}.onnx")
        if p.endswith(".onnx"):  # an already exported model (e.g. from an earlier submission): copied as it is
            shutil.copy(p, os.path.join(args.out, files[-1]))
        else:
            export_onnx(hedged(p, args.alpha), os.path.join(args.out, files[-1]))
    routes = {"models": files, "route": [paths.index(p) for p in args.route], "fallback": paths.index(args.fallback),
              "threshold": args.threshold}
    json.dump(routes, open(os.path.join(args.out, "routes.json"), "w"), indent=1)
    shutil.copy(os.path.join(HERE, "solution_routed.py"), os.path.join(args.out, "solution.py"))
    shutil.copy(os.path.join(HERE, "solution_template.py"), os.path.join(args.out, "wnn_runner.py"))
    shutil.copy(os.path.join(HERE, "features.py"), os.path.join(args.out, "wnn_features.py"))
    json.dump({"models": dict(zip(files, paths)), "alpha": args.alpha, "route": args.route, "fallback": args.fallback},
              open(os.path.join(args.out, "routing.json"), "w"), indent=1)
    print(f"exported {len(files)} models -> {args.out} (route {routes['route']}, fallback {routes['fallback']}, "
          f"alpha {args.alpha:g})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--valid-split", required=True)
    s.add_argument("--base", required=True, help="the shared model not trained on valid (s9_h128)")
    s.add_argument("--base-spec", nargs="*", help="group 1..3 specialists not trained on valid, in group order")
    s.add_argument("--recipes", nargs="+", required=True, help="globs of the NAME_A best.pt files")
    s.add_argument("--refs", nargs="*", help="more models not trained on valid, scored on everything")
    s.add_argument("--hedge", type=float, default=0.0, help="also score (1 - a) * init + a * model")
    s.add_argument("--out")
    s.add_argument("--device", default="auto")
    s.add_argument("--threads", type=int, default=8)
    e = sub.add_parser("export")
    e.add_argument("--route", nargs=4, required=True, help="best.pt (hedged by --alpha) or .onnx (as is) for groups 1..4")
    e.add_argument("--fallback", required=True)
    e.add_argument("--alpha", type=float, default=1.0)
    e.add_argument("--threshold", type=float, default=0.6)
    e.add_argument("--out", required=True)
    args = ap.parse_args()
    score(args) if args.cmd == "score" else export(args)


if __name__ == "__main__":
    main()
