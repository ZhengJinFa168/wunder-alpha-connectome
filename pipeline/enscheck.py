"""Per-group ensembles (prediction averages) of models fine-tuned on validation, scored honestly by cross-fitting.

    python -m pipeline.enscheck score --valid-split split.json --members members.json --out ens.json
    python -m pipeline.enscheck export --route route.json --fallback model.onnx --out DIR

members.json: {"1": {"current": NAME, "members": {NAME: SPEC}}, ..., "4": {...}}  (groups 1..4)
  SPEC {"pair": [A_best.pt, B_best.pt], "alpha": a}: a fine-tune on half A / half B (A's model predicts half B and
       vice versa), weights (1 - a) * its init + a * fine-tune (a = 0.9 as submitted)
  SPEC {"fixed": best.pt}: a model not trained on validation, predicts everything
  SPEC {"wise": [shared best.pt, fine-tune best.pt, w]}: weights (1 - w) * shared + w * fine-tune (sweep 10a's ~wise)
Per group: every single member, current + each other member (2-model average), and a greedy ensemble from the current
model (adds the member that helps most, up to --max-size models). Greedy picks on all of the group's validation
sequences are optimistic, so it is also run honestly: picked on half A, scored on half B and vice versa
("honest gain" = mean over the two folds of ensemble - current on the held-out half). Scored rows, weighted Pearson.

route.json: [[SPEC, ...] for groups 1..4] with SPEC {"ckpt": best.pt, "alpha": a} | {"wise": [...]} | {"onnx": path}
(a group's list of several specs -> their average, one ONNX graph; an "onnx" spec must be alone).
"""
import argparse
import json
import os
import shutil

import numpy as np
import torch

from .crossfit import HERE, hedged
from .data import SeqStore
from .fast_eval import load_ckpt, pick_device, predict_batch
from .groups import label as group_label
from .model import EnsembleNet, export_onnx
from .routed import interpolate, wp


def build(spec):
    """A fixed-weights net from an export / fixed / wise spec."""
    if "ckpt" in spec:
        return hedged(spec["ckpt"], spec.get("alpha", 1.0))
    if "fixed" in spec:
        return load_ckpt(spec["fixed"]).eval()
    p0, p1, w = spec["wise"]
    net = interpolate(load_ckpt(p0).eval(), load_ckpt(p1).eval(), w)
    assert net is not None, f"wise {p0} / {p1}: shapes differ"
    return net.eval()


def greedy(score, current, names, max_size):
    """Forward selection from [current]: add the member that raises score(list) most, while it rises."""
    ens, best = [current], score([current])
    steps = [(list(ens), best)]
    while len(ens) < max_size:
        cand = [(score(ens + [n]), n) for n in names if n not in ens]
        if not cand:
            break
        s, n = max(cand)
        if s <= best:
            break
        ens, best = ens + [n], s
        steps.append((list(ens), best))
    return steps


def score(args):
    split = json.load(open(args.valid_split))
    spec = json.load(open(args.members))
    store = SeqStore("datasets/valid.parquet", args.threads)
    n = store.n_seqs
    half = np.full(n, -1)
    half[split["A"]] = 0
    half[split["B"]] = 1
    device = pick_device(args.device)
    # nets[(g, name)] = (net predicting half A, net predicting half B); the same net twice for fixed / wise members
    nets = {}
    for gs, gspec in spec.items():
        g = int(gs) - 1
        assert gspec["current"] in gspec["members"], f"group {gs}: current {gspec['current']} not among the members"
        for name, m in gspec["members"].items():
            if "pair" in m:
                pa, pb = m["pair"]
                ea = torch.load(pa, map_location="cpu", weights_only=False).get("exp") or {}
                assert ea.get("valid_part") == "A", f"{pa}: not a half-A model"
                nets[(g, name)] = (hedged(pb, m.get("alpha", 1.0)).to(device), hedged(pa, m.get("alpha", 1.0)).to(device))
            else:
                net = build(m).to(device)
                nets[(g, name)] = (net, net)
    print(f"{len(nets)} members over {len(spec)} groups", flush=True)
    Y, G = {}, np.zeros(n, int)
    P = {k: {} for k in nets}
    ids = np.sort(np.flatnonzero(half >= 0))
    for b0, (X, Yb, M) in zip(range(0, len(ids), 128), store.batches(list(ids), 128)):
        seqs = ids[b0:b0 + len(X)]
        G[seqs] = group_label(Yb)
        for b, s in enumerate(seqs):
            Y[s] = Yb[b][M[b]]
        for (g, name), pair in nets.items():
            for h in (0, 1):
                sel = np.flatnonzero((G[seqs] == g) & (half[seqs] == h))
                if len(sel):
                    Pb = predict_batch(pair[h], X[sel], device)
                    for j, b in enumerate(sel):
                        P[(g, name)][seqs[b]] = Pb[j][M[b]]
        if (b0 // 128) % 4 == 3 or b0 + len(X) == len(ids):
            print(f"{b0 + len(X)}/{len(ids)} sequences", flush=True)

    out = {}
    for gs, gspec in spec.items():
        g = int(gs) - 1
        names, cur = list(gspec["members"]), gspec["current"]
        seqs_h = [ids[(G[ids] == g) & (half[ids] == h)] for h in (0, 1)]
        seqs_all = np.concatenate(seqs_h)
        yh = [np.concatenate([Y[s] for s in sq]) if len(sq) else np.zeros((0, 2)) for sq in seqs_h]
        ph = {nm: [np.concatenate([P[(g, nm)][s] for s in sq]) if len(sq) else np.zeros((0, 2)) for sq in seqs_h] for nm in names}

        def sc(ens, hs=(0, 1)):
            y = np.concatenate([yh[h] for h in hs])
            p = np.concatenate([np.mean([ph[nm][h] for nm in ens], 0) for h in hs])
            return wp(y, p) if len(y) else float("nan")

        base = sc([cur])
        singles = {nm: sc([nm]) for nm in names}
        pairs = {nm: sc([cur, nm]) for nm in names if nm != cur}
        steps = greedy(sc, cur, names, args.max_size)
        honest = []
        for pick, test in ((0, 1), (1, 0)):
            st = greedy(lambda e: sc(e, (pick,)), cur, names, args.max_size)
            honest.append({"picked_on": "AB"[pick], "ensemble": st[-1][0],
                           "gain_on_other_half": sc(st[-1][0], (test,)) - sc([cur], (test,))})
        hg = float(np.mean([r["gain_on_other_half"] for r in honest]))
        out[gs] = {"current": cur, "current_score": base, "n_seqs": int(len(seqs_all)), "singles": singles,
                   "pairs_with_current": pairs, "greedy": [{"ensemble": e, "score": s} for e, s in steps],
                   "honest": honest, "honest_gain": hg}
        print(f"\ngroup {gs} ({len(seqs_all)} validation sequences): current {cur} {base:.4f}", flush=True)
        print("  singles:       " + "  ".join(f"{nm} {s - base:+.4f}" for nm, s in sorted(singles.items(), key=lambda t: -t[1])), flush=True)
        print("  current + one: " + "  ".join(f"{nm} {s - base:+.4f}" for nm, s in sorted(pairs.items(), key=lambda t: -t[1])), flush=True)
        for e, s in steps:
            print(f"  greedy {len(e)}: {' + '.join(e)}  {s:.4f} ({s - base:+.4f})", flush=True)
        for r in honest:
            print(f"  honest: picked on half {r['picked_on']} -> {' + '.join(r['ensemble'])}, on the other half "
                  f"{r['gain_on_other_half']:+.4f}", flush=True)
        print(f"  honest gain (mean of the two folds): {hg:+.4f}", flush=True)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1, default=float)


def export(args):
    route = json.load(open(args.route))
    assert len(route) == 4, "route: one list of specs per group 1..4"
    os.makedirs(args.out, exist_ok=True)
    files, desc = [], {}
    for g, specs in enumerate(route):
        f = f"model_{g}.onnx"
        if any("onnx" in s for s in specs):
            assert len(specs) == 1, f"group {g + 1}: an onnx route must be alone"
            shutil.copy(specs[0]["onnx"], os.path.join(args.out, f))
        else:
            mem = [build(s) for s in specs]
            net = mem[0] if len(mem) == 1 else EnsembleNet(mem, np.ones((len(mem), 2)))
            export_onnx(net.eval(), os.path.join(args.out, f))
        files.append(f)
        desc[f] = specs
    files.append(f"model_4.onnx")
    shutil.copy(args.fallback, os.path.join(args.out, files[-1]))
    desc[files[-1]] = [{"onnx": args.fallback, "role": "fallback"}]
    json.dump({"models": files, "route": [0, 1, 2, 3], "fallback": 4, "threshold": args.threshold},
              open(os.path.join(args.out, "routes.json"), "w"), indent=1)
    shutil.copy(os.path.join(HERE, "solution_routed.py"), os.path.join(args.out, "solution.py"))
    shutil.copy(os.path.join(HERE, "solution_template.py"), os.path.join(args.out, "wnn_runner.py"))
    shutil.copy(os.path.join(HERE, "features.py"), os.path.join(args.out, "wnn_features.py"))
    json.dump({"models": desc}, open(os.path.join(args.out, "routing.json"), "w"), indent=1)
    print(f"exported -> {args.out}: " + ", ".join(f"g{g + 1} {len(s)} model(s)" for g, s in enumerate(route)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--valid-split", required=True)
    s.add_argument("--members", required=True)
    s.add_argument("--max-size", type=int, default=3)
    s.add_argument("--out")
    s.add_argument("--device", default="auto")
    s.add_argument("--threads", type=int, default=8)
    e = sub.add_parser("export")
    e.add_argument("--route", required=True)
    e.add_argument("--fallback", required=True)
    e.add_argument("--threshold", type=float, default=0.6)
    e.add_argument("--out", required=True)
    args = ap.parse_args()
    score(args) if args.cmd == "score" else export(args)


if __name__ == "__main__":
    main()
