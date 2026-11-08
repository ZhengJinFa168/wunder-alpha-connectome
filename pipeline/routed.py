"""One model per instrument group: score routings on validation and export a routed submission.

    python -m pipeline.routed --shared runs/sweep9/s9_h128/best.pt --cands "runs/sweep10a/*/best.pt" \
        --valid-split runs/mask_model/split.json --export runs/routed_10a

Every model is run once over all validation sequences. Candidates trained on one group (their saved ExpConfig
has `group`) compete for that group only; the others (and the shared model) for every group. Besides the single
models, each group can take:
  * a blend: the average of the shared model and a candidate (2 models on that group's sequences; costs add up);
  * a weight interpolation ("~wise"): weights halfway between the shared model and a fine-tune of it (candidates
    whose ExpConfig `init` is the shared model; one model's cost).
Per group the route takes the option with the best scored-row WP on that group's half-A sequences (within
--max-cost per sequence), so the half-B numbers stay an honest check. Routed scores use the detector's groups
(features.detect, as solution.py does); the shared model takes sequences where the detector is less sure than
--threshold. Also reported (not exported): a per-group affine calibration of the routed predictions fitted on half A.
"""
import argparse
import copy
import glob
import json
import os
import shutil

import numpy as np
import torch

from utils import weighted_pearson

from .data import SeqStore
from .fast_eval import load_ckpt, net_inputs, pick_device, predict_batch
from .features import detect
from .groups import label as group_label
from .model import BASELINE_MACS, EnsembleNet, GroupGRUNet, build_net, export_onnx, macs

HERE = os.path.dirname(os.path.abspath(__file__))


def wp(y, p):
    return float(np.mean([weighted_pearson(y[:, j], p[:, j]) for j in range(2)]))


def collect(nets, store, device):
    """-> Y [list per seq of scored rows], P [model][seq], true groups, detector probs [n_seqs, 4]."""
    Y, P, G, D = [], [[] for _ in nets], [], []
    for X, Yb, M in store.batches(list(range(store.n_seqs)), 128):
        cache = {}
        G.append(group_label(Yb))
        D.append(detect(X))
        Y += [Yb[b][M[b]] for b in range(len(X))]
        for k, net in enumerate(nets):
            Pb = predict_batch(net, net_inputs(net, X, cache), device)
            P[k] += [Pb[b][M[b]] for b in range(len(X))]
    return Y, P, np.concatenate(G), np.concatenate(D)


def interpolate(shared, net, alpha=0.5):
    """Weights (1 - alpha) * shared + alpha * net, for a fine-tune of `shared` (a ggru fine-tune: from the ggru copy
    of the shared GRU). None when the shapes do not match."""
    base = GroupGRUNet(**{k: v for k, v in net.config.items() if k != "arch"}).from_gru(shared.state_dict()) \
        if isinstance(net, GroupGRUNet) and not isinstance(shared, GroupGRUNet) else shared
    sa, sb = base.state_dict(), net.state_dict()
    if sa.keys() != sb.keys() or any(sa[k].shape != sb[k].shape for k in sa):
        return None
    out = build_net(net.config)
    out.load_state_dict({k: (1 - alpha) * sa[k].float() + alpha * sb[k].float() for k in sa})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", required=True, help="best.pt of the shared model (also the fallback)")
    ap.add_argument("--cands", nargs="+", required=True, help="best.pt paths/globs of candidate models")
    ap.add_argument("--valid-split", required=True)
    ap.add_argument("--threshold", type=float, default=0.6, help="detector confidence below this -> shared model")
    ap.add_argument("--max-cost", type=float, default=2.2, help="max cost (x baseline multiply-adds) per sequence")
    ap.add_argument("--no-blends", action="store_true")
    ap.add_argument("--export", help="folder for the routed submission")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    paths = [args.shared] + [p for c in args.cands for p in sorted(glob.glob(c))
                             if os.path.realpath(p) != os.path.realpath(args.shared)]
    names = [os.path.basename(os.path.dirname(p)) for p in paths]
    exps = [{}] + [torch.load(p, map_location="cpu", weights_only=False).get("exp") or {} for p in paths[1:]]
    groups_of = [None] + [e.get("group") for e in exps[1:]]  # None = all groups
    nets = [load_ckpt(p).eval() for p in paths]
    for k in range(1, len(paths)):  # weight interpolations of fine-tunes of the shared model
        init = exps[k].get("init")
        if init and os.path.realpath(init) == os.path.realpath(args.shared):
            mid = interpolate(nets[0], nets[k])
            if mid is not None:
                nets.append(mid.eval()); names.append(names[k] + "~wise"); groups_of.append(groups_of[k])
                paths.append(f"{paths[k]} (weights 0.5 / 0.5 with the shared model)")
    device = pick_device(args.device)
    nets = [n.to(device) for n in nets]
    cost = [macs(n.config) / BASELINE_MACS for n in nets]
    print(f"{len(nets)} models: " + ", ".join(f"{n}" + (f" (group {g + 1})" if g is not None else "")
                                              for n, g in zip(names, groups_of)), flush=True)

    Y, P, G, D = collect(nets, SeqStore("datasets/valid.parquet", args.threads), device)
    n_single = len(nets)
    # options: the single models, then blends shared + candidate (prediction average)
    opts = [{"name": n, "members": [k], "group": g, "cost": c} for k, (n, g, c) in enumerate(zip(names, groups_of, cost))]
    if not args.no_blends:
        for k in range(1, n_single):
            if cost[0] + cost[k] <= args.max_cost:
                P.append([(a + b) / 2 for a, b in zip(P[0], P[k])])
                opts.append({"name": f"shared+{names[k]}", "members": [0, k], "group": groups_of[k], "cost": cost[0] + cost[k]})
    split = json.load(open(args.valid_split))
    half = np.zeros(len(G), int)
    half[split["B"]] = 1  # 0 = A, 1 = B
    det, conf = D.argmax(1), D.max(1)
    print(f"detector: {(det == G).mean():.4f} of {len(G)} valid sequences right; "
          f"{int((conf < args.threshold).sum())} below the threshold {args.threshold} (-> shared model)", flush=True)

    def pooled(model_of_seq, seqs):
        """Pooled scored-row WP over `seqs`, each predicted by option model_of_seq[s]."""
        seqs = [s for s in seqs if len(Y[s])]
        if not seqs:  # no sequence of that kind in this half
            return float("nan")
        return wp(np.concatenate([Y[s] for s in seqs]), np.concatenate([P[model_of_seq[s]][s] for s in seqs]))

    idx = {h: {g: np.flatnonzero((half == h) & (G == g)) for g in range(4)} for h in (0, 1)}
    per = {}  # (option, half, group) -> WP
    print(f"\nscored-row WP per group (A = selection half, B = honest half)")
    print(f"{'option':34s} {'cost':>5s} " + " ".join(f"A_g{g + 1:<5d} B_g{g + 1:<5d}" for g in range(4)) + "  B_all")
    for k, o in enumerate(opts):
        cells = []
        for g in range(4):
            ok = o["group"] is None or o["group"] == g
            for h in (0, 1):
                per[k, h, g] = pooled(np.full(len(G), k), idx[h][g]) if ok else np.nan
                cells.append(f"{per[k, h, g]:.4f}" if ok else "   -  ")
        b_all = pooled(np.full(len(G), k), np.flatnonzero(half == 1)) if o["group"] is None else np.nan
        print(f"{o['name'][:34]:34s} {o['cost']:5.2f} " + " ".join(f"{a:7s} {b:7s}" for a, b in zip(cells[::2], cells[1::2]))
              + (f"  {b_all:.4f}" if o["group"] is None else ""), flush=True)

    def best_route(max_cost):
        route = []
        for g in range(4):
            ok = [k for k, o in enumerate(opts) if (o["group"] is None or o["group"] == g) and o["cost"] <= max_cost]
            route.append(max(ok, key=lambda k: np.nan_to_num(per[k, 0, g], nan=-1.0)))
        return route

    def routed(route, use_detector=True):
        if not use_detector:
            return np.array([route[g] for g in G])
        return np.array([route[d] if c >= args.threshold else 0 for d, c in zip(det, conf)])

    share = np.bincount(det, minlength=4) / len(det)  # valid mix; the train mix has more of groups 3-4
    share_train = np.array([0.123, 0.119, 0.329, 0.430])
    avg_cost = lambda route: (float(share @ [opts[k]["cost"] for k in route]),
                              float(share_train @ [opts[k]["cost"] for k in route]))
    choice = best_route(args.max_cost)
    single = best_route(max(cost) + 1e-9) if args.no_blends else \
        [max([k for k in range(n_single) if groups_of[k] in (None, g)], key=lambda k: np.nan_to_num(per[k, 0, g], nan=-1.0))
         for g in range(4)]
    for title, r in (("route", choice), ("route without blends", single)):
        print(f"\n{title} (best on half A per group): " + ", ".join(f"g{g + 1} -> {opts[k]['name']}" for g, k in enumerate(r))
              + "; average cost %.2f (valid mix) / %.2f (train mix)" % avg_cost(r))

    B = np.flatnonzero(half == 1)
    rows = [("shared alone", np.zeros(len(G), int))]
    rows += [(f"{o['name']} alone", np.full(len(G), k)) for k, o in enumerate(opts[1:n_single], 1) if o["group"] is None]
    rows += [("routed, true groups", routed(choice, False)), ("routed, detected groups", routed(choice)),
             ("routed without blends, detected", routed(single))]
    for g in range(4):  # one group routed at a time, the shared model elsewhere
        only = [0] * 4
        only[g] = choice[g]
        if choice[g]:
            rows.append((f"route only g{g + 1} ({opts[choice[g]]['name'][:16]})", routed(only)))
    print(f"\n{'combination':44s} {'valid_B':>8s} {'valid_A':>8s} " + " ".join(f"{'B_g' + str(g + 1):>7s}" for g in range(4)))
    results = {}
    for name, m in rows:
        rb, ra = pooled(m, B), pooled(m, np.flatnonzero(half == 0))
        bg = [pooled(m, idx[1][g]) for g in range(4)]
        results[name] = {"valid_B": rb, "valid_A": ra, **{f"B_g{g + 1}": v for g, v in enumerate(bg)}}
        print(f"{name[:44]:44s} {rb:8.5f} {ra:8.5f} " + " ".join(f"{v:7.4f}" for v in bg), flush=True)

    # per-group affine calibration of the routed predictions (fitted on half A, |y|-weighted least squares)
    m = routed(choice)
    cal = {}
    for g in range(4):
        sa = [s for s in idx[0][g] if len(Y[s])]
        p, y = np.concatenate([P[m[s]][s] for s in sa]), np.clip(np.concatenate([Y[s] for s in sa]), -2, 2)
        cal[g] = []
        for j in range(2):
            Z, w = np.stack([p[:, j], np.ones(len(p))], 1), np.abs(y[:, j])
            cal[g].append(np.linalg.solve((Z * w[:, None]).T @ Z, (Z * w[:, None]).T @ y[:, j]))
    sb = [s for s in B if len(Y[s])]
    yb = np.concatenate([Y[s] for s in sb])
    pb = np.concatenate([np.stack([cal[det[s]][j][0] * P[m[s]][s][:, j] + cal[det[s]][j][1] for j in range(2)], 1) for s in sb])
    results["routed + per-group calibration (not exported)"] = {"valid_B": wp(yb, pb)}
    print(f"{'routed + per-group calibration (not exported)':44s} {wp(yb, pb):8.5f}   slopes "
          + ", ".join(f"g{g + 1} {cal[g][0][0]:.2f}/{cal[g][1][0]:.2f}" for g in range(4)), flush=True)

    if args.export:
        used = sorted(set(choice) | {0})  # the shared model is always there as the fallback
        os.makedirs(args.export, exist_ok=True)
        files = []
        for j, k in enumerate(used):
            mem = [nets[i].cpu() for i in opts[k]["members"]]
            net = mem[0] if len(mem) == 1 else EnsembleNet([copy.deepcopy(x) for x in mem], np.ones((len(mem), 2)))
            files.append(f"model_{j}.onnx")
            export_onnx(net, os.path.join(args.export, files[-1]))
        routes = {"models": files, "route": [used.index(k) for k in choice], "fallback": used.index(0),
                  "threshold": args.threshold}
        json.dump(routes, open(os.path.join(args.export, "routes.json"), "w"), indent=1)
        shutil.copy(os.path.join(HERE, "solution_routed.py"), os.path.join(args.export, "solution.py"))
        shutil.copy(os.path.join(HERE, "solution_template.py"), os.path.join(args.export, "wnn_runner.py"))
        shutil.copy(os.path.join(HERE, "features.py"), os.path.join(args.export, "wnn_features.py"))
        json.dump({"models": {f: [paths[i] for i in opts[k]["members"]] for f, k in zip(files, used)},
                   "route_names": [opts[k]["name"] for k in choice], "cost": {opts[k]["name"]: round(opts[k]["cost"], 3) for k in used},
                   "average_cost_valid_train_mix": avg_cost(choice), "scores": results},
                  open(os.path.join(args.export, "routing.json"), "w"), indent=1)
        print(f"exported {len(files)} models -> {args.export} (route {routes['route']}, fallback {routes['fallback']})")


if __name__ == "__main__":
    main()
