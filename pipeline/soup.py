"""Weight soup: one checkpoint whose weights are the average of several fine-tunes of the same start.

    python -m pipeline.soup --out runs/sweep12_ft/vg1_soup_A/best.pt runs/sweep12_ft/bag_vg1_k{0,1,2,3}_A/best.pt

Same shape and same start (init) required. The soup keeps the first member's training config ("exp"), so crossfit
treats it like any recipe (group, valid_part, init for hedging), plus "soup": the member paths. Costs nothing at
inference: it is one model of the usual size.
"""
import argparse
import os

import torch


def soup(paths):
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    first = cks[0]
    for p, ck in zip(paths, cks):
        assert ck["config"] == first["config"], f"{p}: other shape"
        assert ck.get("exp", {}).get("init") == first.get("exp", {}).get("init"), f"{p}: other start"
    sd = {}
    for k, v in first["state_dict"].items():
        if v.is_floating_point():
            sd[k] = (sum(ck["state_dict"][k].double() for ck in cks) / len(cks)).to(v.dtype)
        else:
            sd[k] = v.clone()
    return {**first, "state_dict": sd, "soup": list(paths)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("members", nargs="+")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(soup(args.members), args.out)
    print(f"soup of {len(args.members)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
