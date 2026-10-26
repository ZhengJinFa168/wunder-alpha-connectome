"""Train several experiments side by side from one data stream.

    python -m pipeline.multi_train --config experiments.json --epochs 4 --threads 8 --cache-eval

experiments.json is a list of ExpConfig dicts, e.g.
    [{"name": "wmse"}, {"name": "mix", "loss": "mix"}, {"name": "diff_h192", "diff": true, "hidden": 192}]
Shared settings (epochs, batch, chunk, data/out dirs, ...) are the Common fields, as flags.
"""
import argparse
import dataclasses
import json
import typing

from .trainer import Common, ExpConfig, train_many


def field_type(f):
    """int | None -> int, str | None -> str, plain types unchanged."""
    args = [a for a in typing.get_args(f.type) if a is not type(None)]
    return args[0] if args else f.type


def add_flags(ap, cls, skip=("extra",)):
    for f in dataclasses.fields(cls):
        if f.name in skip:
            continue
        flag = "--" + f.name.replace("_", "-")
        if field_type(f) is bool:
            ap.add_argument(flag, action="store_true")
        else:
            ap.add_argument(flag, type=field_type(f), default=f.default)


def add_common_flags(ap):
    add_flags(ap, Common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON list of experiment configs")
    add_common_flags(ap)
    args = vars(ap.parse_args())
    configs = [ExpConfig(**c) for c in json.load(open(args.pop("config")))]
    names = [c.name for c in configs]
    assert len(set(names)) == len(names), "experiment names must be unique"
    train_many(configs, Common(**args))


if __name__ == "__main__":
    main()
