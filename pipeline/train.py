"""Train one stateful GRU (thin wrapper over trainer.train_many; see multi_train.py for several).

    python -m pipeline.train --name quick --train-seqs 1000 --epochs 2
    python -m pipeline.train --name gru128_wmse --epochs 4

Outputs in runs/NAME/: best.pt, log.jsonl, results.json, submission/{solution.py, model.onnx}.
"""
import argparse
import dataclasses

from .multi_train import add_common_flags, add_flags
from .trainer import Common, ExpConfig, train_many


def main():
    ap = argparse.ArgumentParser()
    shared = ("seed", "chunk")  # one flag each for a single model: used for the model and the data stream
    exp_fields = {f.name for f in dataclasses.fields(ExpConfig)} - set(shared)
    ap.add_argument("--name", required=True)
    add_flags(ap, ExpConfig, skip=("name", *shared))
    add_common_flags(ap)
    args = vars(ap.parse_args())
    exp = ExpConfig(**{k: args.pop(k) for k in list(args) if k in exp_fields}, seed=args["seed"])
    train_many([exp], Common(**args))


if __name__ == "__main__":
    main()
