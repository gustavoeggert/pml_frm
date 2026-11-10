import argparse
import json
from pathlib import Path

import torch

from MS1.model import Model
from MS1.train import _load_fixed, evaluate


def main():
    p = argparse.ArgumentParser(description="Score a saved model on test, seen-train and unseen-train puzzles.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--test-offset", type=int, default=0, help="number of test rows to skip")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.n <= 0 or args.test_offset < 0:
        p.error("--n must be positive and --test-offset must be non-negative")

    ckpt = torch.load(Path(args.run_dir) / "model.pt", map_location=args.device)
    a = ckpt["args"]
    model = Model(a["d_model"], a["n_heads"], a["n_layers"], a["n_steps"],
                  peer_attention=not a["global_attention"]).to(args.device)
    model.load_state_dict(ckpt["state_dict"])

    sets = {
        "test": _load_fixed(args.n, "test", args.data_dir, skip=args.test_offset),
        "seen_train": _load_fixed(min(args.n, a["n_train"]), "train", args.data_dir),
        "fresh_train": _load_fixed(args.n, "train", args.data_dir, skip=a["n_train"]),
    }
    out = {}
    for name, (X, Y) in sets.items():
        loss, acc = evaluate(model, X, Y, torch.device(args.device))
        out[name] = {"loss": loss, **acc.summary()}
        if name == "test":
            out[name]["offset"] = args.test_offset
        print(name, json.dumps(out[name]), flush=True)
    filename = f"eval_test_offset_{args.test_offset}.json" if args.test_offset else "eval_fresh.json"
    (Path(args.run_dir) / filename).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
