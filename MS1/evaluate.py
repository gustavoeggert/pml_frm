import argparse
import json
import time
from pathlib import Path

import torch

from MS1.explore import test_metadata
from MS1.metrics import breakdowns
from MS1.model import Model
from MS1.train import _load_fixed, evaluate


def main():
    p = argparse.ArgumentParser(description="Score a saved model on test, seen-train and unseen-train puzzles.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--test-offset", type=int, default=0, help="number of test rows to skip")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--shared-flags", default="MS1/results/test_shared_pattern.npy",
                   help="per-test-row flags written by MS1.explore (optional)")
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
        start = time.perf_counter()
        loss, acc = evaluate(model, X, Y, torch.device(args.device))
        out[name] = {"loss": loss, **acc.summary(), "sec_per_puzzle": (time.perf_counter() - start) / len(X)}
        if name == "test":
            out[name]["offset"] = args.test_offset
            out[name].update(breakdowns(acc, **test_metadata(len(X), args.test_offset, args.data_dir,
                                                               args.shared_flags)))
        print(name, json.dumps({k: v for k, v in out[name].items() if not k.startswith("by_")}), flush=True)
    filename = f"eval_test_offset_{args.test_offset}.json" if args.test_offset else "eval_fresh.json"
    (Path(args.run_dir) / filename).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
