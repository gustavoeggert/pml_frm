"""Report figures and the results table from MS1/results (copied from Hydra).

Run from the repo root:  python -m MS1.make_figures
Missing runs are skipped, so the script works with any subset of the results.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from MS1.data_loader import load_data
from MS1.metrics import MetricAccumulator
from MS1.model import Model, decode, to_flat
from MS1.visualization import show

# label -> run directory; the order is the table order
RUNS = {
    "d128, 100k x 3 ep": "run_noaug",
    "d128, 100k x 3 ep, aug.": "run_aug",
    "d192, 100k x 6 ep": "run_long",
    "d192, 300k x 3 ep": "run_more_data",
    "d192, 300k x 3 ep, aug.": "run_more_data_aug",
    "d192, 300k x 3 ep (seed 1)": "run_more_data_seed1",
    "d192, 300k x 6 ep": "run_more_data_long",
    "d192, 300k x 6 ep (seed 1)": "run_more_data_long_seed1",
}
MAIN_RUN = "run_more_data_long"
CURVE_RUNS = {"300k x 6 epochs": ("run_more_data_long", "#d62728"),
              "300k x 3 epochs": ("run_more_data", "#1f77b4")}
BASELINES = {"random": "Random digit", "majority": "Majority digit",
             "naive_candidates": "Naive candidates (no search)", "backtracking": "Backtracking solver"}
MODEL_COLOR, NAIVE_COLOR = "#d62728", "#ff7f0e"


def _load_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def _available(results_dir, runs):
    return {k: v for k, v in runs.items() if (results_dir / (v if isinstance(v, str) else v[0]) / "results.json").is_file()}


def results_table(results_dir):
    rows = []
    baselines = _load_json(results_dir / "baselines.json") or {}
    for key, name in BASELINES.items():
        if key in baselines:
            r = baselines[key]
            rows.append({"method": name, "n_test": r["n"], **{k: r["summary"][k] for k in ("cell_acc", "exact", "valid")},
                         "ms_per_puzzle": r["sec_per_puzzle"] * 1e3})
    for label, run in _available(results_dir, RUNS).items():
        res = _load_json(results_dir / run / "results.json")
        s, a = res["summary"], res["args"]
        row = {"method": f"Recurrent transformer ({label})", "n_test": s["n"],
               "cell_acc": s["cell_acc"], "exact": s["exact"], "valid": s["valid"], "ms_per_puzzle": np.nan,
               "n_train": a["n_train"], "epochs": a["epochs"], "d_model": a["d_model"], "n_steps": a["n_steps"],
               "augment": a["augment"], "seed": a["seed"]}
        fresh = _load_json(results_dir / run / "eval_fresh.json")
        if fresh:
            row.update({"ms_per_puzzle": fresh["test"].get("sec_per_puzzle", np.nan) * 1e3,
                        "seen_train_cell_acc": fresh["seen_train"]["cell_acc"],
                        "fresh_train_cell_acc": fresh["fresh_train"]["cell_acc"],
                        "fresh_train_exact": fresh["fresh_train"]["exact"]})
        second = _load_json(results_dir / run / "eval_test_offset_2000.json")
        if second:
            row.update({"slice2_cell_acc": second["test"]["cell_acc"], "slice2_exact": second["test"]["exact"]})
        rows.append(row)
    return pd.DataFrame(rows)


def learning_curves(results_dir, out):
    runs = _available(results_dir, CURVE_RUNS) or {"100k x 3 epochs": ("run_noaug", "#1f77b4")}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [("loss", "Cross-entropy loss"), ("cell_acc", "Cell accuracy (blank cells)"), ("exact", "Exact-match accuracy")]
    for label, (run, color) in runs.items():
        log = pd.read_csv(results_dir / run / "log.csv")
        for ax, (metric, title) in zip(axes, panels):
            ax.plot(log["step"], log[f"test_{metric}"], color=color, label=f"{label}: test")
            ax.plot(log["step"], log[f"probe_{metric}"], color=color, linestyle="--", label=f"{label}: seen train")
            if f"fresh_{metric}" in log:
                ax.plot(log["step"], log[f"fresh_{metric}"], color=color, linestyle=":", label=f"{label}: unseen train")
            ax.set(title=title, xlabel="training step")
            ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _bars(ax, groups, series, title):
    """series: {label: (breakdown dict, color)}; bars for each group present in the first series."""
    width = 0.8 / len(series)
    x = np.arange(len(groups))
    for i, (label, (breakdown, color)) in enumerate(series.items()):
        vals = [breakdown.get(g, {}).get("exact", np.nan) for g in groups]
        pos = x + (i - (len(series) - 1) / 2) * width
        ax.bar(pos, vals, width, color=color, label=label)
        for xi, v in zip(pos, vals):
            if not np.isnan(v):
                ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    first = next(iter(series.values()))[0]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n(n={first[g]['n']})" for g in groups], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)


def breakdown_figure(results_dir, out):
    """Exact-match by clue count, rating band and shared clue pattern; model vs naive candidates, same test rows."""
    model = (_load_json(results_dir / MAIN_RUN / "eval_fresh.json") or {}).get("test")
    naive = (_load_json(results_dir / "baselines.json") or {}).get("naive_candidates")
    if model is None or "by_rating" not in model or naive is None:
        print("breakdown figure skipped: run MS1.evaluate on the main run and MS1.baselines first")
        return
    keys = [("by_clue_bucket", "by number of given clues"), ("by_rating", "by difficulty rating"),
            ("by_shared_pattern", "by clue pattern shared with train")]
    keys = [(k, t) for k, t in keys if k in model]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.8), sharey=True)
    for ax, (key, title) in zip(np.atleast_1d(axes), keys):
        series = {"recurrent transformer": (model[key], MODEL_COLOR),
                  "naive candidates": (naive.get(key, {}), NAIVE_COLOR)}
        _bars(ax, list(model[key]), series, f"Exact match {title}")
    np.atleast_1d(axes)[0].set_ylabel("exact-match accuracy")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _load_model(results_dir, run):
    ckpt = torch.load(results_dir / run / "model.pt", map_location="cpu")
    a = ckpt["args"]
    model = Model(a["d_model"], a["n_heads"], a["n_layers"], a["n_steps"], peer_attention=not a["global_attention"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


@torch.no_grad()
def _predict(model, X, batch_size=250):
    return np.concatenate([decode(model(X[i:i + batch_size]), X[i:i + batch_size]).numpy()
                           for i in range(0, len(X), batch_size)])


def examples(results_dir, out, run, n_test, data_dir):
    """One solved puzzle and two failures (few / many wrong cells) of the given run."""
    if not (results_dir / run / "model.pt").is_file():
        print(f"examples skipped: {run}/model.pt not found")
        return
    _, test = load_data(transformation=to_flat, n_train=0, n_test=n_test, data_dir=data_dir)
    xs, ys = map(np.stack, zip(*test))
    pred = _predict(_load_model(results_dir, run), torch.from_numpy(xs))
    acc = MetricAccumulator()
    acc.update(pred, xs, ys)
    print(f"local re-evaluation of {run} on {n_test} test puzzles: {acc.summary()}")

    exact = (pred == ys).all(axis=1)
    wrong = ((pred != ys) & (xs == 0)).sum(axis=1)
    failures = np.flatnonzero(~exact)
    picks = ([int(np.flatnonzero(exact)[0])] if exact.any() else []) + \
            [int(failures[np.argmin(wrong[failures])]), int(failures[np.argmax(wrong[failures])])]
    titles = [f"test puzzle {i}: {81 - int((xs[i] == 0).sum())} clues\n"
              + ("solved" if wrong[i] == 0 else f"{wrong[i]} wrong cell{'s' if wrong[i] != 1 else ''}") for i in picks]
    fig = show([pred[i] for i in picks], clues=[xs[i] for i in picks], target=[ys[i] for i in picks],
               titles=titles, show_plot=False)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Build report figures and the results table.")
    p.add_argument("--results-dir", default="MS1/results")
    p.add_argument("--figures-dir", default="MS1/figures")
    p.add_argument("--example-run", default=MAIN_RUN)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--data-dir", default=None)
    args = p.parse_args()

    results_dir, figures_dir = Path(args.results_dir), Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    table = results_table(results_dir)
    table.to_csv(figures_dir / "results_table.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    learning_curves(results_dir, figures_dir / "learning_curves.png")
    breakdown_figure(results_dir, figures_dir / "breakdowns.png")
    run = args.example_run if (results_dir / args.example_run / "model.pt").is_file() else "run_noaug"
    examples(results_dir, figures_dir / "failure_examples.png", run, args.n_test, args.data_dir)


if __name__ == "__main__":
    main()
