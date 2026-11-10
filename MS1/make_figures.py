import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle

from MS1.data_loader import load_data
from MS1.metrics import MetricAccumulator
from MS1.model import Model, decode, to_flat
from MS1.visualization import show

RUNS = {"no augmentation": "run_noaug", "augmentation": "run_aug"}
COLORS = {"no augmentation": "#1f77b4", "augmentation": "#d62728"}


def _load_json(path):
    return json.loads(Path(path).read_text())


def learning_curves(results_dir, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for label, run in RUNS.items():
        log = pd.read_csv(results_dir / run / "log.csv")
        for ax, (seen, unseen, title) in zip(axes, [
            ("probe_loss", "test_loss", "Cross-entropy loss"),
            ("probe_cell_acc", "test_cell_acc", "Cell accuracy (blank cells)"),
            ("probe_exact", "test_exact", "Exact-match accuracy"),
        ]):
            ax.plot(log["step"], log[unseen], color=COLORS[label], label=f"{label}, test")
            ax.plot(log["step"], log[seen], color=COLORS[label], linestyle="--", label=f"{label}, seen train")
            ax.set_title(title)
            ax.set_xlabel("training step")
            ax.grid(alpha=0.3)
    args = _load_json(results_dir / "run_noaug" / "results.json")["args"]
    steps_per_epoch = -(-args["n_train"] // args["batch_size"])
    for ax in axes:
        for e in range(1, args["epochs"]):
            ax.axvline(e * steps_per_epoch, color="grey", linestyle=":", linewidth=0.8)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _results_table(results_dir):
    rows = []
    baselines = _load_json(results_dir / "baselines.json")
    names = {
        "random": "Random digit",
        "majority": "Majority digit",
        "naive_candidates": "Naive candidates (no search)",
        "backtracking": "Backtracking solver",
    }
    for key, name in names.items():
        r = baselines[key]
        rows.append({"method": name, "n_test": r["n"], **{k: r["summary"][k] for k in ("cell_acc", "exact")},
                     "ms_per_puzzle": r["sec_per_puzzle"] * 1e3})
    for label, run in RUNS.items():
        s = _load_json(results_dir / run / "results.json")["summary"]
        rows.append({"method": f"Recurrent transformer ({label})", "n_test": s["n"],
                     "cell_acc": s["cell_acc"], "exact": s["exact"], "ms_per_puzzle": np.nan})
    return pd.DataFrame(rows)


def baseline_comparison(table, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    y = np.arange(len(table))[::-1]
    for ax, col, title in zip(axes, ("cell_acc", "exact"), ("Cell accuracy (blank cells)", "Exact-match accuracy")):
        ax.barh(y, table[col], color=["#999999"] * 2 + ["#ff7f0e", "#2ca02c"] + [COLORS[l] for l in RUNS])
        for yi, v in zip(y, table[col]):
            ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=9)
        ax.set_xlim(0, 1.12)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(table["method"])
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def clue_buckets(results_dir, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    width = 0.38
    for i, (label, run) in enumerate(RUNS.items()):
        buckets = _load_json(results_dir / run / "results.json")["by_clue_bucket"]
        names = [f"{k}\n(n={v['n']})" for k, v in buckets.items()]
        x = np.arange(len(names))
        for ax, col, title in zip(axes, ("cell_acc", "exact"), ("Cell accuracy", "Exact-match accuracy")):
            vals = [v[col] for v in buckets.values()]
            ax.bar(x + (i - 0.5) * width, vals, width, color=COLORS[label], label=label)
            for xi, v in zip(x + (i - 0.5) * width, vals):
                ax.text(xi, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(names)
            ax.set_title(f"{title} by number of given clues")
            ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _load_model(results_dir, run):
    ckpt = torch.load(results_dir / run / "model.pt", map_location="cpu")
    a = ckpt["args"]
    model = Model(a["d_model"], a["n_heads"], a["n_layers"], a["n_steps"],
                  peer_attention=not a["global_attention"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


@torch.no_grad()
def _predict(model, X, batch_size=250):
    return np.concatenate([
        decode(model(X[i:i + batch_size]), X[i:i + batch_size]).numpy()
        for i in range(0, len(X), batch_size)
    ])


def examples(results_dir, figures_dir, run, n_test, data_dir):
    _, test = load_data(transformation=to_flat, n_train=0, n_test=n_test, data_dir=data_dir)
    xs, ys = map(np.stack, zip(*test))
    pred = _predict(_load_model(results_dir, run), torch.from_numpy(xs))

    acc = MetricAccumulator()
    acc.update(pred, xs, ys)
    reference = _load_json(results_dir / run / "results.json")["summary"]
    print(f"local re-evaluation of {run}: cell_acc={acc.summary()['cell_acc']:.4f} "
          f"exact={acc.summary()['exact']:.4f} (saved: {reference['cell_acc']:.4f} / {reference['exact']:.4f})")

    exact = (pred == ys).all(axis=1)
    wrong_cells = ((pred != ys) & (xs == 0)).sum(axis=1)
    picks = [("ok", int(np.flatnonzero(exact)[0]))] if exact.any() else []
    failures = np.flatnonzero(~exact)
    picks += [("fail_few_errors", int(failures[np.argmin(wrong_cells[failures])])),
              ("fail_many_errors", int(failures[np.argmax(wrong_cells[failures])]))]

    for tag, i in picks:
        boards = [xs[i].reshape(9, 9), pred[i].reshape(9, 9), ys[i].reshape(9, 9)]
        titles = ["Puzzle", f"Prediction ({wrong_cells[i]} wrong cell{'s' if wrong_cells[i] != 1 else ''})", "Solution"]
        fig = show(boards, titles=titles, show_plot=False)
        for r, c in zip(*np.nonzero((boards[1] != boards[2]) & (boards[0] == 0))):
            fig.axes[1].add_patch(Rectangle((c, r), 1, 1, fill=True, facecolor="#ff0000", alpha=0.25, linewidth=0))
        fig.savefig(figures_dir / f"example_{tag}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Build report figures and the results table.")
    p.add_argument("--results-dir", default="MS1/results")
    p.add_argument("--figures-dir", default="MS1/figures")
    p.add_argument("--example-run", default="run_noaug")
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--data-dir", default=None)
    args = p.parse_args()

    results_dir, figures_dir = Path(args.results_dir), Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    table = _results_table(results_dir)
    table.to_csv(figures_dir / "results_table.csv", index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    learning_curves(results_dir, figures_dir / "learning_curves.png")
    baseline_comparison(table, figures_dir / "model_vs_baselines.png")
    clue_buckets(results_dir, figures_dir / "clue_buckets.png")
    examples(results_dir, figures_dir, args.example_run, args.n_test, args.data_dir)


if __name__ == "__main__":
    main()
