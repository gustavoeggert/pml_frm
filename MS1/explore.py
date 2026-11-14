"""Dataset exploration for the report: statistics, leakage screen and Figures 1-4.

Run from the repo root:  python -m MS1.explore [--nrows 50000 for a quick check]
"""
import argparse
import json
from itertools import permutations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from MS1.metrics import GROUPS
from MS1.visualization import show

TRAIN_COLOR, TEST_COLOR = "#4c72b0", "#c44e52"


def dataset_root(data_dir=None, dataset="sudoku_extreme"):
    if data_dir is not None:
        return Path(data_dir)
    hydra = Path("/home/space/datasets") / dataset
    return hydra if hydra.is_dir() else Path(__file__).resolve().parents[1] / "data" / "sudoku_extreme_full"


def read_split(root, split, nrows=None):
    df = pd.read_csv(root / f"{split}.csv", nrows=nrows, dtype={"source": "category", "question": str, "answer": str})
    df["source"] = df["source"].astype(str).str.replace("puzzles", "p", regex=False)
    return df


def test_metadata(n, offset=0, data_dir=None, shared_path="MS1/results/test_shared_pattern.npy"):
    """Rating, source and (if explore.py has been run) the shared-pattern flag of test rows offset..offset+n."""
    df = pd.read_csv(dataset_root(data_dir) / "test.csv", skiprows=range(1, offset + 1), nrows=n,
                     usecols=["source", "rating"])
    meta = {"rating": df["rating"].to_numpy(),
            "source": df["source"].str.replace("puzzles", "p", regex=False).to_numpy()}
    if shared_path and Path(shared_path).is_file():
        meta["shared"] = np.load(shared_path)[offset:offset + n]
    return meta


def to_digits(strings):
    """Series of 81-char strings -> (N, 81) uint8, blanks ('.' or '0') as 0."""
    raw = np.frombuffer("".join(strings).encode("ascii"), dtype=np.uint8).reshape(-1, 81)
    out = raw - np.uint8(ord("0"))
    out[out > 9] = 0
    return out


# ---------------------------------------------------------------- validity

def check_validity(q_str, a_str, chunk=250_000):
    lengths_ok = bool((q_str.str.len() == 81).all() and (a_str.str.len() == 81).all())
    chars_ok = bool(q_str.str.fullmatch(r"[.0-9]{81}").all() and a_str.str.fullmatch(r"[1-9]{81}").all())
    q, a = to_digits(q_str), to_digits(a_str)
    legal = np.empty(len(a), dtype=bool)
    for i in range(0, len(a), chunk):
        bits = np.left_shift(1, a[i:i + chunk].astype(np.int16))
        # 9 cells whose bitwise OR covers digits 1-9 are a permutation of 1-9
        legal[i:i + chunk] = (np.bitwise_or.reduce(bits[:, GROUPS], axis=2) == 0b1111111110).all(axis=1)
    consistent = ((q == 0) | (q == a)).all(axis=1)
    return {"lengths_ok": lengths_ok, "chars_ok": chars_ok,
            "n_illegal_solutions": int((~legal).sum()), "n_inconsistent_clues": int((~consistent).sum())}, q, a


# ---------------------------------------------------------------- leakage screen

_PERMS = [list(p) for p in permutations(range(3))]
_SHIFTS = (np.arange(16, dtype=np.uint64)[::-1] * np.uint64(4))


def _pack(v):
    """(N, 81) values < 16 -> (N, 6) uint64, preserving lexicographic order."""
    v = np.concatenate([v, np.zeros((len(v), 15), dtype=v.dtype)], axis=1).reshape(len(v), 6, 16)
    return (v.astype(np.uint64) << _SHIFTS).sum(axis=2, dtype=np.uint64)


def _lex_less(a, b):
    less = np.zeros(len(a), dtype=bool)
    eq = np.ones(len(a), dtype=bool)
    for w in range(a.shape[1]):
        less |= eq & (a[:, w] < b[:, w])
        eq &= a[:, w] == b[:, w]
    return less


def pattern_signature(mask, chunk=100_000):
    """Invariant of a clue pattern under the Sudoku position symmetries.

    Row swaps within a band, band swaps, the same for columns, and transposition leave the
    signature unchanged, so equivalent patterns always share it (a necessary condition).
    Per box it keeps the clue count and the sorted row and column counts; per band/stack the
    sorted line counts; the lexicographic minimum over the 72 band/stack/transpose orders is taken.
    """
    out = np.empty((len(mask), 6), dtype=np.uint64)
    for start in range(0, len(mask), chunk):
        m = mask[start:start + chunk].reshape(-1, 3, 3, 3, 3).astype(np.uint8)  # band, row, stack, col
        best = None
        for mm in (m, m.transpose(0, 3, 4, 1, 2)):
            box = mm.sum(axis=(2, 4))                                        # (N, band, stack)
            rows = np.sort(mm.sum(axis=4).transpose(0, 1, 3, 2), axis=-1)    # (N, band, stack, 3)
            cols = np.sort(mm.sum(axis=2), axis=-1)                          # (N, band, stack, 3)
            feat = np.concatenate([box[..., None], rows, cols], axis=-1)     # (N, 3, 3, 7)
            band_rows = np.sort(mm.sum(axis=(3, 4)), axis=-1)                # (N, band, 3)
            stack_cols = np.sort(mm.sum(axis=(1, 2)), axis=-1)               # (N, stack, 3)
            for p in _PERMS:
                for s in _PERMS:
                    vec = np.concatenate([
                        feat[:, p][:, :, s].reshape(len(m), -1),
                        band_rows[:, p].reshape(len(m), -1),
                        stack_cols[:, s].reshape(len(m), -1),
                    ], axis=1)
                    packed = _pack(vec)
                    if best is None:
                        best = packed
                    else:
                        less = _lex_less(packed, best)
                        best[less] = packed[less]
        out[start:start + len(m)] = best
    return out


def relabel_digits(q):
    """Rename digits in order of first appearance (row-major): invariant under digit relabelling."""
    pos = np.empty((len(q), 9), dtype=np.int64)
    for d in range(1, 10):
        has = q == d
        pos[:, d - 1] = np.where(has.any(axis=1), has.argmax(axis=1), 81)
    rank = np.argsort(np.argsort(pos, axis=1, kind="stable"), axis=1) + 1
    lut = np.concatenate([np.zeros((len(q), 1), dtype=np.int64), rank], axis=1)
    return np.take_along_axis(lut, q.astype(np.int64), axis=1).astype(np.uint8)


def _ids(train_rows, test_rows):
    both = np.ascontiguousarray(np.concatenate([train_rows, test_rows]))
    keys = both.view(np.dtype((np.void, both.dtype.itemsize * both.shape[1]))).ravel()
    _, inv = np.unique(keys, return_inverse=True)
    return inv[:len(train_rows)], inv[len(train_rows):]


def _shared(train_rows, test_rows):
    tr, te = _ids(train_rows, test_rows)
    return np.isin(te, tr), tr, te


def leakage(q_tr, q_te, src_te):
    sig_tr = pattern_signature(q_tr > 0)
    sig_te = pattern_signature(q_te > 0)
    shared, tr, te = _shared(sig_tr, sig_te)
    within_tr = np.bincount(tr)[tr] > 1
    within_te = np.bincount(te)[te] > 1
    raw_shared, _, _ = _shared(np.packbits(q_tr > 0, axis=1), np.packbits(q_te > 0, axis=1))
    relabel_shared, _, _ = _shared(relabel_digits(q_tr), relabel_digits(q_te))
    exact_shared, _, _ = _shared(q_tr, q_te)
    per_source = pd.Series(shared).groupby(src_te.values).mean().sort_values(ascending=False)
    stats = {
        "n_shared_signatures": int(len(np.intersect1d(np.unique(tr), np.unique(te)))),
        "n_test_with_shared_signature": int(shared.sum()),
        "frac_test_with_shared_signature": float(shared.mean()),
        "frac_shared_by_source": {k: float(v) for k, v in per_source.items()},
        "n_test_same_raw_pattern": int(raw_shared.sum()),
        "n_test_same_after_digit_relabel": int(relabel_shared.sum()),
        "n_test_identical_puzzle": int(exact_shared.sum()),
        "n_train_repeated_signature": int(within_tr.sum()),
        "frac_train_repeated_signature": float(within_tr.mean()),
        "n_test_repeated_signature": int(within_te.sum()),
        "frac_test_repeated_signature": float(within_te.mean()),
    }
    return stats, shared


# ---------------------------------------------------------------- figures

def fig_examples(train, q, a, out):
    picks = [("p2_17_clue", "rating", True), ("p0_kaggle", "clues", False),
             ("p4_forum_hardest_1905", "rating", False), ("p6_forum_hardest_1106", "rating", False)]
    idx = []
    for source, key, ascending in picks:
        sub = train[train["source"] == source]
        if len(sub):
            idx.append(sub.sort_values([key, "rating"], ascending=[ascending, True]).index[0])
    titles = [f"{train.at[i, 'source']}\n{train.at[i, 'clues']} clues, rating {train.at[i, 'rating']}" for i in idx]
    fig = show([a[i] for i in idx], clues=[q[i] for i in idx], titles=titles, show_plot=False,
               ncols=len(idx), suptitle="Clue count does not determine difficulty")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_distributions(train, test, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.arange(16.5, 38.5)
    for df, name, color in ((train, "train", TRAIN_COLOR), (test, "test", TEST_COLOR)):
        w = np.full(len(df), 1 / len(df))
        axes[0].hist(df["clues"], bins=bins, weights=w, histtype="step", lw=1.6, color=color, label=name)
        axes[1].hist(df["rating"], bins=np.arange(0, df["rating"].max() + 2), weights=w,
                     histtype="step", lw=1.2, color=color, label=name)
    axes[0].set(title="Clue count", xlabel="number of given clues", ylabel="fraction of puzzles")
    axes[1].set(title="Difficulty", xlabel="difficulty rating (solver backtracks)", ylabel="fraction of puzzles (log)",
                yscale="log")
    for ax in axes:
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_sources(train, test, out):
    share = pd.DataFrame({"train": train["source"].value_counts(normalize=True),
                          "test": test["source"].value_counts(normalize=True)}).fillna(0) * 100
    share = share.sort_values("train")
    y = np.arange(len(share))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(y - 0.2, share["train"], 0.4, color=TRAIN_COLOR, label="train")
    ax.barh(y + 0.2, share["test"], 0.4, color=TEST_COLOR, label="test")
    ax.set_yticks(y)
    ax.set_yticklabels(share.index)
    ax.set(xlabel="share of the split (%)", title="Where the puzzles come from")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_difficulty(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts = sorted(df["clues"].unique())
    groups = [df.loc[df["clues"] == c, "rating"].values for c in counts]
    axes[0].boxplot(groups, positions=counts, showfliers=False, widths=0.6,
                    medianprops={"color": TEST_COLOR})
    axes[0].set_xticks(counts[::2])
    axes[0].set_xticklabels(counts[::2])
    axes[0].set(title="Difficulty at each clue count", xlabel="number of given clues", ylabel="difficulty rating")
    med = df.groupby("source").agg(clues=("clues", "median"), rating=("rating", "median"))
    axes[1].scatter(med["clues"], med["rating"], color=TRAIN_COLOR)
    for name, r in med.iterrows():
        # put the label on the left if a nearby point or the plot edge is on the right
        crowded = ((med["clues"] - r["clues"]).between(0.01, 3) & ((med["rating"] - r["rating"]).abs() < 5)).any()
        left = crowded or r["clues"] == med["clues"].max()
        axes[1].annotate(name, (r["clues"], r["rating"]), textcoords="offset points", xytext=(-6 if left else 6, 2),
                         ha="right" if left else "left", fontsize=7)
    axes[1].margins(x=0.08, y=0.1)
    axes[1].set(title="Each source, as one point", xlabel="median clues in source",
                ylabel="median rating in source")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main

def _describe(s):
    q = s.quantile([0.25, 0.5, 0.75, 0.99])
    return {"min": float(s.min()), "q25": float(q[0.25]), "median": float(q[0.5]), "mean": float(s.mean()),
            "q75": float(q[0.75]), "q99": float(q[0.99]), "max": float(s.max())}


def main():
    p = argparse.ArgumentParser(description="Explore Sudoku-Extreme and build the dataset figures.")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--nrows", type=int, default=None, help="read only the first rows of each split (quick check)")
    p.add_argument("--figures-dir", default="MS1/figures")
    p.add_argument("--results-dir", default="MS1/results")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    figures, results = Path(args.figures_dir), Path(args.results_dir)
    figures.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    root = dataset_root(args.data_dir)
    train, test = read_split(root, "train", args.nrows), read_split(root, "test", args.nrows)
    stats = {"n_train": len(train), "n_test": len(test)}

    stats["validity_train"], q_tr, a_tr = check_validity(train["question"], train["answer"])
    stats["validity_test"], q_te, a_te = check_validity(test["question"], test["answer"])
    for df, q in ((train, q_tr), (test, q_te)):
        df["clues"] = (q > 0).sum(axis=1)

    for name, df, q, a in (("train", train, q_tr, a_tr), ("test", test, q_te, a_te)):
        stats[name] = {
            "clues": _describe(df["clues"]),
            "rating": _describe(df["rating"]),
            "frac_cells_given": float((q > 0).mean()),
            "solution_digit_share": (np.bincount(a.ravel(), minlength=10)[1:] / a.size).tolist(),
            "clue_digit_share": (np.bincount(q[q > 0], minlength=10)[1:] / (q > 0).sum()).tolist(),
            "source_share": df["source"].value_counts(normalize=True).to_dict(),
            "n_duplicate_puzzles": int(df["question"].duplicated().sum()),
            "n_duplicate_solutions": int(df["answer"].duplicated().sum()),
        }
    stats["n_test_puzzles_in_train"] = int(test["question"].isin(train["question"]).sum())
    stats["n_test_solutions_in_train"] = int(test["answer"].isin(train["answer"]).sum())

    both = pd.concat([train[["source", "clues", "rating"]], test[["source", "clues", "rating"]]], ignore_index=True)
    stats["spearman_clues_rating"] = float(both["clues"].rank().corr(both["rating"].rank()))
    per_source = both.groupby("source").agg(n=("rating", "size"), median_rating=("rating", "median"),
                                            median_clues=("clues", "median"))
    stats["per_source"] = per_source.astype(float).to_dict(orient="index")
    easy = per_source.index[per_source["median_rating"] <= 1]
    stats["easy_sources"] = list(easy)
    stats["frac_from_easy_sources"] = float(both["source"].isin(easy).mean())
    stats["median_rating_by_clues"] = {int(k): float(v) for k, v in both.groupby("clues")["rating"].median().items()}

    # Copy clues, guess uniformly elsewhere: expected 1/9 on blanks
    rng = np.random.default_rng(args.seed)
    sub_q, sub_a = q_te[:20_000], a_te[:20_000]
    guess = np.where(sub_q > 0, sub_q, rng.integers(1, 10, size=sub_q.shape))
    stats["trivial_predictor"] = {"n": len(sub_q), "cell_acc_all_cells": float((guess == sub_a).mean()),
                                  "exact": float((guess == sub_a).all(axis=1).mean())}

    stats["leakage"], shared = leakage(q_tr, q_te, test["source"])
    np.save(results / "test_shared_pattern.npy", shared)

    (results / "explore_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps({k: stats[k] for k in ("n_train", "n_test", "validity_train", "validity_test",
                                            "spearman_clues_rating", "frac_from_easy_sources", "leakage")},
                     indent=2))

    fig_examples(train, q_tr, a_tr, figures / "examples_by_source.png")
    fig_distributions(train, test, figures / "distributions.png")
    fig_sources(train, test, figures / "sources.png")
    fig_difficulty(both, figures / "clues_vs_difficulty.png")


if __name__ == "__main__":
    main()
