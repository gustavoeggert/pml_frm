import argparse
import json
import time
from pathlib import Path

import numpy as np

from MS1.metrics import MetricAccumulator

ALL = 0x3FE  # candidate bitmask for digits 1-9

_ROWS = [[r * 9 + c for c in range(9)] for r in range(9)]
_COLS = [[r * 9 + c for r in range(9)] for c in range(9)]
_BOXES = [
    [(br * 3 + r) * 9 + bc * 3 + c for r in range(3) for c in range(3)]
    for br in range(3) for bc in range(3)
]
UNITS = _ROWS + _COLS + _BOXES
PEERS = [
    tuple(sorted({j for u in UNITS if i in u for j in u} - {i}))
    for i in range(81)
]


def _assign(cands, i, d):
    """Fix cell i to digit d and propagate naked singles; False on contradiction."""
    bit = 1 << d
    if not cands[i] & bit:
        return False
    cands[i] = bit
    for p in PEERS[i]:
        if cands[p] & bit:
            cands[p] &= ~bit
            if cands[p] == 0:
                return False
            if cands[p] & (cands[p] - 1) == 0:
                if not _assign(cands, p, cands[p].bit_length() - 1):
                    return False
    return True


def _setup(grid):
    cands = [ALL] * 81
    for i, d in enumerate(grid):
        if d and not _assign(cands, i, int(d)):
            return None
    return cands


def _hidden_singles(cands):
    """Repeat hidden singles until stable; False on contradiction."""
    changed = True
    while changed:
        changed = False
        for unit in UNITS:
            for d in range(1, 10):
                bit = 1 << d
                places = [i for i in unit if cands[i] & bit]
                if not places:
                    return False
                if len(places) == 1 and cands[places[0]] != bit:
                    if not _assign(cands, places[0], d):
                        return False
                    changed = True
    return True


def _to_grid(cands):
    # Lowest remaining candidate per cell; identical to the digit for solved cells.
    return np.array([(c & -c).bit_length() - 1 for c in cands], dtype=np.int64)


def _is_solved(cands):
    return all(c & (c - 1) == 0 for c in cands)


class RandomBaseline:
    """Uniform random digit in every blank cell."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def predict(self, x):
        return np.where(x > 0, x, self.rng.integers(1, 10, size=x.shape))


class MajorityBaseline:
    """Most frequent solution digit among blank training cells, in every blank cell."""

    def fit(self, x, y):
        self.digit = int(np.argmax(np.bincount(y[x == 0], minlength=10)[1:])) + 1
        return self

    def predict(self, x):
        return np.where(x > 0, x, self.digit)


class NaiveCandidateBaseline:
    """Naked and hidden singles only, no search; unresolved cells take their lowest candidate."""

    def predict_with_status(self, x):
        preds, solved = [], []
        for row in x:
            cands = _setup(row)
            if cands is None or not _hidden_singles(cands):
                preds.append(np.asarray(row, dtype=np.int64))
                solved.append(False)
                continue
            preds.append(_to_grid(cands))
            solved.append(_is_solved(cands))
        return np.stack(preds), np.array(solved)

    def predict(self, x):
        return self.predict_with_status(x)[0]


class _BudgetExceeded(Exception):
    pass


class BacktrackingSolver:
    """Exact solver: naked-single propagation with minimum-remaining-values branching."""

    def __init__(self, max_nodes=100_000):
        self.max_nodes = max_nodes
        self.n_unsolved = 0

    def _search(self, cands, budget):
        open_cells = [i for i in range(81) if cands[i] & (cands[i] - 1)]
        if not open_cells:
            return cands
        i = min(open_cells, key=lambda k: bin(cands[k]).count("1"))
        for d in range(1, 10):
            if not cands[i] & (1 << d):
                continue
            budget[0] -= 1
            if budget[0] < 0:
                raise _BudgetExceeded
            trial = cands.copy()
            if _assign(trial, i, d):
                result = self._search(trial, budget)
                if result is not None:
                    return result
        return None

    def predict(self, x):
        preds = []
        for row in x:
            cands = _setup(row)
            if cands is None:
                self.n_unsolved += 1
                preds.append(np.asarray(row, dtype=np.int64))
                continue
            try:
                solved = self._search(cands, [self.max_nodes])
            except _BudgetExceeded:
                solved = None
            if solved is None:
                self.n_unsolved += 1
                solved = cands
            preds.append(_to_grid(solved))
        return np.stack(preds)


def _evaluate(predict, x, y):
    start = time.perf_counter()
    pred = predict(x)
    elapsed = time.perf_counter() - start
    acc = MetricAccumulator()
    acc.update(pred, x, y)
    return {
        "n": len(x),
        "summary": acc.summary(),
        "by_clue_bucket": acc.by_clue_bucket(),
        "sec_per_puzzle": elapsed / len(x),
    }


def main():
    from MS1.data_loader import load_data
    from MS1.model import to_flat

    parser = argparse.ArgumentParser(description="Evaluate non-learned Sudoku baselines.")
    parser.add_argument("--n-fit", type=int, default=10_000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--n-solver", type=int, default=300)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="MS1/results/baselines.json")
    args = parser.parse_args()

    train, test = load_data(
        transformation=to_flat, n_train=args.n_fit, n_test=args.n_test, data_dir=args.data_dir
    )
    fit_x, fit_y = map(np.stack, zip(*train))
    test_x, test_y = map(np.stack, zip(*test))

    results = {
        "random": _evaluate(RandomBaseline(args.seed).predict, test_x, test_y),
        "majority": _evaluate(MajorityBaseline().fit(fit_x, fit_y).predict, test_x, test_y),
    }

    naive = NaiveCandidateBaseline()
    solved_flags = []

    def naive_predict(x):
        pred, solved = naive.predict_with_status(x)
        solved_flags.append(solved)
        return pred

    results["naive_candidates"] = _evaluate(naive_predict, test_x, test_y)
    results["naive_candidates"]["fully_solved_by_propagation"] = float(np.concatenate(solved_flags).mean())

    solver = BacktrackingSolver()
    n = min(args.n_solver, len(test_x))
    results["backtracking"] = _evaluate(solver.predict, test_x[:n], test_y[:n])
    results["backtracking"]["n_unsolved"] = solver.n_unsolved

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    for name, r in results.items():
        s = r["summary"]
        print(
            f"{name:18s} n={r['n']:5d} cell_acc={s['cell_acc']:.3f} "
            f"exact={s['exact']:.3f} valid={s['valid']:.3f} {r['sec_per_puzzle'] * 1e3:.1f} ms/puzzle"
        )


if __name__ == "__main__":
    main()
