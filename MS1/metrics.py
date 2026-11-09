import numpy as np

_IDX = np.arange(81).reshape(9, 9)
_BOXES = [
    _IDX[r:r + 3, c:c + 3].ravel()
    for r in range(0, 9, 3)
    for c in range(0, 9, 3)
]
# (27, 9) cell indices of every row, column and box
GROUPS = np.stack(list(_IDX) + list(_IDX.T) + _BOXES)


def count_violations(pred):
    """Number of duplicate digits summed over all rows, columns and boxes, per puzzle."""
    pred = np.asarray(pred)
    digits = np.arange(1, 10)
    onehot = pred[:, GROUPS][..., None] == digits  # (B, 27, 9 cells, 9 digits)
    counts = onehot.sum(axis=2)
    return np.maximum(counts - 1, 0).sum(axis=(1, 2))


class MetricAccumulator:
    """Collects per-puzzle results for (B, 81) integer arrays of puzzles, solutions and predictions."""

    def __init__(self):
        self._parts = []

    def update(self, pred, x, y):
        pred, x, y = np.asarray(pred), np.asarray(x), np.asarray(y)
        blank = x == 0
        correct = pred == y
        self._parts.append({
            "correct_blank": (correct & blank).sum(axis=1),
            "n_blank": blank.sum(axis=1),
            "exact": correct.all(axis=1),
            "valid": count_violations(pred) == 0,
            "clues": 81 - blank.sum(axis=1),
        })

    def _cat(self):
        return {k: np.concatenate([p[k] for p in self._parts]) for k in self._parts[0]}

    @staticmethod
    def _summarize(d, mask):
        n_blank = d["n_blank"][mask].sum()
        return {
            "n": int(mask.sum()),
            "cell_acc": float(d["correct_blank"][mask].sum() / max(n_blank, 1)),
            "exact": float(d["exact"][mask].mean()),
            "valid": float(d["valid"][mask].mean()),
        }

    def summary(self):
        d = self._cat()
        return self._summarize(d, np.ones(len(d["exact"]), dtype=bool))

    def by_clue_bucket(self, n_buckets=3):
        d = self._cat()
        quantiles = np.quantile(d["clues"], np.linspace(0, 1, n_buckets + 1)[1:-1])
        ids = np.searchsorted(np.unique(quantiles), d["clues"], side="left")
        out = {}
        for b in np.unique(ids):
            mask = ids == b
            lo, hi = d["clues"][mask].min(), d["clues"][mask].max()
            out[f"{lo}-{hi} clues"] = self._summarize(d, mask)
        return out