import numpy as np

_IDX = np.arange(81).reshape(9, 9)
_BOXES = [
    _IDX[r:r + 3, c:c + 3].ravel()
    for r in range(0, 9, 3)
    for c in range(0, 9, 3)
]
# (27, 9) cell indices of every row, column and box
GROUPS = np.stack(list(_IDX) + list(_IDX.T) + _BOXES)
# Clue-count bucket upper edges: 17-24, 25-26, 27+ (roughly terciles of the test set)
CLUE_EDGES = (24, 26)
# Difficulty-rating band upper edges: 0-1 (no search needed), 2-20, 21-50, 51+
RATING_EDGES = (1, 20, 50)


def rating_band(rating, edges=RATING_EDGES):
    rating = np.asarray(rating)
    lows = [0, *[e + 1 for e in edges]]
    names = [f"{lo}-{hi}" for lo, hi in zip(lows, edges)] + [f"{lows[-1]}+"]
    return np.array(names)[np.searchsorted(np.asarray(edges), rating, side="left")]


def breakdowns(acc, rating=None, source=None, shared=None):
    """All per-subset results for one evaluated set; metadata arrays align with the puzzles."""
    out = {"by_clue_bucket": acc.by_clue_bucket()}
    if rating is not None:
        out["by_rating"] = acc.by_group(rating_band(rating))
    if source is not None:
        out["by_source"] = acc.by_group(source)
    if shared is not None:
        out["by_shared_pattern"] = acc.by_group(np.where(shared, "shared", "not shared"))
    return out


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

    def by_clue_bucket(self, edges=CLUE_EDGES):
        """Fixed buckets (upper edges inclusive) so that all methods are broken down identically."""
        d = self._cat()
        ids = np.searchsorted(np.asarray(edges), d["clues"], side="left")
        bounds = [17, *[e + 1 for e in edges]]
        uppers = [*edges, 81]
        out = {}
        for b in np.unique(ids):
            label = f"{bounds[b]}-{uppers[b]} clues" if b < len(edges) else f"{bounds[b]}+ clues"
            out[label] = self._summarize(d, ids == b)
        return out

    def by_group(self, labels):
        """Breakdown by an arbitrary per-puzzle label (e.g. source, rating band, leakage flag)."""
        d = self._cat()
        labels = np.asarray(labels)
        return {str(g): self._summarize(d, labels == g) for g in np.unique(labels)}