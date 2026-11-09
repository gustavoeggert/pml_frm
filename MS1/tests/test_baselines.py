import unittest

import numpy as np

from MS1.baselines import (
    BacktrackingSolver,
    MajorityBaseline,
    NaiveCandidateBaseline,
    RandomBaseline,
)
from MS1.metrics import count_violations

SOLUTION = np.array([int(c) for c in (
    "247659318689137425531482967874921653915346782362875149798514236423768591156293874"
)])


def blank(cells):
    x = SOLUTION.copy()
    x[cells] = 0
    return x


class TestBaselines(unittest.TestCase):

    def test_naive_solves_isolated_blanks(self):
        # One blank per row, column and box, so every blank is a naked single.
        x = blank([0, 12, 24])
        pred, solved = NaiveCandidateBaseline().predict_with_status(x[None])

        self.assertTrue(solved[0])
        np.testing.assert_array_equal(pred[0], SOLUTION)

    def test_backtracking_returns_valid_grid_with_clues(self):
        x = blank(list(range(0, 81, 2)))
        pred = BacktrackingSolver().predict(x[None])[0]

        self.assertEqual(count_violations(pred[None])[0], 0)
        self.assertTrue(np.all((x == 0) | (pred == x)))

    def test_backtracking_handles_empty_grid(self):
        solver = BacktrackingSolver()
        pred = solver.predict(np.zeros((1, 81), dtype=np.int64))

        self.assertEqual(count_violations(pred)[0], 0)
        self.assertEqual(solver.n_unsolved, 0)

    def test_random_and_majority_keep_clues(self):
        x = np.stack([blank(list(range(30)))])
        y = SOLUTION[None]

        for pred in (RandomBaseline().predict(x), MajorityBaseline().fit(x, y).predict(x)):
            self.assertTrue(np.all((x == 0) | (pred == x)))
            self.assertTrue(np.all((pred >= 1) & (pred <= 9)))


if __name__ == "__main__":
    unittest.main()
