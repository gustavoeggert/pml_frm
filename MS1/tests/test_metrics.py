import unittest

import numpy as np

from MS1.metrics import MetricAccumulator, count_violations

SOLUTION = np.array([int(c) for c in (
    "247659318689137425531482967874921653915346782362875149798514236423768591156293874"
)])


class TestMetrics(unittest.TestCase):

    def test_valid_solution_has_no_violations(self):
        self.assertEqual(count_violations(SOLUTION[None])[0], 0)

    def test_duplicate_is_counted(self):
        broken = SOLUTION.copy()
        broken[0] = broken[1]
        self.assertGreater(count_violations(broken[None])[0], 0)

    def test_accumulator(self):
        x = SOLUTION.copy()
        x[:10] = 0
        wrong = SOLUTION.copy()
        wrong[0] = wrong[1]

        acc = MetricAccumulator()
        acc.update(np.stack([SOLUTION, wrong]), np.stack([x, x]), np.stack([SOLUTION, SOLUTION]))
        s = acc.summary()

        self.assertEqual(s["n"], 2)
        self.assertAlmostEqual(s["exact"], 0.5)
        self.assertAlmostEqual(s["valid"], 0.5)
        self.assertAlmostEqual(s["cell_acc"], 19 / 20)

    def test_clue_buckets_cover_all_puzzles(self):
        xs = []
        for n_blank in (5, 20, 40, 60):
            x = SOLUTION.copy()
            x[:n_blank] = 0
            xs.append(x)
        acc = MetricAccumulator()
        acc.update(np.stack([SOLUTION] * 4), np.stack(xs), np.stack([SOLUTION] * 4))

        self.assertEqual(sum(b["n"] for b in acc.by_clue_bucket(2).values()), 4)


if __name__ == "__main__":
    unittest.main()
