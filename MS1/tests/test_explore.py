import unittest

import numpy as np

from MS1.explore import pattern_signature


def _axis_permutation(rng):
    return np.concatenate([b * 3 + rng.permutation(3) for b in rng.permutation(3)])


class TestPatternSignature(unittest.TestCase):

    def test_invariant_under_sudoku_symmetries(self):
        rng = np.random.default_rng(0)
        masks = rng.random((500, 9, 9)) < 0.3
        moved = []
        for m in masks:
            m = m[_axis_permutation(rng)][:, _axis_permutation(rng)]
            moved.append(m.T if rng.random() < 0.5 else m)
        a = pattern_signature(masks.reshape(-1, 81))
        b = pattern_signature(np.stack(moved).reshape(-1, 81))
        np.testing.assert_array_equal(a, b)

    def test_distinguishes_different_patterns(self):
        rng = np.random.default_rng(1)
        masks = rng.random((500, 81)) < 0.3
        self.assertGreater(len(np.unique(pattern_signature(masks), axis=0)), 490)


if __name__ == "__main__":
    unittest.main()
