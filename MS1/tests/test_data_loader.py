
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from MS1.data_loader import load_data


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

        answer = (
            "247659318689137425531482967"
            "874921653915346782362875149"
            "798514236423768591156293874"
        )

        datasets = {
            "train.csv": [
                ("2" + "." * 80, answer),
                ("." * 80 + "4", answer),
            ],
            "test.csv": [
                ("." * 81, answer),
            ],
        }

        for filename, rows in datasets.items():
            with (self.root / filename).open(
                "w", newline="", encoding="utf-8"
            ) as f:
                writer = csv.writer(f)
                writer.writerow(["question", "answer"])
                writer.writerows(rows)

    def load(self, **kwargs):
        return load_data(data_dir=self.root, **kwargs)

    def test_sample_counts(self):
        train, test = self.load()

        self.assertEqual(sum(1 for _ in train), 2)
        self.assertEqual(sum(1 for _ in test), 1)

    def test_shapes_and_clues(self):
        train, test = self.load()

        for x, y in list(train) + list(test):
            self.assertEqual(x.shape, (9, 9))
            self.assertEqual(y.shape, (9, 9))
            self.assertTrue(np.all((x == 0) | (x == y)))

    def test_non_overlapping_fixture(self):
        train, test = self.load()

        train_keys = {x.tobytes() for x, _ in train}
        test_keys = {x.tobytes() for x, _ in test}

        self.assertTrue(train_keys.isdisjoint(test_keys))

    def test_sample_limits(self):
        train, test = self.load(n_train=1, n_test=0)

        self.assertEqual(sum(1 for _ in train), 1)
        self.assertEqual(sum(1 for _ in test), 0)

    def test_transformation(self):
        def transform(x, y):
            return x.astype(np.float32), y

        train, _ = self.load(transformation=transform)
        x, _ = next(train)

        self.assertEqual(x.dtype, np.float32)

    def test_lazy_loading(self):
        train, _ = self.load()

        # Generator should read data only when iterated.
        self.assertEqual(next(train)[0].shape, (9, 9))

    def test_invalid_limit(self):
        with self.assertRaises(ValueError):
            self.load(n_train=-1)


if __name__ == "__main__":
    unittest.main()