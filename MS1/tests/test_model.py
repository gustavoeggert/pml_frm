import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from MS1.metrics import count_violations
from MS1.model import Model, augment, augmented, decode, make_transform, sudoku_loss, to_flat
from MS1.evaluate import main as evaluate_main
from MS1.train import iter_batches

SOLUTION = np.array([int(c) for c in (
    "247659318689137425531482967874921653915346782362875149798514236423768591156293874"
)]).reshape(9, 9)


def small_model():
    return Model(d_model=16, n_heads=2, n_layers=1, n_steps=3)


def batch(n=4, n_blank=30, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for _ in range(n):
        x = SOLUTION.copy().ravel()
        x[rng.choice(81, n_blank, replace=False)] = 0
        xs.append(x)
        ys.append(SOLUTION.ravel())
    return torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys))


class TestAugment(unittest.TestCase):

    def test_augmented_pair_stays_valid_and_consistent(self):
        rng = np.random.default_rng(0)
        x = SOLUTION.copy()
        x[rng.random((9, 9)) < 0.5] = 0

        for _ in range(20):
            ax, ay = augment(x, SOLUTION, rng)
            self.assertEqual(count_violations(ay.reshape(1, 81))[0], 0)
            self.assertTrue(np.all((ax == 0) | (ax == ay)))
            self.assertEqual((ax == 0).sum(), (x == 0).sum())

    def test_augmented_stream_flattens(self):
        fx, fy = next(augmented(iter([(SOLUTION, SOLUTION)])))

        self.assertEqual(fx.shape, (81,))
        self.assertEqual(fy.dtype, np.int64)
        self.assertEqual(to_flat(SOLUTION, SOLUTION)[0].shape, (81,))

    def test_make_transform_matches_augmented_stream(self):
        transform = make_transform(use_augmentation=True, seed=3)
        expected = next(augmented(iter([(SOLUTION, SOLUTION)]), seed=3))
        actual = transform(SOLUTION, SOLUTION)

        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_array_equal(make_transform()(SOLUTION, SOLUTION)[0], SOLUTION.ravel())


class TestEvaluate(unittest.TestCase):

    def test_test_offset_preserves_existing_results(self):
        checkpoint = {
            "args": {"d_model": 16, "n_heads": 2, "n_layers": 1, "n_steps": 3,
                     "global_attention": False, "n_train": 1},
            "state_dict": small_model().state_dict(),
        }
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "eval_fresh.json").write_text("existing")
            torch.save(checkpoint, run_dir / "model.pt")
            with patch.object(sys, "argv", ["evaluate", "--run-dir", directory, "--n", "1",
                                           "--test-offset", "2000"]):
                with patch("MS1.evaluate._load_fixed", side_effect=[batch(1)] * 3) as load_fixed:
                    evaluate_main()

            self.assertEqual(load_fixed.call_args_list[0].kwargs["skip"], 2000)
            result = json.loads((run_dir / "eval_test_offset_2000.json").read_text())
            self.assertEqual(result["test"]["offset"], 2000)
            self.assertEqual((run_dir / "eval_fresh.json").read_text(), "existing")


class TestModel(unittest.TestCase):

    def test_output_shapes(self):
        X, _ = batch()
        model = small_model()

        self.assertEqual(model(X).shape, (4, 81, 9))
        self.assertEqual(model(X, return_all=True).shape, (4, 3, 81, 9))

    def test_predict_keeps_clues_and_range(self):
        X, _ = batch()
        pred = small_model().predict(X)

        self.assertTrue(torch.all((X == 0) | (pred == X)))
        self.assertTrue(torch.all((pred >= 1) & (pred <= 9)))

    def test_all_parameters_receive_gradients(self):
        X, Y = batch()
        model = small_model()
        sudoku_loss(model(X, return_all=True), X, Y).backward()

        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)

    def test_loss_decreases_when_overfitting(self):
        torch.manual_seed(0)
        X, Y = batch(n=8)
        model = small_model()
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)

        first = None
        for _ in range(150):
            loss = sudoku_loss(model(X, return_all=True), X, Y)
            first = loss.item() if first is None else first
            opt.zero_grad()
            loss.backward()
            opt.step()

        self.assertLess(loss.item(), 0.5 * first)

    def test_peer_mask_allows_21_cells_each(self):
        allowed = ~small_model().attn_mask

        self.assertTrue(torch.all(allowed.sum(dim=1) == 21))
        self.assertTrue(torch.equal(allowed, allowed.T))

    def test_decode_matches_predict(self):
        X, _ = batch()
        model = small_model().eval()

        torch.testing.assert_close(decode(model(X), X), model.predict(X))


class TestBatching(unittest.TestCase):

    def test_iter_batches_shapes_and_coverage(self):
        samples = [(np.full(81, i % 9 + 1), np.full(81, i % 9 + 1)) for i in range(10)]
        batches = list(iter_batches(iter(samples), batch_size=4, shuffle_buffer=5, seed=1))

        self.assertEqual([len(b[0]) for b in batches], [4, 4, 2])
        self.assertEqual(sum(len(b[0]) for b in batches), 10)


if __name__ == "__main__":
    unittest.main()
