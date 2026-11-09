import unittest

import numpy as np
import torch

from MS1.metrics import count_violations
from MS1.model import Model, augment, augmented, decode, sudoku_loss, to_flat
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
