import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def to_flat(x, y):
    """(9, 9) grids -> 81 integers each; the clue mask is x > 0."""
    return x.reshape(81).astype(np.int64), y.reshape(81).astype(np.int64)


def _axis_permutation(rng):
    # Permute the three bands, then the rows inside every band.
    return np.concatenate([b * 3 + rng.permutation(3) for b in rng.permutation(3)])


def augment(x, y, rng):
    """Validity-preserving symmetry of a (9, 9) puzzle/solution pair."""
    rows, cols = _axis_permutation(rng), _axis_permutation(rng)
    x, y = x[rows][:, cols], y[rows][:, cols]
    if rng.random() < 0.5:
        x, y = x.T, y.T
    digits = np.concatenate([[0], rng.permutation(np.arange(1, 10))]).astype(np.int64)
    return digits[x], digits[y]


def make_transform(use_augmentation=False, seed=0):
    rng = np.random.default_rng(seed)

    def transform(x, y):
        if use_augmentation:
            x, y = augment(x, y, rng)
        return to_flat(x, y)

    return transform


def augmented(samples, seed=0):
    """Augment and flatten a (9, 9) (x, y) stream; apply to the training stream only, never via load_data."""
    rng = np.random.default_rng(seed)
    for x, y in samples:
        yield to_flat(*augment(x, y, rng))


class Model(torch.nn.Module):
    """Recurrent transformer: one shared block iterated n_steps times over the 81 cells."""

    def __init__(self, d_model=128, n_heads=4, n_layers=2, n_steps=8, ff_mult=4, dropout=0.0,
                 peer_attention=True):
        super().__init__()
        self.n_steps = n_steps
        self.peer_attention = peer_attention
        self.digit_emb = nn.Embedding(10, d_model)
        self.row_emb = nn.Parameter(torch.randn(9, d_model) * 0.02)
        self.col_emb = nn.Parameter(torch.randn(9, d_model) * 0.02)
        self.box_emb = nn.Parameter(torch.randn(9, d_model) * 0.02)
        self.z0 = nn.Parameter(torch.zeros(d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * ff_mult, dropout, batch_first=True, norm_first=True
        )
        self.block = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 9)

        idx = torch.arange(81)
        self.register_buffer("rows", idx // 9, persistent=False)
        self.register_buffer("cols", idx % 9, persistent=False)
        self.register_buffer("boxes", (idx // 27) * 3 + (idx % 9) // 3, persistent=False)
        # True = attention blocked; a cell may attend to itself and its 20 row/column/box peers.
        same = (
            (self.rows[:, None] == self.rows[None, :])
            | (self.cols[:, None] == self.cols[None, :])
            | (self.boxes[:, None] == self.boxes[None, :])
        )
        self.register_buffer("attn_mask", ~same, persistent=False)

    def forward(self, X, return_all=False):
        """X: (B, 81) digits 0-9. Returns logits (B, 81, 9), or (B, n_steps, 81, 9) if return_all."""
        pos = self.row_emb[self.rows] + self.col_emb[self.cols] + self.box_emb[self.boxes]
        inp = self.digit_emb(X) + pos
        z = self.z0.expand_as(inp)
        logits = []
        mask = self.attn_mask if self.peer_attention else None
        for _ in range(self.n_steps):
            z = self.block(z + inp, mask=mask)
            logits.append(self.head(self.norm(z)))
        out = torch.stack(logits, dim=1)
        return out if return_all else out[:, -1]

    @torch.no_grad()
    def predict(self, X):
        return decode(self(X), X)


def decode(logits, X):
    """(B, 81, 9) logits -> (B, 81) digits 1-9, with the given clues kept."""
    return torch.where(X > 0, X, logits.argmax(dim=-1) + 1)


def sudoku_loss(logits, X, Y):
    """Cross-entropy on blank cells, averaged over recurrence steps (deep supervision)."""
    if logits.dim() == 3:
        logits = logits.unsqueeze(1)
    B, T = logits.shape[:2]
    target = (Y - 1).unsqueeze(1).expand(B, T, 81)
    ce = F.cross_entropy(logits.reshape(-1, 9), target.reshape(-1), reduction="none")
    blank = (X == 0).unsqueeze(1).to(ce.dtype).expand(B, T, 81)
    return (ce.view(B, T, 81) * blank).sum() / blank.sum().clamp_min(1)
