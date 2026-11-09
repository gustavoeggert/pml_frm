import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from MS1.data_loader import load_data
from MS1.metrics import MetricAccumulator
from MS1.model import Model, decode, make_transform, sudoku_loss


def _shuffled(samples, buffer_size, rng):
    if buffer_size <= 1:
        yield from samples
        return
    buf = []
    for s in samples:
        buf.append(s)
        if len(buf) >= buffer_size:
            j = rng.randrange(len(buf))
            buf[j], buf[-1] = buf[-1], buf[j]
            yield buf.pop()
    rng.shuffle(buf)
    yield from buf


def _collate(batch):
    xs, ys = zip(*batch)
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def iter_batches(samples, batch_size, shuffle_buffer=0, seed=0):
    """Batch a (x, y) generator into (B, 81) tensors with an approximate streaming shuffle."""
    batch = []
    for s in _shuffled(samples, shuffle_buffer, random.Random(seed)):
        batch.append(s)
        if len(batch) == batch_size:
            yield _collate(batch)
            batch = []
    if batch:
        yield _collate(batch)


def _load_fixed(n, split, data_dir):
    kwargs = {"n_train": n, "n_test": 0} if split == "train" else {"n_train": 0, "n_test": n}
    gen = load_data(transformation=make_transform(False), data_dir=data_dir, **kwargs)[
        0 if split == "train" else 1
    ]
    return _collate(list(gen))


@torch.no_grad()
def evaluate(model, X, Y, device, batch_size=256):
    model.eval()
    acc = MetricAccumulator()
    loss_sum = 0.0
    for i in range(0, len(X), batch_size):
        x, y = X[i:i + batch_size].to(device), Y[i:i + batch_size].to(device)
        final = model(x)
        loss_sum += sudoku_loss(final, x, y).item() * len(x)
        acc.update(decode(final, x).cpu().numpy(), x.cpu().numpy(), y.cpu().numpy())
    return loss_sum / len(X), acc


def main():
    p = argparse.ArgumentParser(description="Train the recurrent Sudoku prototype.")
    p.add_argument("--n-train", type=int, default=100_000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--n-probe", type=int, default=2000, help="seen training puzzles used for the overfitting check")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--shuffle-buffer", type=int, default=10_000)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-steps", type=int, default=8)
    p.add_argument("--global-attention", action="store_true", help="let every cell attend to all cells")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default="MS1/results/run")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_X, test_Y = _load_fixed(args.n_test, "test", args.data_dir)
    probe_X, probe_Y = _load_fixed(min(args.n_probe, args.n_train), "train", args.data_dir)

    model = Model(
        args.d_model, args.n_heads, args.n_layers, args.n_steps,
        peer_attention=not args.global_attention,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * math.ceil(args.n_train / args.batch_size)

    def lr_scale(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)

    fields = [
        "step", "epoch", "lr", "train_loss_all_steps",
        "probe_loss", "probe_cell_acc", "probe_exact",
        "test_loss", "test_cell_acc", "test_exact", "test_valid",
    ]
    log_file = (out_dir / "log.csv").open("w", newline="")
    log = csv.DictWriter(log_file, fieldnames=fields)
    log.writeheader()

    def log_eval(step, epoch, running):
        probe_loss, probe = evaluate(model, probe_X, probe_Y, device)
        test_loss, test = evaluate(model, test_X, test_Y, device)
        ps, ts = probe.summary(), test.summary()
        row = {
            "step": step, "epoch": epoch, "lr": opt.param_groups[0]["lr"],
            "train_loss_all_steps": float(np.mean(running)),
            "probe_loss": probe_loss, "probe_cell_acc": ps["cell_acc"], "probe_exact": ps["exact"],
            "test_loss": test_loss, "test_cell_acc": ts["cell_acc"],
            "test_exact": ts["exact"], "test_valid": ts["valid"],
        }
        log.writerow(row)
        log_file.flush()
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

    step, last_logged, running = 0, 0, []
    for epoch in range(args.epochs):
        train_gen, _ = load_data(
            transformation=make_transform(args.augment, args.seed + epoch),
            n_train=args.n_train, n_test=0, data_dir=args.data_dir,
        )
        for X, Y in iter_batches(train_gen, args.batch_size, args.shuffle_buffer, args.seed + epoch):
            model.train()
            X, Y = X.to(device), Y.to(device)
            loss = sudoku_loss(model(X, return_all=True), X, Y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            running.append(loss.item())
            step += 1
            if step % args.eval_every == 0:
                log_eval(step, epoch, running)
                last_logged, running = step, []
            if step >= total_steps:
                break
    if last_logged != step:
        log_eval(step, epoch, running or [float("nan")])
    log_file.close()

    _, test = evaluate(model, test_X, test_Y, device)
    results = {"args": vars(args), "summary": test.summary(), "by_clue_bucket": test.by_clue_bucket()}
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, out_dir / "model.pt")


if __name__ == "__main__":
    main()
