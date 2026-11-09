# Project state (hand-over for new sessions)

Last updated: 2026-11-09. Read this first, then [README.md](README.md) and [MS1/README.md](MS1/README.md).

## Goal and deadlines

- Course project on Flow Reasoning Models (arXiv 2606.29150), Milestone 1: data, visualization, prototype model, baselines, report.
- Report due 15.11 (max 10 pages, LaTeX template from ISIS, no code in the report). Presentation 17.11; our group presents MS1.
- Code must be in `~/MS1` on the Hydra cluster, executable there, no notebooks.

## Rules

- Coding agents are prohibited on Hydra. Agents work locally only; the user runs cluster commands manually (ssh, sbatch, apptainer build). Agents may only write commands for the user to run.
- The report needs an AI-usage section.
- Do not create extra documentation files unless asked.

## Team split

1. Data (done): loader, dataset on cluster, validation. Also writes the discussion, AI-usage section and final consistency pass of the report.
2. Exploration and visualization (done by a teammate): `show()` and the dataset-overview report section.
3. Model and baselines (this session's work, code done, not yet validated on the cluster): model, loss, training loop, baselines, baseline-and-evaluation report section (3 points).

## Data representation (agreed)

- Loader yields `(x, y)` as `(9, 9)` `uint8`, blanks `0`. Part 1 did not add the originally planned augmentation, mask or rating output.
- Model side: `MS1/model.py::to_flat` gives 81 `int64` digits; clue mask is `x > 0`.
- `augment` is optional, training only, lives in `model.py` (part 1 was not reopened).
- `show()` takes `(9, 9)` boards, so model predictions need `reshape(9, 9)` before plotting.

## Code map (all in `MS1/`)

- `model.py`: `Model` (embedding of digit + row/col/box position, one shared transformer block iterated `n_steps` times, 9-way head per cell), `decode`, `sudoku_loss` (cross-entropy on blank cells, averaged over steps).
- `train.py`: AdamW, warmup + cosine, grad clipping, shuffle-buffer batching, logs train loss plus a probe of seen training puzzles versus the test slice (overfitting check) to `log.csv`; writes `results.json` and `model.pt`.
- `baselines.py`: random, majority digit, naive candidates (naked and hidden singles, no search), backtracking solver (MRV); CLI writes `baselines.json`.
- `metrics.py`: cell accuracy on blanks, exact match, valid-grid rate, clue-count bucket breakdown.
- `run_train.sbatch`: `smoke` (gpu-test, 15 min) and `full` modes, via `apptainer run --nv`.
- Tests: `MS1/tests/` (24 tests, all pass). Run: `python -m unittest discover -s MS1/tests -v` from the repo root (do not pass `-t .`).
- Always run modules as `python -m MS1.<module>` from the repo root.

## Results so far

Small local run (200 test puzzles; backtracking on 50):

| Method | Cell acc | Exact |
|---|---|---|
| Random | 0.110 | 0.000 |
| Majority digit | 0.112 | 0.000 |
| Naive candidates | 0.452 | 0.175 |
| Backtracking | 1.000 | 1.000 |

Model: the cluster smoke test (313 steps, global attention) stayed at chance: test loss 2.182 (ln 9 = 2.197), test cell acc 0.138, exact 0. Local runs showed the same. Cause suspected: with full attention the model cannot easily discover the row/column/box rules.

Change after that: `Model(peer_attention=True)` is now the default (each cell attends only to itself and its 20 row/column/box peers); `--global-attention` in `train.py` restores the old behaviour. Local CPU check (600 steps, d_model 96, 1 layer, 6 steps, lr 2e-3, 38.4k puzzles, 300 test): test loss 1.087, test cell acc 0.510, exact 0.02, valid 0.02; probe (seen) cell acc 0.493, so no overfitting gap. Still to be confirmed on the cluster.

Cluster smoke run with peer attention (1000 steps, 64k puzzles, lr 2e-3, default model d128/2 layers/8 steps): test loss 1.038, test cell acc 0.530, exact 0.026, valid 0.026; probe (seen) cell acc 0.513, exact 0.022, so no overfitting gap. Learning is steady but slow and the learning-rate schedule had already decayed to 0.

Cluster smoke baselines (200 test puzzles; backtracking on 50): naive candidates 0.452 cell acc / 0.175 exact (2.8 ms/puzzle), backtracking 1.000 / 1.000 (73 ms/puzzle), random 0.110, majority 0.112.

## Cluster status (Hydra)

- Login: `ssh -i $env:USERPROFILE\.ssh\id_rsa_hydra_work pml05@hydra.ml.tu-berlin.de` (key needs the passphrase typed by the user). Files copied with `scp -i ... -r MS1 pml.def requirements.txt pml05@hydra.ml.tu-berlin.de:~/`.
- Dataset at `/home/space/datasets/sudoku_extreme/` is present.
- `~/pml.sif` is built (needed `--mem=32G`; the first attempt with default memory died with exit 137). GPU check printed `True` on a `gpu-test` node (head058). Rebuild recipe: `srun --partition=cpu-2h --mem=32G --cpus-per-task=4 --pty bash`, then `cd ~` and `apptainer build pml.sif pml.def`.
- Type commands one at a time on the node; text pasted before the node shell is ready is lost.
- Partitions: cpu-9m/2h/5h/2d/7d, gpu-9m/2h/5h/2d/7d, cpu-test, gpu-test. `gpu-test` has no 40 GB A100 node, so the sbatch script requests `--gpus=1` (an untyped GPU; do not combine it with `--gpus-per-node=<type>:N`).
- `run_train.sbatch` may need `sed -i 's/\r$//'` after copying from Windows.
- The smoke job ran end to end on the cluster (baselines and 313 training steps finished inside the 15-minute limit).

## Next steps

1. Re-copy `MS1/` to the cluster (model.py, train.py changed) and rerun the smoke job with peer attention; check that `test_loss` falls clearly below 2.18.
2. If still at chance: higher `--lr`, more `--n-steps`/`--n-layers`, longer run, more data.
3. Check whether `train.csv` is ordered (source/rating); the first 100k rows may be a biased subset.
4. `sbatch MS1/run_train.sbatch full` (baselines, augmented and non-augmented runs; 2 h limit on gpu-2h, measure step speed first), copy `MS1/results/` back with scp.
5. Figures for the report: loss and accuracy curves (`log.csv`), accuracy by clue bucket, failure cases via `show()`.
6. Write the baseline-and-evaluation section: baselines, metrics (why binary metrics do not fit), results by clue bucket, overfitting check (probe vs test), non-technical view. Hand results to part 1 for the discussion.

## Known gaps

- Loader has no rating/source output, so difficulty breakdown uses clue-count buckets only.
- Near-duplicate/equivalent-puzzle leakage between train and test is not checked (exact overlaps: 0).
- `MS1/results/` is generated output and should not be committed.
