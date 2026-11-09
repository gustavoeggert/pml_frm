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
- `augment` is optional, training only, lives in `model.py` (part 1 was not reopened). RULE: `load_data` applies its `transformation` to BOTH the train and the test generator, so augmentation must never go through it. It is applied to the training stream only via `model.augmented(...)` (used in `train.py`); test and probe sets use `to_flat` only.
- `show()` takes `(9, 9)` boards, so model predictions need `reshape(9, 9)` before plotting.

## Code map (all in `MS1/`)

- `model.py`: `Model` (embedding of digit + row/col/box position, one shared transformer block iterated `n_steps` times, 9-way head per cell), `decode`, `sudoku_loss` (cross-entropy on blank cells, averaged over steps).
- `train.py`: AdamW, warmup + cosine, grad clipping, shuffle-buffer batching, logs train loss plus a probe of seen training puzzles versus the test slice (overfitting check) to `log.csv`; writes `results.json` and `model.pt`.
- `baselines.py`: random, majority digit, naive candidates (naked and hidden singles, no search), backtracking solver (MRV); CLI writes `baselines.json`.
- `metrics.py`: cell accuracy on blanks, exact match, valid-grid rate, clue-count bucket breakdown.
- `run_train.sbatch`: modes `smoke` (gpu-test, 15 min), `baselines`, `noaug`, `aug`, `long`, via `apptainer run --nv`.
- `make_figures.py`: `python -m MS1.make_figures` builds `MS1/figures/` (learning_curves, model_vs_baselines, clue_buckets, example_*.png, results_table.csv) from `MS1/results/` copied from Hydra; also re-evaluates the saved `run_noaug` model locally as a check (matched the cluster numbers exactly).
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

Cluster `noaug` run (job 5202373, gpu-test on head058, 7:47 total, ~10 steps/s; 100k puzzles, 3 epochs, 4689 steps, default model d128/2 layers/8 steps, lr 2e-3, peer attention): final test loss 0.970, test cell acc 0.5756, exact 0.037, valid 0.037; probe (seen puzzles) cell acc 0.620, probe loss 0.870. Test cell acc was 0.509 at step 500 and 0.530 after the 1000-step smoke run. A train/test gap appeared after epoch 1 (it was about zero in epoch 1). Cluster `aug` run (job 5202374, same settings plus `--augment`, 8:10): final test loss 0.9625, test cell acc 0.5779, exact 0.0375, probe cell acc 0.617. Augmentation made no measurable difference to test accuracy, and the probe-vs-test gap (about 0.04) stayed, so it is NOT established that the gap is overfitting from repeated puzzles; it may be a train/test distribution difference. To settle it, evaluate on fresh train puzzles the model never saw (e.g. rows 100k-102k of train.csv). `long` mode: the 300k x 2 change was undone by the user, so the local `run_train.sbatch` again matches Hydra and the queued job 5202375 (100k puzzles x 6 epochs, d_model 192, 2 layers, 12 steps, lr 2e-3). Decision (user): leave the queued job as is.

Cluster baselines on 1000 test puzzles (backtracking on 300), finished 2026-11-09 18:46 as a CPU job (`~/MS1/results/baselines.json` on Hydra): random 0.110 cell acc, majority 0.111, naive candidates 0.420 cell acc / 0.110 exact (0.6 ms/puzzle), backtracking 1.000 / 1.000 (17.9 ms/puzzle).

Job queue at 18:47 (user left; jobs may start while away): 5202079 `noaug`, 5202080 `aug` (gpu-2h), 5202081 `long` (gpu-5h), all pending on the typed 40 GB GPU. Speed from the smoke job: baselines plus 1000 default-model steps took about 2 min, so `noaug`/`aug` should need roughly 10 min and `long` roughly 1 h once running. If the queue stays stuck, cancel them and use `sbatch --partition=gpu-test --gpus-per-node=1 MS1/run_train.sbatch noaug|aug` (15 min limit) and `sbatch --time=01:30:00 MS1/run_train.sbatch long`. Baselines can run on a CPU node: `sbatch --partition=cpu-2h --cpus-per-task=2 --mem=8G --chdir=$HOME --wrap="apptainer run --bind /home/space/datasets $HOME/pml.sif python -m MS1.baselines --n-test 1000 --n-solver 300 --out MS1/results/baselines.json"`.

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
4. Submit three separate jobs on gpu-2h: `sbatch MS1/run_train.sbatch baselines`, `... noaug`, `... aug` (100k puzzles, 3 epochs, lr 2e-3). If a job ends in TIMEOUT, rerun it with `--partition=gpu-5h`. Copy `MS1/results/` back with scp.
5. Figures for the report: loss and accuracy curves (`log.csv`), accuracy by clue bucket, failure cases via `show()`.
6. Write the baseline-and-evaluation section: baselines, metrics (why binary metrics do not fit), results by clue bucket, overfitting check (probe vs test), non-technical view. Hand results to part 1 for the discussion.

## Known gaps

- Loader has no rating/source output, so difficulty breakdown uses clue-count buckets only.
- Near-duplicate/equivalent-puzzle leakage between train and test is not checked (exact overlaps: 0).
- `MS1/results/` is generated output and should not be committed.
