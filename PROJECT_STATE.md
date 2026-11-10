# Project state (hand-over for new sessions)

Last updated: 2026-11-10. Read this first, then [README.md](README.md) and [MS1/README.md](MS1/README.md).

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
3. Model and baselines (cluster experiments completed; further runs and report work ongoing): model, loss, training loop, baselines, baseline-and-evaluation report section (3 points).

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
- `run_train.sbatch`: default partition `gpu-teaching-2h`; modes `smoke`, `baselines`, `noaug`, `aug`, `long`, `aug300` (300k x 3 epochs, augmentation, `run_more_data_aug`), `long300` (300k x 6 epochs, `run_more_data_long`), via `apptainer run --nv`.
- `train.py --n-fresh N` (default 2000): also logs loss/accuracy on N subsequent train.csv rows as `fresh_*` columns. `evaluate.py --run-dir DIR` scores a saved `model.pt` on test, seen-train and fresh-train puzzles and writes `eval_fresh.json`; used for runs finished before `--n-fresh` existed. `--test-offset 2000 --n 2000` instead scores test rows 2,001-4,000 and writes `eval_test_offset_2000.json`; this slice was evaluated for the selected 6-epoch configuration. Three of its rows were previously used in a local toy-model smoke test, and overlap with earlier rows has not been checked. The fresh comparison is informative but cannot by itself separate memorisation from dataset ordering, distribution differences or equivalent puzzles.
- `make_figures.py`: `python -m MS1.make_figures` builds `MS1/figures/` (learning_curves, model_vs_baselines, clue_buckets, example_*.png, results_table.csv) from `MS1/results/` copied from Hydra; also re-evaluates the saved `run_noaug` model locally as a check (matched the cluster numbers exactly).
- Tests: run `python -m unittest discover -s MS1/tests -v` from the repo root (do not pass `-t .`). The 10 focused model tests passed locally after restoring `make_transform` for compatibility with the Hydra tests.
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

Change after that: `Model(peer_attention=True)` is now the default (each cell attends only to itself and its 20 row/column/box peers); `--global-attention` in `train.py` restores the old behaviour. Local CPU check (600 steps, d_model 96, 1 layer, 6 steps, lr 2e-3, 38.4k puzzles, 300 test): test loss 1.087, test cell acc 0.510, exact 0.02, valid 0.02; probe (seen) cell acc 0.493, so no overfitting gap. Subsequent cluster runs also used peer attention.

Cluster smoke run with peer attention (1000 steps, 64k puzzles, lr 2e-3, default model d128/2 layers/8 steps): test loss 1.038, test cell acc 0.530, exact 0.026, valid 0.026; probe (seen) cell acc 0.513, exact 0.022, so no overfitting gap. Learning is steady but slow and the learning-rate schedule had already decayed to 0.

Cluster `noaug` run (job 5202373, gpu-test on head058, 7:47 total, ~10 steps/s; 100k puzzles, 3 epochs, 4689 steps, default model d128/2 layers/8 steps, lr 2e-3, peer attention): final test loss 0.970, test cell acc 0.5756, exact 0.037, valid 0.037; probe (seen puzzles) cell acc 0.620, probe loss 0.870. Test cell acc was 0.509 at step 500 and 0.530 after the 1000-step smoke run. A train/test gap appeared after epoch 1 (it was about zero in epoch 1). Cluster `aug` run (job 5202374, same settings plus `--augment`, 8:10): final test loss 0.9625, test cell acc 0.5779, exact 0.0375, probe cell acc 0.617. These single runs do not establish whether augmentation affects generalization or why the probe-vs-test gap appears. The original long job 5202375 was held; check its current status before cancelling it.

Cluster baselines on 1000 test puzzles (backtracking on 300), finished 2026-11-09 18:46 as a CPU job (`~/MS1/results/baselines.json` on Hydra): random 0.110 cell acc, majority 0.111, naive candidates 0.420 cell acc / 0.110 exact (0.6 ms/puzzle), backtracking 1.000 / 1.000 (17.9 ms/puzzle).

Larger runs finished 2026-11-10 on head025 (80 GB A100, `gpu-teaching-2h`/`-5h`; both same settings except data size, d_model 192, 2 layers, 12 steps, lr 2e-3, warmup 300, 2000 test puzzles, no augmentation). Logs are in `~/ms1_frm_5203618.log` and `~/ms1_more_data_5206769.log` on Hydra; results are in `MS1/results/run_long` and `MS1/results/run_more_data` on Hydra and NOT yet copied locally (`make_figures.py` does not read them yet).

| Run | Job | Steps | Time | Test loss | Test cell acc | Test exact | Probe (seen) cell acc | Probe exact |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `noaug` (100k x 3, d128, 8 steps) | 5202373 | 4689 | 7:47 | 0.970 | 0.576 | 0.037 | 0.620 | n/a |
| `long` (100k x 6, d192, 12 steps) | 5203618 | 9378 | 13:40 | 0.860 | 0.665 | 0.221 | 0.767 | 0.421 |
| `more_data` (300k x 3, d192, 12 steps) | 5206769 | 14064 | 20:09 | 0.675 | 0.728 | 0.329 | 0.791 | 0.488 |
| `long300` (300k x 6, d192, 12 steps) | 5208128 | 28128 | 39:00 | 0.566 | 0.769 | 0.428 | 0.825 | 0.578 |

Reading: the larger model and longer training together improved over `noaug`, while `more_data` improved further. These comparisons change multiple factors, so they do not establish which caused the improvement. The naive baseline reached 0.420 cell accuracy / 0.110 exact on 1,000 test puzzles; backtracking reached 1.000 on 300. Exact match rose late in `long`, but the logs alone do not establish the reasoning mechanism or convergence.

Follow-up runs: `aug300` (5208127) completed in 22:20 with test cell accuracy 0.7207 and exact 0.3165, versus 0.7281 and 0.3285 for the unaugmented `more_data` run. This single-seed comparison does not establish a general augmentation effect. The fresh-puzzle evaluation (5208129) completed: for `long`, seen/fresh/test cell accuracy was 0.7666/0.6807/0.6646; for `more_data`, 0.7913/0.7000/0.7281. `long300` (5208128) completed successfully in 39:00: final test cell accuracy 0.7685, exact 0.4275, seen-train cell accuracy 0.8252 and fresh-train cell accuracy 0.7378 (fresh exact 0.356). The 3- versus 6-epoch comparison changes the number of steps and the learning-rate schedule; it is a single-seed result on the same test slice. The initial jobs 5207391, 5207392 and 5207394 failed on an incompatible Hydra `model.py`; the compatible file was copied and the import check 5208125 passed before resubmission.

Seed-1 repeats reached their final logged training steps: 3 epochs (5208435, `run_more_data_seed1`) had test cell accuracy 0.7296, exact 0.3355 and fresh cell accuracy 0.7032; 6 epochs (5208439, `run_more_data_long_seed1`) had test cell accuracy 0.7693, exact 0.4155 and fresh cell accuracy 0.7389. Their Slurm exit statuses were not reported; the 6-epoch checkpoint was subsequently loaded successfully. Across seeds 0 and 1, exact match was 0.3285/0.3355 for 3 epochs and 0.4275/0.4155 for 6 epochs. Both favor the longer schedule on the same monitored test slice; two seeds are insufficient to establish robust uncertainty.

Final-slice evaluation job 5208581 completed successfully in 3:11 on test rows 2,001-4,000 (n=2,000). For the selected 300k, 6-epoch configuration, seed 0 achieved cell accuracy 0.76217 and exact 0.3945; seed 1 achieved cell accuracy 0.76092 and exact 0.3830. Their means are 0.76155 and 0.38875 respectively. These are lower than the monitored-slice results (exact 0.4275/0.4155). The slice is not guaranteed independent: three rows were used for a toy-model smoke test, and equivalent puzzle overlap and ordering effects are unverified. Outputs are `eval_test_offset_2000.json` in `run_more_data_long` and `run_more_data_long_seed1` on Hydra.

Baselines can run on a CPU node: `sbatch --partition=cpu-2h --cpus-per-task=2 --mem=8G --chdir=$HOME --wrap="apptainer run --bind /home/space/datasets $HOME/pml.sif python -m MS1.baselines --n-test 1000 --n-solver 300 --out MS1/results/baselines.json"`.

Cluster smoke baselines (200 test puzzles; backtracking on 50): naive candidates 0.452 cell acc / 0.175 exact (2.8 ms/puzzle), backtracking 1.000 / 1.000 (73 ms/puzzle), random 0.110, majority 0.112.

## Cluster status (Hydra)

- Login: `ssh -i $env:USERPROFILE\.ssh\id_rsa_hydra_work pml05@hydra.ml.tu-berlin.de` (key needs the passphrase typed by the user). Files copied with `scp -i ... -r MS1 pml.def requirements.txt pml05@hydra.ml.tu-berlin.de:~/`.
- Dataset at `/home/space/datasets/sudoku_extreme/` is present.
- `~/pml.sif` is built (needed `--mem=32G`; the first attempt with default memory died with exit 137). GPU check printed `True` on a `gpu-test` node (head058). Rebuild recipe: `srun --partition=cpu-2h --mem=32G --cpus-per-task=4 --pty bash`, then `cd ~` and `apptainer build pml.sif pml.def`.
- Type commands one at a time on the node; text pasted before the node shell is ready is lost.
- Partitions: cpu-9m/2h/5h/2d/7d, gpu-9m/2h/5h/2d/7d, cpu-test, gpu-test. `gpu-test` has no 40 GB A100 node, so the sbatch script requests `--gpus=1` (an untyped GPU; do not combine it with `--gpus-per-node=<type>:N`).
- IMPORTANT: the administrator directed account `pml` to use `gpu-teaching-9m/2h/5h/2d/7d`, not regular `gpu-*` partitions. The 2h and 5h teaching partitions showed the same five nodes on 2026-11-10: head022 MIG, head025 80 GB A100, head034/035 RTX 6000 and head055 RTX 3090. Head034/035 and head022 were unavailable at that time. `run_train.sbatch` now defaults to `gpu-teaching-2h`.
- A pending job can be moved without resubmitting: `scontrol update JobId=<id> Partition=gpu-teaching-2h`.
- `run_train.sbatch` may need `sed -i 's/\r$//'` after copying from Windows.
- The smoke job ran end to end on the cluster (baselines and 313 training steps finished inside the 15-minute limit).

## Next steps

1. Copy `MS1/results/` back from Hydra (new `run_long` and `run_more_data`), cancel the stale held job 5202375 if it still exists (`scancel 5202375`), and add the two runs to `make_figures.py` (learning curves, results table, clue-bucket plot).
2. Copy the completed follow-up outputs from Hydra and update the report and figures with their results. Freeze experiments in time for the 15.11 report deadline.
3. Check whether `train.csv` is ordered (source/rating); the first 100k-300k rows may be a biased subset.
4. Figures for the report: loss and accuracy curves (`log.csv`), accuracy by clue bucket, failure cases via `show()`.
5. Write the baseline-and-evaluation section: baselines, metrics (why binary metrics do not fit), results by clue bucket, overfitting check (probe vs test), non-technical view. Hand results to part 1 for the discussion.

## Known gaps

- Loader has no rating/source output, so difficulty breakdown uses clue-count buckets only.
- Near-duplicate/equivalent-puzzle leakage between train and test is not checked (exact overlaps: 0).
- `MS1/results/` is generated output and should not be committed.
