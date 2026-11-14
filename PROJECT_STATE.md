# Project state

Last updated: 2026-11-14. See also [README.md](README.md) and [MS1/README.md](MS1/README.md).

## Milestone 1

- Topic: Flow Reasoning Models (arXiv 2606.29150) on Sudoku-Extreme.
- Report due 15.11 (max 10 pages, ISIS LaTeX template). Presentation 17.11.
- Code lives in `~/MS1` on Hydra and runs there via `apptainer` (`pml.sif` built from `pml.def`).
- Course rules: no code in the report, no notebooks, report contains an AI-usage section. AI assistants were used locally only; all cluster commands were run by team members.

## Team split

1. Data: loader, dataset on the cluster, validation; discussion and final pass of the report.
2. Exploration and visualization: `show()`, dataset-overview section.
3. Model and baselines: model, training, baselines, evaluation; baseline-and-evaluation section.

## Data representation

- `load_data` yields `(x, y)` as `(9, 9)` `uint8`, blank = 0. The transformation is applied to both splits, so it is only the fixed string-to-array conversion.
- Augmentation (Sudoku symmetries) is applied to the training stream only, via `model.augmented(...)` in `train.py`, never through `load_data`.
- The model works on 81 tokens (`model.to_flat`); the clue mask is `x > 0`.

## Code (`MS1/`, run as `python -m MS1.<module>` from the repo root)

| Module | Purpose |
|---|---|
| `data_loader.py` | `load_data` |
| `visualization.py` | `show()`: boards (9x9 or flat), optional clue shading and wrong-cell marking |
| `explore.py` | dataset statistics, leakage screen, Figures 1-4; `--nrows N` for a quick run (full run needs ~32 GB RAM) |
| `model.py` | recurrent transformer with peer attention, loss, augmentation |
| `train.py` | training loop; logs test, seen-train and unseen-train metrics to `log.csv` |
| `evaluate.py` | scores a saved model; `--test-offset 2000` for the second test slice; breakdowns by clues, rating, source, shared pattern |
| `baselines.py` | random, majority, naive candidates, backtracking |
| `metrics.py` | cell accuracy, exact match, valid-grid rate, breakdowns |
| `make_figures.py` | results table and Figures 5-7 from `MS1/results/` |
| `run_train.sbatch` | cluster runs (default partition `gpu-teaching-2h`) |

Tests: `python -m unittest discover -s MS1/tests` (27 tests, pass locally and on Hydra).

## Runs

All on test rows 1-2,000 unless noted; w = width, steps = refinement steps.

| Run directory | Config | Cell acc | Exact |
|---|---|---:|---:|
| `run_noaug` | w128, 8 steps, 100k x 3 ep | 0.576 | 0.037 |
| `run_aug` | same + augmentation | 0.578 | 0.038 |
| `run_long` | w192, 12 steps, 100k x 6 ep | 0.665 | 0.221 |
| `run_more_data` | w192, 12 steps, 300k x 3 ep | 0.728 | 0.329 |
| `run_more_data_seed1` | same, seed 1 | 0.730 | 0.336 |
| `run_more_data_aug` | same + augmentation | 0.721 | 0.317 |
| `run_more_data_long` | w192, 12 steps, 300k x 6 ep (final) | 0.769 | 0.428 |
| `run_more_data_long_seed1` | same, seed 1 | 0.769 | 0.416 |
| final, test rows 2,001-4,000 | mean of both seeds | 0.762 | 0.389 |

- Global attention stayed at chance; all runs above use peer attention (row, column, box).
- Final model: seen train 0.825 / 0.578, unseen train 0.738 / 0.356 (cell / exact).
- Baselines (test rows 1-2,000): naive candidates 0.427 / 0.118; backtracking solves 1,998 of 2,000 within its node budget.
- Exact by rating band 0-1 / 2-20 / 21-50 / 51+: 0.86 / 0.34 / 0.30 / 0.23.
- Leakage screen: up to 47.6% of test puzzles share a clue-pattern signature with train (upper bound); no identical puzzles. Shared puzzles are solved less often (0.17 vs 0.60), since they come mostly from the hardest sources.

## Hydra notes

- Login: `ssh -i $env:USERPROFILE\.ssh\id_rsa_hydra_work pml05@hydra.ml.tu-berlin.de`; copy code with `scp` from a local terminal.
- Dataset: `/home/space/datasets/sudoku_extreme/`.
- Use the `gpu-teaching-*` partitions (admin instruction); CPU jobs on `cpu-2h`.
- Building `pml.sif` needs `--mem=32G`.
- After copying `run_train.sbatch` from Windows: `sed -i 's/\r$//' MS1/run_train.sbatch`.

## Next steps

1. Submit report, prepare presentation.
2. Merge `feature/ms1-model-baselines` into `master`.
3. MS2: flow matching, self-conditioning, Fixed-Point Forcing; more refinement steps at test time; use the rule check to reject or repair outputs; 1,000-puzzle regime with augmentation.

## Known limitations

- The leakage screen compares clue patterns only, not digits.
- Two seeds for the final configuration.
- `MS1/results/` is generated output and not committed.
