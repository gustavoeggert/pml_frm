# pml_frm
University project on the paper **"Flow Reasoning Models: Turning Flows Into Efficient Recurrent Reasoners"** (Helbling, Bryutkin et al., arXiv 2606.29150). The first milestone (MS1) works on Sudoku-Extreme: a puzzle is given and a model has to produce its unique solution.

Current status and hand-over notes for new sessions: [PROJECT_STATE.md](PROJECT_STATE.md).

## Repository layout

| Path | Content |
|---|---|
| `MS1/data_loader.py` | `load_data()`: lazy train/test generators over the Sudoku-Extreme CSVs |
| `MS1/validate_dataset.py` | Full-dataset integrity check (run on the cluster) |
| `MS1/visualization.py` | `show()`: renders Sudoku boards, optionally to a file |
| `MS1/model.py` | Transforms (`to_flat`, `augment`, `make_transform`), `Model`, `decode`, `sudoku_loss` |
| `MS1/train.py` | Training loop, evaluation, logging and checkpointing |
| `MS1/baselines.py` | Random, majority-digit, naive-candidate and backtracking baselines |
| `MS1/metrics.py` | Cell accuracy, exact match, rule violations, clue-bucket breakdown |
| `MS1/run_train.sbatch` | Hydra job script (`smoke` and `full` modes) |
| `MS1/tests/` | Unit tests |
| `pml.def` | Apptainer recipe for the Hydra container |
| `data/` | Local CSVs (not for the cluster) |

Details for MS1 are in [MS1/README.md](MS1/README.md).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Run all modules from the repository root with `python -m`, for example `python -m MS1.train`.

## Quick start

```bash
# unit tests
python -m unittest discover -s MS1/tests -v

# non-learned baselines on a small slice
python -m MS1.baselines --n-fit 2000 --n-test 200 --n-solver 50 --out MS1/results/baselines.json

# short local training run
python -m MS1.train --n-train 3000 --n-test 200 --epochs 2 --out-dir MS1/results/smoke
```

```python
from MS1.data_loader import load_data
from MS1.model import make_transform

train, test = load_data(transformation=make_transform(use_augmentation=True), n_train=1000, n_test=100)
x, y = next(train)   # two int64 arrays of 81 digits; 0 marks a blank, clue mask = x > 0
```

## Data representation

- The loader yields `(9, 9)` `uint8` grids; blanks are `0`.
- Model code flattens to 81 `int64` values with `to_flat`; the clue mask is `x > 0`.
- `augment` (digit permutation, band/stack and row/column permutations, transpose) is optional and for training only.

## Working on the cluster (Hydra, TU Berlin)

- Dataset: `/home/space/datasets/sudoku_extreme/` (`train.csv`, `test.csv`).
- Code lives in the home folder as `~/MS1/`; the container is `~/pml.sif`, built from `pml.def`.
- Jobs go through SLURM. Never run heavy work on the login node.
- Run a short test first, then the full run:

```bash
sbatch --partition=gpu-test MS1/run_train.sbatch smoke
sbatch MS1/run_train.sbatch full
```

- Building the container needs memory: `srun --partition=cpu-2h --mem=32G --cpus-per-task=4 --pty bash`, then `apptainer build pml.sif pml.def`.
- Coding agents must not be used on Hydra (course rules). Run cluster commands manually.

## Team split for MS1

1. **Data:** loader, dataset on the cluster, discussion and AI-usage sections of the report, final consistency pass.
2. **Exploration and visualization:** `show()`, dataset-overview section of the report.
3. **Model and baselines:** model, loss, training loop, baselines, baseline-and-evaluation section of the report.

Deadlines: report 15.11, presentation 17.11 (our group presents MS1).
