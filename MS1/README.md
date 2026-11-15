# Milestone 1 — Sudoku-Extreme: Data, Baselines and Prototype Model
**Components:** Data loading, visualization, model prototype, baselines. Sections 1–8 describe the data component; section 9 describes the model and baselines.

## 1. Overview

This module provides a memory-efficient data-loading pipeline for the Sudoku-Extreme dataset, used to reproduce and investigate Flow Reasoning Models.

The implementation supports:

- Lazy loading of Sudoku puzzles and solutions from CSV files
- Official training and test splits
- Optional preprocessing transformations
- Configurable sample limits
- Execution on both local machines and the TU Berlin Hydra cluster
- Dataset integrity validation

## 2. Dataset

We use the Sudoku-Extreme dataset from the [Sapient Intelligence Hugging Face repository](https://huggingface.co/datasets/sapientinc/sudoku-extreme).

The full dataset is stored on Hydra:

`/home/space/datasets/sudoku_extreme/`

| File | Number of samples |
|---|---:|
| train.csv | 3,831,994 |
| test.csv | 422,786 |
| **Total** | **4,254,780** |

Each CSV contains the following fields:

| Column | Description |
|---|---|
| `question` | Sudoku puzzle encoded as an 81-character string |
| `answer` | Complete 81-character solution |
| `source` | Original puzzle source |
| `rating` | Puzzle difficulty rating |

Empty cells are represented by `.` or `0`.

### Storage format

We use **raw CSV with streaming** rather than memory-mapped NumPy arrays.

This decision provides straightforward compatibility with the original dataset and avoids an additional conversion step.

Samples are read sequentially using Python's `csv.DictReader`. The loader does not load the complete dataset into memory.

## 3. Data Loader

**Implementation:** `MS1/data_loader.py`

### Function interface

```python
def load_data(
    dataset="sudoku_extreme",
    transformation=None,
    n_train=None,
    n_test=None,
    data_dir=None,
):
```

### Parameters

| Parameter | Description |
|---|---|
| `dataset` | Dataset name, default `sudoku_extreme` |
| `transformation` | Optional callable applied to each `(x, y)` pair |
| `n_train` | Maximum number of training samples |
| `n_test` | Maximum number of test samples |
| `data_dir` | Optional directory containing the CSV files |

### Output

The function returns two Python generators:

`train_generator, test_generator`

Each generator yields `(x, y)` pairs:

- `x`: Sudoku puzzle, NumPy array of shape `(9, 9)`
- `y`: Complete solution, NumPy array of shape `(9, 9)`
- Data type: `np.uint8`
- Empty cells: `0`
- Filled cells: integers from `1` to `9`

### Example

```python
from MS1.data_loader import load_data

train, test = load_data(
    n_train=100,
    n_test=20,
)

for x, y in train:
    print("Puzzle:", x.shape)
    print("Solution:", y.shape)
    break
```

### Custom transformations

```python
import numpy as np

def transform(x, y):
    return x.astype(np.float32), y

train, test = load_data(
    transformation=transform,
    n_train=100,
)
```

### Dataset paths

On Hydra, the loader automatically uses:

`/home/space/datasets/sudoku_extreme/`

For local development, the default location is:

`data/sudoku_extreme_full/`

Both `train.csv` and `test.csv` must be available in the selected directory.

Alternatively, specify a custom directory using `data_dir`.

## 4. Dataset Validation

**Implementation:** `MS1/validate_dataset.py`

The full dataset was validated on the Hydra cluster.

### Validation results

| Check | Result |
|---|---|
| Training samples | 3,831,994 |
| Test samples | 422,786 |
| Invalid records detected | 0 |
| Duplicate test rows | 0 |
| Exact train/test puzzle overlaps | 0 |
| Overall validation | PASS |

Validation includes checking puzzle lengths, allowed characters, clue-solution consistency and exact puzzle overlaps.

Exact overlaps are checked using BLAKE2b hashes of normalized puzzle strings.

**Limitations:** Equivalent but non-identical puzzles are not checked here; `MS1/explore.py` screens for them with a symmetry-invariant clue-pattern signature.

### Running validation on Hydra

First, request a CPU node:

```bash
srun --partition=cpu-2h --cpus-per-task=1 --pty bash
```

Then execute:

```bash
python3 ~/MS1/validate_dataset.py
```

Results are saved to:

`~/MS1/dataset_integrity.json`

## 5. Testing

Automated unit tests are available in `MS1/tests/`:

- `test_data_loader.py`: sample counts, board shapes, clue consistency, sample limits, transformations, generator behavior.
- `test_metrics.py`, `test_baselines.py`, `test_model.py`: metrics, baselines, augmentation, model shapes/gradients/overfitting, batching.
- `test_explore.py`: invariance of the clue-pattern signature under Sudoku symmetries.

Run them locally from the repository root:

```bash
python -m unittest discover -s MS1/tests -v
```

The full-dataset integrity check is performed separately on Hydra.

## 6. Hydra Integration

The dataset is stored in the shared directory:

`/home/space/datasets/sudoku_extreme/`

The files are readable by other cluster users.

Project code is stored separately in the user's home directory, for example:

`/home/pmlXX/MS1/`

The loader uses the shared dataset directly without copying its contents into each user's home directory.

CPU-intensive validation and training experiments must run through SLURM compute allocations rather than on the login node.

## 7. Known Limitations and Future Improvements

- Streaming CSV is memory-efficient but less suitable for high-speed random access.
- Each generator iteration reads the relevant CSV sequentially.
- Repeated experiments may benefit from preprocessing into memory-mapped arrays or another indexed format.
- The current implementation preserves the official train/test split rather than constructing a new split.

## 8. Dependencies

The data loader requires Python 3 and NumPy.

Additional project dependencies are listed in the repository's root `requirements.txt`.

The data loader and initial smoke tests have been executed successfully on Windows and on a Hydra CPU compute node.

## 9. Model, Baselines and Training

### Representation

Model code flattens each `(9, 9)` grid to 81 `int64` digits (`MS1/model.py::to_flat`). The clue mask is `x > 0`. `augment` (digit, band/stack and row/column permutations, transpose) is optional and applied to the training stream only through `augmented(...)`, never through the `transformation` of `load_data`, because `load_data` applies its transformation to both the train and the test generator.

### Model (`MS1/model.py`)

- Digit embedding plus learned row, column and box embeddings and a learnable initial state (`nn.Parameter`).
- One shared transformer block applied `n_steps` times (fixed loop, no data-dependent control flow), input re-injected each step.
- By default each cell attends to itself and its row, column and box peers; `--global-attention` restores unrestricted attention.
- Output: logits `(B, 81, 9)`; `return_all=True` gives all steps `(B, n_steps, 81, 9)`. `decode` returns digits 1-9 and keeps the given clues.
- Loss `sudoku_loss`: cross-entropy on blank cells, averaged over steps.

### Baselines (`MS1/baselines.py`)

| Baseline | Idea |
|---|---|
| Random | Uniform digit in every blank cell |
| Majority | Most frequent solution digit in every blank cell |
| Naive candidates | Naked and hidden singles only, no search |
| Backtracking | Exact solver (propagation + minimum-remaining-values branching) |

### Metrics (`MS1/metrics.py`)

Cell accuracy on blank cells, exact-match puzzle accuracy (primary), valid-grid rate (no row/column/box duplicates), and breakdowns by clue count, difficulty rating, source and shared clue pattern. Precision/recall do not fit the task: it is a 9-class, structured prediction with one correct answer per puzzle.

### Training (`MS1/train.py`)

`python -m MS1.train --n-train 100000 --n-test 2000 --epochs 3 [--augment]` logs train loss, a probe of seen training puzzles, test puzzles and, by default, 2000 subsequent train-file rows (`--n-fresh 0` disables this last evaluation). It writes `log.csv`, `results.json` and `model.pt` into `--out-dir`. Checkpoints are saved during training; the final results file is written on completion.

`python -m MS1.evaluate --run-dir MS1/results/run_more_data` evaluates a saved model on test, seen-train and unseen-train rows and writes `eval_fresh.json`. `--test-offset 2000` evaluates test rows 2,001-4,000 instead and writes `eval_test_offset_2000.json`; this second slice was used once for the final configuration.

### Exploration and figures

`python -m MS1.explore` computes the dataset statistics and the leakage screen and builds the dataset figures (full run needs ~32 GB RAM; `--nrows N` for a quick run). `python -m MS1.make_figures` builds the results table and result figures from `MS1/results/`.

### Running on Hydra

`MS1/run_train.sbatch` runs inside the Apptainer container `~/pml.sif` (built from `pml.def`). The `pml` account should use the `gpu-teaching-*` partitions; the script defaults to `gpu-teaching-2h`. Submit from `~`:

```bash
sbatch --time=00:30:00 MS1/run_train.sbatch smoke
sbatch MS1/run_train.sbatch baselines
sbatch MS1/run_train.sbatch noaug
sbatch MS1/run_train.sbatch aug
sbatch --time=01:30:00 MS1/run_train.sbatch long
sbatch --time=01:30:00 MS1/run_train.sbatch aug300
sbatch --time=01:30:00 MS1/run_train.sbatch long300
```

`long` trains the larger model on 100k puzzles for 6 epochs. `aug300` trains it on 300k for 3 epochs with augmentation; `long300` uses 300k for 6 epochs without augmentation. They write to separate result directories. Use `squeue -j JOB_ID` for active jobs and `sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed` for completed jobs. Output logs are `~/ms1_frm_JOB_ID.log`.

### Results

Final configuration (width 192, 12 steps, 300k puzzles, 6 epochs): 76.9% cell accuracy and 42.8% exact match on test rows 1-2,000; 76.2% / 38.9% (mean of two seeds) on test rows 2,001-4,000. All runs are listed in [figures/results_table.csv](figures/results_table.csv).
