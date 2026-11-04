# Milestone 1 — Sudoku-Extreme Data Loading and Validation
**Component:** Data Loading and Dataset Management

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

**Limitations:** Equivalent but non-identical Sudoku puzzles have not been checked. Duplicates within the training set have not been assessed.

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

Automated unit tests are available in:

`MS1/tests/test_data_loader.py`

Run them locally from the repository root:

```bash
python -m unittest discover -s MS1/tests -v
```

The tests cover sample counts, board shapes, clue consistency, sample limits, transformations and basic generator behavior.

The full-dataset integrity check is performed separately on Hydra.

## 6. Hydra Integration

The dataset is stored in the shared directory:

`/home/space/datasets/sudoku_extreme/`

The files are readable by other cluster users.

Project code is stored separately in the user's home directory, for example:

`/home/pmlXX/MS1/`

The loader uses the shared dataset directly without copying its contents into each user's home directory.

CPU-intensive validation and training experiments must run through SLURM compute allocations rather than on the login node.

**Important:** Coding agents are prohibited on Hydra under the PML course rules. AI-assisted coding is permitted locally, but computational jobs must be submitted manually.

## 7. Known Limitations and Future Improvements

- Streaming CSV is memory-efficient but less suitable for high-speed random access.
- Each generator iteration reads the relevant CSV sequentially.
- Repeated experiments may benefit from preprocessing into memory-mapped arrays or another indexed format.
- The current implementation preserves the official train/test split rather than constructing a new split.
- Additional checks for equivalent Sudoku puzzles may be useful for more rigorous evaluation.

## 8. Dependencies

The data loader requires Python 3 and NumPy.

Additional project dependencies are listed in the repository's root `requirements.txt`.

The data loader and initial smoke tests have been executed successfully on Windows and on a Hydra CPU compute node.