
import csv
from pathlib import Path

import numpy as np


def load_data(
    dataset="sudoku_extreme",
    transformation=None,
    n_train=None,
    n_test=None,
    data_dir=None,
):
    """
    Load Sudoku-Extreme using the official train/test split.

    Returns:
        train_generator, test_generator

    Each generator yields (x, y), where both are 9x9 arrays.
    """

    # Locate dataset on Hydra or Windows
    if data_dir is not None:
        root = Path(data_dir)
    else:
        hydra_path = Path("/home/space/datasets") / dataset
        local_path = (
            Path(__file__).resolve().parents[1]
            / "data" / "sudoku_extreme_full"
        )

        root = hydra_path if hydra_path.is_dir() else local_path

    train_path = root / "train.csv"
    test_path = root / "test.csv"

    for path in (train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset file: {path}")

    for name, limit in (("n_train", n_train), ("n_test", n_test)):
        if limit is not None and (
            type(limit) is not int or limit < 0
        ):
            raise ValueError(f"{name} must be a non-negative integer")

    def stream_csv(path, limit):
        if limit == 0:
            return

        count = 0

        with path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as f:
            reader = csv.DictReader(f)

            if not {"question", "answer"}.issubset(
                reader.fieldnames or []
            ):
                raise ValueError(
                    f"Missing question/answer columns in {path}"
                )

            for row in reader:
                question = row["question"].strip()
                answer = row["answer"].strip()

                if (
                    len(question) != 81
                    or len(answer) != 81
                    or any(c not in ".0123456789" for c in question)
                    or any(c not in "123456789" for c in answer)
                ):
                    raise ValueError("Invalid Sudoku row")

                # Convert strings to 9x9 arrays
                x = np.array(
                    [0 if c in ".0" else int(c)
                     for c in question],
                    dtype=np.uint8,
                ).reshape(9, 9)

                y = np.array(
                    [int(c) for c in answer],
                    dtype=np.uint8,
                ).reshape(9, 9)

                if not np.all((x == 0) | (x == y)):
                    raise ValueError(
                        "Sudoku clues do not match solution"
                    )

                if transformation is not None:
                    x, y = transformation(x, y)

                yield x, y
                count += 1

                if limit is not None and count >= limit:
                    break

    return (
        stream_csv(train_path, n_train),
        stream_csv(test_path, n_test),
    )