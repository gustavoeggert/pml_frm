
import csv
import hashlib
import json
from pathlib import Path

DATA_DIR = Path("/home/space/datasets/sudoku_extreme")


def read_puzzles(filename):
    path = DATA_DIR / filename

    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not {"question", "answer"}.issubset(
            reader.fieldnames or []
        ):
            raise ValueError(f"Missing columns in {filename}")

        for line_number, row in enumerate(reader, start=2):
            question = row["question"].strip().replace("0", ".")
            answer = row["answer"].strip()

            if (
                len(question) != 81
                or len(answer) != 81
                or any(c not in ".123456789" for c in question)
                or any(c not in "123456789" for c in answer)
            ):
                raise ValueError(
                    f"Invalid Sudoku at {filename}:{line_number}"
                )

            if any(
                q != "." and q != a
                for q, a in zip(question, answer)
            ):
                raise ValueError(
                    f"Inconsistent clues at {filename}:{line_number}"
                )

            yield question


def puzzle_hash(question):
    return hashlib.blake2b(
        question.encode("ascii"), digest_size=16
    ).digest()


def main():
    print("Checking official test dataset...", flush=True)

    test_hashes = set()
    test_count = 0
    test_duplicates = 0

    for question in read_puzzles("test.csv"):
        test_count += 1
        key = puzzle_hash(question)

        if key in test_hashes:
            test_duplicates += 1

        test_hashes.add(key)

    print(f"Test samples: {test_count:,}", flush=True)

    print("Checking training dataset...", flush=True)

    train_count = 0
    overlapping_hashes = set()

    for question in read_puzzles("train.csv"):
        train_count += 1
        key = puzzle_hash(question)

        if key in test_hashes:
            overlapping_hashes.add(key)

        if train_count % 500_000 == 0:
            print(
                f"Processed {train_count:,} training samples",
                flush=True,
            )

    report = {
        "train_samples": train_count,
        "test_samples": test_count,
        "duplicate_test_rows": test_duplicates,
        "unique_exact_train_test_overlaps": len(overlapping_hashes),
        "validation": "PASS"
        if train_count > 0
        and test_count > 0
        and not overlapping_hashes
        else "FAIL",
    }

    output = Path.home() / "MS1" / "dataset_integrity.json"
    output.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))

    if report["validation"] != "PASS":
        raise SystemExit("Dataset integrity check failed")


if __name__ == "__main__":
    main()