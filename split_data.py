"""Create the assignment's 70/30 split using only Python's standard library."""

import csv
import math
import random
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_DIR / "gym_members.csv"
OUTPUT_DIR = PROJECT_DIR / "data"

TEST_SIZE = 0.30
RANDOM_SEED = 42
EXPECTED_COLUMNS = [
    "member_id",
    "age",
    "visits_per_month",
    "avg_visit_minutes",
    "classes_per_month",
    "months_member",
    "monthly_extras_spend",
]


def write_csv(path, header, rows):
    """Save the header and original values without adding a row-index column."""
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    # Read whole rows as text so all seven original values stay together.
    # utf-8-sig also handles a CSV saved with an Excel-style UTF-8 byte-order mark.
    with INPUT_FILE.open("r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.reader(input_file)
        header = next(reader, None)
        rows = [row for row in reader if row]

    if header != EXPECTED_COLUMNS or len(rows) != 330:
        raise ValueError("Expected the assignment CSV: 330 rows and seven original columns.")
    if any(len(row) != len(EXPECTED_COLUMNS) for row in rows):
        raise ValueError("Every member row must contain exactly seven values.")

    source_ids = [row[0].strip() for row in rows]
    if any(not member_id for member_id in source_ids):
        raise ValueError("Every row must have a non-missing member_id.")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Each member_id must appear exactly once in the source CSV.")

    # One seeded shuffle makes a random permutation of the complete member rows.
    shuffled_rows = rows.copy()
    random.Random(RANDOM_SEED).shuffle(shuffled_rows)

    # 30% of all 330 members = 99 test members; the other 231 are for training.
    test_count = math.ceil(len(rows) * TEST_SIZE)
    test_rows = shuffled_rows[:test_count]
    train_rows = shuffled_rows[test_count:]

    train_ids = {row[0].strip() for row in train_rows}
    test_ids = {row[0].strip() for row in test_rows}
    if len(train_rows) != 231 or len(test_rows) != 99:
        raise RuntimeError("The expected split is 231 training rows and 99 test rows.")
    if train_ids & test_ids:
        raise RuntimeError("A member appears in both sets.")
    if (train_ids | test_ids) != set(source_ids):
        raise RuntimeError("The split does not contain exactly the source members.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_file = OUTPUT_DIR / "train_members.csv"
    test_file = OUTPUT_DIR / "test_members.csv"
    write_csv(train_file, header, train_rows)
    write_csv(test_file, header, test_rows)

    print(f"Source members: {len(rows)}")
    print(f"Training members: {len(train_rows)} ({len(train_rows) / len(rows):.0%})")
    print(f"Test members: {len(test_rows)} ({len(test_rows) / len(rows):.0%})")
    print(f"Member overlap: {len(train_ids & test_ids)}")
    print(f"Random seed: {RANDOM_SEED}")
    print("Split method: Python standard-library random.Random.shuffle")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Saved: {train_file}")
    print(f"Saved: {test_file}")


if __name__ == "__main__":
    main()
