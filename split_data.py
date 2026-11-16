"""Create the reproducible 70/30 split for the gym clustering assignment."""

from importlib.metadata import version
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Paths are relative to this script, so running from another folder also works.
PROJECT_DIR = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_DIR / "gym_members.csv"
OUTPUT_DIR = PROJECT_DIR / "data"

# Reserve 30% of the members for evaluation. The remaining 70% are for training.
TEST_SIZE = 0.30
RANDOM_STATE = 42


def main():
    # Load the original, unscaled data, keeping member IDs for traceability.
    df = pd.read_csv(INPUT_FILE)
    expected_columns = [
        "member_id",
        "age",
        "visits_per_month",
        "avg_visit_minutes",
        "classes_per_month",
        "months_member",
        "monthly_extras_spend",
    ]
    if list(df.columns) != expected_columns or len(df) != 330:
        raise ValueError("Expected the assignment CSV: 330 rows and seven original columns.")
    if df["member_id"].isna().any() or not df["member_id"].is_unique:
        raise ValueError("Each row must have a unique, non-missing member_id.")

    # Shuffle whole rows, then allocate 99 members to test and 231 to training.
    # The seed makes the allocation reproducible for the same input row order.
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # Check that every source member appears once across the two files.
    train_ids = set(train_df["member_id"])
    test_ids = set(test_df["member_id"])
    if len(train_df) != 231 or len(test_df) != 99:
        raise RuntimeError("The expected split is 231 training rows and 99 test rows.")
    if train_ids & test_ids:
        raise RuntimeError("A member appears in both sets.")
    if (train_ids | test_ids) != set(df["member_id"]):
        raise RuntimeError("The split does not contain exactly the source members.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_file = OUTPUT_DIR / "train_members.csv"
    test_file = OUTPUT_DIR / "test_members.csv"

    # index=False keeps the original seven columns without an extra row-number column.
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"Source members: {len(df)}")
    print(f"Training members: {len(train_df)} ({len(train_df) / len(df):.0%})")
    print(f"Test members: {len(test_df)} ({len(test_df) / len(df):.0%})")
    print(f"Member overlap: {len(train_ids & test_ids)}")
    print(f"Random seed: {RANDOM_STATE}")
    print(f"Saved: {train_file}")
    print(f"Saved: {test_file}")
    print(
        f"Versions: pandas={pd.__version__}, "
        f"scikit-learn={version('scikit-learn')}, numpy={version('numpy')}"
    )


if __name__ == "__main__":
    main()
