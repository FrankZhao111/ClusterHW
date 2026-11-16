"""Standardize the existing gym member split using training statistics only."""

import csv
import math
from pathlib import Path
from statistics import fmean, pstdev

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "data"
FEATURES = [
    "age",
    "visits_per_month",
    "avg_visit_minutes",
    "classes_per_month",
    "months_member",
    "monthly_extras_spend",
]
COLUMNS = ["member_id", *FEATURES]


def read_members(path, expected_count):
    """Read one split, checking its columns, values, row count, and unique IDs."""
    with path.open("r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"{path.name} must contain the seven original columns.")
        rows = list(reader)

    if len(rows) != expected_count:
        raise ValueError(f"{path.name} must contain {expected_count} members.")

    members = []
    for row in rows:
        if None in row:
            raise ValueError(f"{path.name} contains a row with extra values.")
        member_id = (row["member_id"] or "").strip()
        if not member_id:
            raise ValueError(f"{path.name} contains a missing member_id.")
        member = {"member_id": member_id}
        for feature in FEATURES:
            try:
                value = float(row[feature])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Member {member_id}: {feature} must be a number."
                ) from error
            if not math.isfinite(value):
                raise ValueError(f"Member {member_id}: {feature} must be finite.")
            member[feature] = value
        members.append(member)

    if len({row["member_id"] for row in members}) != len(members):
        raise ValueError(f"{path.name} contains duplicate member IDs.")
    return members


def as_member_map(rows):
    """Compare copies by member ID and values, regardless of file row order."""
    return {
        row["member_id"]: tuple(row[feature] for feature in FEATURES)
        for row in rows
    }


def load_split():
    """Accept a complete raw split in data/ or beside this script."""
    candidates = []
    for folder in (PROJECT_DIR / "data", PROJECT_DIR):
        train_file = folder / "train_members.csv"
        test_file = folder / "test_members.csv"
        if train_file.is_file() and test_file.is_file():
            candidates.append(
                (
                    train_file,
                    test_file,
                    read_members(train_file, 231),
                    read_members(test_file, 99),
                )
            )

    if not candidates:
        raise FileNotFoundError(
            "Place train_members.csv and test_members.csv together in data/ "
            "or in the same folder as scale_data.py."
        )

    train_file, test_file, train_rows, test_rows = candidates[0]
    for _, _, other_train, other_test in candidates[1:]:
        if (
            as_member_map(train_rows) != as_member_map(other_train)
            or as_member_map(test_rows) != as_member_map(other_test)
        ):
            raise ValueError(
                "The root folder and data/ contain different raw splits. "
                "Keep one consistent training/test pair before scaling."
            )

    train_ids = {row["member_id"] for row in train_rows}
    test_ids = {row["member_id"] for row in test_rows}
    if train_ids & test_ids:
        raise ValueError("Training and test files contain overlapping member IDs.")
    return train_file, test_file, train_rows, test_rows


def transform(rows, parameters):
    """Apply the stored TRAINING mean and standard deviation to any member."""
    scaled_rows = []
    for row in rows:
        scaled = {"member_id": row["member_id"]}
        for feature in FEATURES:
            mean, std = parameters[feature]
            scaled[feature] = (row[feature] - mean) / std
        scaled_rows.append(scaled)
    return scaled_rows


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    train_file, test_file, train_rows, test_rows = load_split()

    # Fit scaling on the 231 training members. Test members do not enter this loop.
    parameters = {}
    for feature in FEATURES:
        values = [row[feature] for row in train_rows]
        mean = fmean(values)
        # Population standard deviation divides variance by n, matching StandardScaler.
        std = pstdev(values)
        if std == 0:
            raise ValueError(f"{feature} has no training variation; review this feature.")
        parameters[feature] = (mean, std)

    # Apply exactly the SAME parameters to the training and test sets.
    train_scaled = transform(train_rows, parameters)
    test_scaled = transform(test_rows, parameters)

    # Numerical checks allow tiny floating-point rounding differences.
    for feature in FEATURES:
        values = [row[feature] for row in train_scaled]
        if not math.isclose(fmean(values), 0.0, abs_tol=1e-10):
            raise RuntimeError(f"Scaled training mean check failed for {feature}.")
        if not math.isclose(pstdev(values), 1.0, rel_tol=1e-10, abs_tol=1e-10):
            raise RuntimeError(f"Scaled training std check failed for {feature}.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scaled_train_file = OUTPUT_DIR / "train_scaled.csv"
    scaled_test_file = OUTPUT_DIR / "test_scaled.csv"
    parameters_file = OUTPUT_DIR / "scaling_parameters.csv"
    write_csv(scaled_train_file, COLUMNS, train_scaled)
    write_csv(scaled_test_file, COLUMNS, test_scaled)

    parameter_rows = [
        {
            "feature": feature,
            "training_mean": parameters[feature][0],
            "training_std": parameters[feature][1],
            "training_count": len(train_rows),
            "ddof": 0,
        }
        for feature in FEATURES
    ]
    write_csv(
        parameters_file,
        ["feature", "training_mean", "training_std", "training_count", "ddof"],
        parameter_rows,
    )

    print(f"Training input: {train_file}")
    print(f"Test input: {test_file}")
    print(f"Training members scaled: {len(train_scaled)}")
    print(f"Test members scaled: {len(test_scaled)}")
    print(f"Features standardized: {len(FEATURES)}")
    print("Training scaling checks: PASS (mean ~0, population std ~1)")
    print("member_id retained for tracking; use only the six feature columns in K-means.")
    print()
    print(f"{'Feature':<24} {'Training mean':>16} {'Training std':>16}")
    for feature in FEATURES:
        mean, std = parameters[feature]
        print(f"{feature:<24} {mean:>16.6f} {std:>16.6f}")
    print()
    print(f"Saved: {scaled_train_file}")
    print(f"Saved: {scaled_test_file}")
    print(f"Saved: {parameters_file}")


if __name__ == "__main__":
    main()
