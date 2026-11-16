# ClusterHW

Personal_HW: gym member segmentation and monthly credit targeting.

This repository contains the supplied assignment, lecture notes, and gym_members.csv. The analysis is being built in stages. Step 1 creates the training and test files.

## Step 1: create a reproducible 70/30 split

Randomly reserve 30% of the 330 members for testing and use the remaining 70% for training.

- Test members: 330 x 0.30 = 99.
- Training members: 330 - 99 = 231.
- Random seed: 42.
- Shuffle whole rows before splitting, so every member's values stay together.

split_data.py uses only Python's standard library. No pip installation is needed for this step.

The central Python statements are:

```python
shuffled_rows = rows.copy()
random.Random(42).shuffle(shuffled_rows)
test_count = math.ceil(len(rows) * 0.30)
test_rows = shuffled_rows[:test_count]
train_rows = shuffled_rows[test_count:]
```

| Statement | Meaning |
| --- | --- |
| rows.copy() | Copy all original member rows |
| random.Random(42).shuffle(...) | Randomly reorder whole rows using a fixed seed |
| math.ceil(len(rows) * 0.30) | Reserve 30% of all rows for testing; here this is exactly 99 |
| shuffled_rows[:test_count] | The first 99 rows after shuffling form the test set |
| shuffled_rows[test_count:] | The remaining 231 rows form the training set |

The value 42 is an arbitrary seed, not a choice made for better model results. Keep the original CSV contents, its row order, the seed, and the Python version fixed for a matching rerun. The script prints the Python version.

This version replaces the earlier pandas/scikit-learn splitter after a Windows Application Control policy blocked a NumPy DLL during import. Python's shuffle and scikit-learn's shuffle do not select identical members from the same numeric seed. Both satisfy the assignment's random 70/30 requirement. Use the CSVs produced by this version for every subsequent stage instead of mixing splits.

## Run on Windows

Get the updated repository through **Code > Download ZIP** and extract it. Open the folder containing split_data.py and gym_members.csv, then open PowerShell there. In File Explorer, you can type powershell into the address bar and press Enter.

Run:

```powershell
py split_data.py
```

If your computer uses python instead of py, use python split_data.py.

### Refresh an older local copy

If you already downloaded the project and are in its folder, replace just the script with the current repository version:

```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/FrankZhao111/ClusterHW/main/split_data.py" -OutFile ".\split_data.py"
py .\split_data.py
```

## Files created on your computer

| Output file | Data rows | Columns | Purpose |
| --- | ---: | ---: | --- |
| data/train_members.csv | 231 | 7 | Fit the scaler and cluster centers in the next stage |
| data/test_members.csv | 99 | 7 | Evaluate the fitted models using unseen members |

Each CSV also has a header row. The data folder is created automatically beside the script. All seven original columns are retained, including member_id for checking membership. member_id will be excluded from model features in the later modeling stage. The CSV writer preserves the original values without adding a row-index column.

Expected summary:

```text
Source members: 330
Training members: 231 (70%)
Test members: 99 (30%)
Member overlap: 0
Random seed: 42
```

The script also prints the split method, Python version, and both saved file paths. It checks that the input has the expected shape and columns, IDs are unique, the two sets do not overlap, and their union contains every source member.

Rerunning the script overwrites the two generated CSVs. With the same input, seed, and Python version, the result is the same split.

## How to use the split in later steps

Keep this one split throughout the analysis. Fit StandardScaler and KMeans on train_members.csv. Transform test_members.csv using the training scaler and evaluate it using the existing training centers. At this stage the exported values remain in their original units.

After the local run, upload the generated data folder to this repository so the saved split is available alongside the code.

requirements.txt lists pandas and scikit-learn for the later modeling stage. That stage needs an environment where their NumPy dependency can load successfully; the standard-library splitter does not repair the separate Windows DLL policy issue.

The final submission is one PDF containing a title page and two content slides. The assignment permits AI assistance with code and chart-generating code; the student writes and assembles the slides.

References: [Python csv documentation](https://docs.python.org/3.14/library/csv.html) and [Python random documentation](https://docs.python.org/3.14/library/random.html).
