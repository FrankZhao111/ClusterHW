# ClusterHW

Personal_HW: gym member segmentation and monthly credit targeting.

This repository contains the supplied assignment, lecture notes, and gym_members.csv. The analysis is being built in stages. Step 1 creates the training and test files; Step 2 standardizes their features using training statistics.

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

## Step 2: standardize the existing split

The assignment's Task 1 requires fitting everything, including scaling, on the 70% training set. Scaling puts the six features into comparable standard-deviation units before K-means measures distances.

Use the existing split. Do not rerun the split to get more favorable results.

scale_data.py uses only Python's built-in csv, math, pathlib, and statistics modules; no pip installation is needed for this step.

### Formula and feature choices

For each feature:

```text
scaled value = (original value - training mean) / training standard deviation
```

The script calculates a mean and population standard deviation from the 231 training members for each of these six features:

- age
- visits_per_month
- avg_visit_minutes
- classes_per_month
- months_member
- monthly_extras_spend

It applies those same six pairs of parameters to both the training and test files. member_id is copied unchanged for tracking and must be excluded from the K-means feature matrix.

statistics.pstdev uses the population convention: variance is the sum of squared deviations divided by 231, rather than 230. This matches StandardScaler's default ddof=0 convention for these nonconstant features. If checking in Excel, use AVERAGE and STDEV.P on the training rows.

For example, the current training set has mean visits 10.6103896104 and standard deviation 5.0670547235. A member with four visits per month becomes approximately (4 - 10.6103896104) / 5.0670547235 = -1.3046, or 1.30 standard deviations below the training average.

### Run locally on Windows

In the existing project folder, download and run the new script:

```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/FrankZhao111/ClusterHW/main/scale_data.py" -OutFile ".\scale_data.py"
py .\scale_data.py
```

If your computer uses python instead of py, use python .\scale_data.py for the second command.

The script accepts train_members.csv and test_members.csv together in the data folder or together beside scale_data.py. The uploaded repository currently stores the raw split in its main folder; split_data.py creates it locally inside data. Both arrangements work.

If both locations contain a pair, the script verifies that their members and values match and uses the pair in data. If the pairs differ, it stops with a message so two different splits cannot be mixed.

### Generated files

| Output file | Data rows | Contents |
| --- | ---: | --- |
| data/train_scaled.csv | 231 | Original member IDs plus the six standardized training features |
| data/test_scaled.csv | 99 | Original member IDs plus test features transformed with training parameters |
| data/scaling_parameters.csv | 6 | One row per feature: training_mean, training_std, training_count=231, ddof=0 |

The original raw CSV files are preserved. These three generated files are overwritten when the scaling script is rerun. CSVs retain full floating-point precision; the terminal table is rounded to six decimal places for readability.

Expected checks:

```text
Training members scaled: 231
Test members scaled: 99
Features standardized: 6
Training scaling checks: PASS (mean ~0, population std ~1)
```

The script also prints the two actual input paths, all six training means and standard deviations, and the three saved output paths. Every scaled training feature should have a mean approximately zero and population standard deviation approximately one. The test means and standard deviations need not be zero and one.

Expected training parameters for the current uploaded split, rounded for display:

| Feature | Training mean | Training standard deviation |
| --- | ---: | ---: |
| age | 40.246753 | 13.328506 |
| visits_per_month | 10.610390 | 5.067055 |
| avg_visit_minutes | 53.268398 | 11.639863 |
| classes_per_month | 3.203463 | 2.742512 |
| months_member | 28.004329 | 18.442258 |
| monthly_extras_spend | 26.484848 | 17.921389 |

After running, upload the three generated CSV files so the numerical transformations can be checked against the code.

## How to use the scaled data in later steps

Fit KMeans on the six feature columns in train_scaled.csv and evaluate its existing centers on the same six columns in test_scaled.csv. Exclude member_id and do not standardize already standardized files again.

Use scaling_parameters.csv to transform the supplied new member using the same training parameters. Convert fitted centers back into original units with original center = scaled center x training_std + training_mean for interpretation.

requirements.txt lists pandas and scikit-learn for the later modeling stage. That stage needs an environment where their NumPy dependency can load successfully; the standard-library splitter does not repair the separate Windows DLL policy issue.

The final submission is one PDF containing a title page and two content slides. The assignment permits AI assistance with code and chart-generating code; the student writes and assembles the slides.

References: [Python csv documentation](https://docs.python.org/3.14/library/csv.html), [Python random documentation](https://docs.python.org/3.14/library/random.html), [Python statistics documentation](https://docs.python.org/3.14/library/statistics.html), and [StandardScaler documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).
