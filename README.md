# ClusterHW

Personal_HW: gym member segmentation and monthly credit targeting.

This repository contains the supplied assignment, lecture notes, and gym_members.csv. The analysis is being built in stages. Step 1 creates the training and test files.

## Step 1: create a reproducible 70/30 split

Randomly reserve 30% of the 330 members for testing and use the remaining 70% for training.

- Test members: 330 x 0.30 = 99.
- Training members: 330 - 99 = 231.
- Random seed: 42.
- Shuffle whole rows before splitting, so every member's values stay together.

The central Python statement is:

```python
train_df, test_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    shuffle=True,
)
```

| Setting | Meaning |
| --- | --- |
| df | The complete original member table |
| test_size=0.30 | Put 30% of the rows into the test set; the remainder goes into training |
| random_state=42 | Fix the random seed so the allocation can be reproduced |
| shuffle=True | Randomize row selection instead of taking the first 70% |

The value 42 is an arbitrary seed, not a model parameter chosen for better results. Keep the same seed, original CSV contents and row order, and compatible library versions for a matching rerun. The script prints the installed package versions.

## Run on Windows

1. Get the updated repository. On GitHub, choose **Code > Download ZIP** and extract it. If you already cloned it and have no local changes, you can use **git pull** instead.
2. Open the extracted project folder containing split_data.py, requirements.txt, and gym_members.csv.
3. Open a terminal in that folder. In Windows File Explorer, you can type powershell into the address bar and press Enter.
4. Install the two required packages:

```powershell
py -m pip install -r requirements.txt
```

5. Run the split script:

```powershell
py split_data.py
```

If your computer uses python instead of py, replace py in both commands with python.

## Files created on your computer

| Output file | Data rows | Columns | Purpose |
| --- | ---: | ---: | --- |
| data/train_members.csv | 231 | 7 | Fit the scaler and cluster centers in the next stage |
| data/test_members.csv | 99 | 7 | Evaluate the fitted models using unseen members |

Each CSV also has a header row. The data folder is created automatically beside the script. All seven original columns are retained, including member_id for checking membership. member_id will be excluded from model features in the later modeling stage. index=False prevents an extra row-number column from being written.

Expected summary:

```text
Source members: 330
Training members: 231 (70%)
Test members: 99 (30%)
Member overlap: 0
Random seed: 42
```

The script also prints both saved file paths and the pandas, scikit-learn, and numpy versions. It checks that the input has the expected shape and columns, IDs are unique, the two sets do not overlap, and their union contains every source member.

Rerunning the script overwrites the two generated CSVs. With the same input, settings, and software behavior, the result is the same split.

## How to use the split in later steps

Keep this one split throughout the analysis. Fit StandardScaler and KMeans on train_members.csv. Transform test_members.csv using the training scaler and evaluate it using the existing training centers. At this stage the exported values remain in their original units.

After the local run, upload the generated data folder to this repository so the saved split is available alongside the code.

The final submission is one PDF containing a title page and two content slides. The assignment permits AI assistance with code and chart-generating code; the student writes and assembles the slides.

Reference: [scikit-learn train_test_split documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).
