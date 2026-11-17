# ClusterHW

Personal_HW: gym member segmentation and monthly credit targeting.

This repository contains the supplied assignment, lecture notes, and gym_members.csv. The analysis is being built in stages. Step 1 creates the training and test files; Step 2 standardizes their features using training statistics.

[Run the candidate comparison in Google Colab](https://colab.research.google.com/github/FrankZhao111/ClusterHW/blob/main/kmeans_comparison.ipynb).

[Run the four-segment business analysis and Excel export in Colab](https://colab.research.google.com/github/FrankZhao111/ClusterHW/blob/main/business_decision.ipynb).

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

## Windows DLL block: run the modeling environment check

The Windows error `An Application Control policy has blocked this file` occurs while importing a compiled NumPy component. The traceback alone does not establish whether Smart App Control or an organization-managed App Control policy caused it. A virtual environment isolates packages but does not override a Windows application-control decision.

### Recommended assignment workaround: Google Colab

[Open environment_check.ipynb in Google Colab](https://colab.research.google.com/github/FrankZhao111/ClusterHW/blob/main/environment_check.ipynb)

1. Open the link and sign in to Google if prompted.
2. Keep the standard hosted CPU runtime; a GPU is unnecessary.
3. Choose **Runtime > Run all**. Review and run the notebook code when Colab asks.
4. Look for `Environment check passed: imports, data checks, and K-means fitting work.`

The notebook installs missing modeling packages, prints their versions, explicitly imports numpy.random, loads the existing raw/scaled CSVs and scaling parameters from a pinned GitHub commit, checks the transformations, and fits k=1. Expected training spread is 1386.0000 and expected test spread is 666.1366.

This cloud-runtime workaround leaves local Windows security settings and the existing data preparation intact. The notebook has been prepared for the user to execute; no successful Colab run is claimed before its output is available.

### Optional local troubleshooting attempt

A fresh Python 3.13 environment uses a different Python/package build combination from the failing global Python 3.14 installation. This is a troubleshooting attempt, not a confirmed fix for an application-control block.

If the Python install manager is available, run these commands from the project folder:

```powershell
py install 3.13
py -3.13 -m venv .venv-kmeans
.\.venv-kmeans\Scripts\python.exe -m pip install --upgrade pip
.\.venv-kmeans\Scripts\python.exe -m pip install --only-binary=:all: numpy scipy pandas scikit-learn matplotlib
.\.venv-kmeans\Scripts\python.exe -c "import numpy.random; from sklearn.cluster import KMeans; print('Ready for K-means')"
```

If `py install` is unavailable, use the [official Python downloads](https://www.python.org/downloads/windows/) or install the [Python install manager](https://www.python.org/downloads/), then create the environment with the selected Python 3.13 interpreter.

Using the environment's python.exe directly avoids needing a PowerShell activation script. The binary-only install requests existing wheels rather than a local source build. If the same Application Control error persists, the new environment did not resolve the policy issue; use the hosted notebook for the assignment.

### Identify the local policy for a permanent repair

Microsoft documents block event 3077 in the CodeIntegrity Operational log. In Event Viewer, open **Applications and Services Logs > Microsoft > Windows > CodeIntegrity > Operational**, then inspect recent events for the NumPy file. The event identifies the blocked file and policy. An administrator who manages a business policy can review an appropriate allow rule or approved installation. Smart App Control has no individual-app exception setting according to Microsoft's FAQ.

Sources: [Microsoft App Control troubleshooting](https://learn.microsoft.com/windows/security/application-security/application-control/app-control-for-business/operations/appcontrol-debugging-and-troubleshooting), [Smart App Control FAQ](https://support.microsoft.com/en-us/windows/security/threat-malware-protection/smart-app-control-frequently-asked-questions), [Colab FAQ](https://research.google.com/colaboratory/faq.html), [Python on Windows](https://docs.python.org/3.14/using/windows.html), and [Python virtual environments](https://docs.python.org/3.14/library/venv.html).

## Step 3: compare K-means candidates in Colab

[Open kmeans_comparison.ipynb in Google Colab](https://colab.research.google.com/github/FrankZhao111/ClusterHW/blob/main/kmeans_comparison.ipynb)

This follows the successful environment check and covers the candidate fitting and held-out fit evaluation in Tasks 2 and 3. Open the new notebook and select **Runtime > Run all**. It is self-contained and loads the same frozen input revision as the environment check.

Every candidate uses the six existing standardized feature columns; member_id is excluded. For k=1 through k=8, KMeans fits on the 231 training members and assigns the 99 test members to those existing centers. It does not refit on test members, resplit the data, or fit scaling again.

All candidates use init='k-means++', n_init=20, random_state=42, max_iter=300, tol=1e-4, and algorithm='lloyd'. n_init=20 means twenty initializations, with the lowest training spread retained. Only k varies between candidates.

The code sums each member's squared distance to its assigned center and checks that the training total matches inertia_ and the test total matches -score(X_test). Expected k=1 baseline totals are training 1386.0000 and test 666.1366.

### Outputs

| File | Purpose |
| --- | --- |
| fit_metrics.csv | Eight rows: spreads, spreads per member, drops from previous k, reductions versus k=1, and iteration counts |
| elbow_train_test.png / .svg | Required total-spread curves in adjacent training and test panels |
| elbow_per_member.png / .svg | Optional comparison after dividing training spread by 231 and test spread by 99 |
| candidate_assignments.csv | 2,640 rows: each of the 330 members under each of eight candidates, including cluster and squared distance |
| candidate_centers_scaled.csv | The 36 fitted centers across all candidates, in scaled units |
| candidate_centers_original.csv | Those centers converted back to original units for interpretation |
| scaling_parameters.csv | A copy of the six existing training scaling parameter rows |
| model_checkpoint.json | Portable record of fitted centers, preprocessing, input-data revision, configuration, and software versions |

Files are generated in outputs/kmeans_comparison inside the notebook runtime. A final cell downloads kmeans_outputs.zip containing the results. Extract and upload the CSVs, checkpoint, and charts to GitHub; save the executed notebook as well. The source notebook is prepared for the student to run; its model outputs are not claimed before execution.

### Read the comparison

- Spread is the sum of squared distances from members to their assigned centers, using six standardized features.
- A drop from previous k equals the preceding spread minus the current spread. The first row's drop is blank because there is no previous candidate.
- Reduction versus k=1 equals (baseline spread - candidate spread) / baseline spread x 100. This measures spread reduction, not accuracy.
- The two required panels show totals for different member counts. Use the optional per-member chart to compare magnitudes fairly.
- Look for diminishing improvements and a similar elbow in the held-out curve. The assignment requests three to five useful segments, but the model comparison must examine k=1 through k=8.
- Do not automatically copy the lecture example's k=3. If the curves do not support the same elbow, report that and reconsider k as the assignment instructs.

After the run, review fit_metrics.csv and elbow_train_test.png before choosing k. The saved centers and assignments support later segment naming and financial calculations without fitting centers on test or all members.

References: [KMeans documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) and [Matplotlib savefig documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html).

## Step 4: describe segments, decide the credit policy, and quantify value

[Open business_decision.ipynb in Google Colab](https://colab.research.google.com/github/FrankZhao111/ClusterHW/blob/main/business_decision.ipynb)

Select **Runtime > Run all**. The notebook loads a pinned version of business_analysis.py, reuses the saved k=4 centers, and creates business_outputs.zip. No new clustering fit is performed. This covers the interpretation, new-member function, and member-level financial calculations.

Python is the assignment's requested computational method. The Excel and CSV files are supporting analysis outputs; the final submission remains one PDF containing a title page and two student-written content slides.

### Policy and assumptions

Each credited member spending below 20 dollars on extras contributes 25 x 60% - 10 = +5 dollars per month. Every credited member spending 20 dollars or more contributes -10 dollars. An uncredited member contributes zero incremental value from the credit policy.

Choose eligibility using training members only: give a segment the credit if its sum of training-member contributions is positive. A break-even segment receives no credit. Then apply the fixed segment policy to every historical member, including the test members. This separates policy choice from the required full-330-member simulation.

The expected segment eligibility flags are cluster 0: yes; clusters 1, 2, and 3: no. Cluster 1 has training contribution exactly zero, although its average extras spend is below 20. Member counts and individual spending, rather than a centroid's average spend, determine profitability.

Expected monthly totals for the supplied data:

| Plan | Recipients | Added extras revenue | Added margin | Credit cost | Net contribution |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline: everyone | 330 | 3500 | 2100 | 3300 | -1200 |
| V1: cluster 0 | 70 | 1475 | 885 | 700 | 185 |

The improvement is 185 - (-1200) = 1385 dollars per month. These are incremental contributions under the provided assumptions, not total gym profit or measured campaign effects.

The supplied new member is an example outside the 330-member population. It maps to cluster 0 and receives the credit; it is not added to plan totals.

### Excel workbook

business_analysis.xlsx contains a Guide and separate sheets for assumptions, scaling, saved centers, member-level calculations, training-based segment policy, training-centroid summaries, full-population profiles, plan comparison, contribution by split, model-fit comparison, and the new member.

Financial calculations, segment decisions, scaling, inverse scaling, fit reductions, and the new-member prediction have visible Excel formulas. Python supplies cached results for spreadsheet previews; Excel recalculates formulas on opening. Blue assumption and new-member input cells support what-if inspection. Historical data and fitted centers are fixed source records; changing them requires rerunning the Python workflow.

The writer checks key formula caches and formula presence using openpyxl. Those checks do not substitute for Excel's formula calculation engine.

### CSV and JSON outputs

| File | Meaning |
| --- | --- |
| segment_summary.csv | Counts for all 330 members plus training-centroid characteristics |
| segment_population_profile.csv | Means across all members assigned to each segment |
| segment_policy.csv | Training-based eligibility, individual-spend counts, and segment economics |
| member_audit.csv | One row per historical member; both policies' costs and contributions |
| plan_comparison.csv | Baseline, V1, and V1-minus-baseline monthly values |
| value_by_split.csv | Training, test, and all-member contribution totals |
| model_fit_summary.csv | Baseline versus k=4 spread and reduction on training and test members |
| fit_metrics.csv | The original eight candidate results |
| scaling_parameters.csv / selected_centers.csv | The preprocessing and frozen centers used by inference |
| new_member_summary.csv | Segment and credit decision for the supplied example |
| new_member_scaling.csv / new_member_distances.csv | Every scaling input and distance used in that example |
| selected_model.json | Selected model, feature order, scaling, policy, assumptions, revisions, and source hash |
| validation_checks.json | Workbook cache/structure checks completed during the notebook run |

The ZIP also contains business_analysis.py. Upload the extracted outputs to GitHub in a business_outputs folder and preserve the executed notebook.

### One-member function

```python
recommend_offer = analysis.make_recommender(results['selected_model'])
recommend_offer({
    'age': 30, 'visits_per_month': 4, 'avg_visit_minutes': 42,
    'classes_per_month': 1, 'months_member': 5, 'monthly_extras_spend': 10,
})
```

The returned function has one input argument and reports cluster, segment, credit eligibility, credit amount, and distance to the selected center. Its output is checked against all 330 historical assignments and policy flags during the analysis run.

Source modules and notebook contents are prepared and saved in GitHub. The notebook's actual execution and workbook generation occur when the student runs it in Colab.
