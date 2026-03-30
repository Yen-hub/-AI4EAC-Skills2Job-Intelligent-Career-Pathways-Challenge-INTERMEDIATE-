# Running the Winning `make_oof_stacker.py` Pipeline

## Purpose

This document explains how to run the final winning pipeline cleanly and consistently.

Repository note:

- the datasets are intentionally not committed to this repository
- download them from [Zindi challenge data page](https://zindi.africa/competitions/skills2job-intelligent-career-pathways-challenge/data)
- in this GitHub repo layout, place the challenge CSV files inside `winning_solution/` before running the script

The goal of this pipeline is to generate the final stacker-based submission family for the AI4EAC Skills2Job Intelligent Career Pathways Challenge, including:

- `submission_general_recall_stacker.csv`
- `submission_general_recall_stacker_blend.csv`

The main entry-point is:

- `make_oof_stacker.py`

This script depends on:

- `make_map_ranker_stack.py`
- `make_hierarchical_ranker.py`

These three Python files must stay together in the same working directory.

## What This Pipeline Produces

When run successfully, the pipeline:

1. loads the official challenge data
2. builds retrieval artifacts and candidate occupation pools
3. trains the base ranking models
4. builds out-of-fold meta-features
5. trains the final stacker
6. writes final submission CSV files

Primary output files:

- `submission_general_recall_stacker.csv`
- `submission_general_recall_stacker_blend.csv`

## Recommended Folder Layout

Run the script from a directory that contains at least the following files:

- `Train.csv`
- `Test.csv`
- `Skills.csv`
- `Occupations.csv`
- `SampleSubmission.csv`
- `make_oof_stacker.py`
- `make_map_ranker_stack.py`
- `make_hierarchical_ranker.py`

Optional but recommended:

- `requirements.txt`
- `SOLUTION_DOCUMENTATION.md`
- the exact submitted artifact `submission_general_recall_stacker_blend.csv`

## Environment Setup

### 1. Open a terminal in the project directory

Example PowerShell command:

```powershell
cd C:\Users\lucia\Downloads\skills2job-intelligent-career-pathways-challenge20260325-11058-1i0y245
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the virtual environment

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can temporarily allow local scripts in the current user scope:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 4. Upgrade `pip`

```powershell
python -m pip install --upgrade pip
```

### 5. Install the required open-source packages

If `requirements.txt` is available:

```powershell
pip install -r repo_readme_target\requirements.txt
```

If you need to install directly, use:

```powershell
pip install numpy pandas scipy scikit-learn xgboost lightgbm implicit sentence-transformers catboost torch threadpoolctl
```

## Hardware Notes

The original work was run on a machine with:

- `NVIDIA GeForce RTX 2050`
- `Intel(R) Arc(TM) Graphics`

Important notes:

- GPU is helpful but not a formal requirement
- the pipeline can still run without identical hardware, but runtime may increase
- exact bitwise reproduction is not guaranteed across machines

## Step-by-Step Execution

### Step 1. Confirm the required files exist

Before running the script, verify that the core data and code files are present:

```powershell
Get-ChildItem
```

Make sure you can see:

- `Train.csv`
- `Test.csv`
- `Skills.csv`
- `Occupations.csv`
- `SampleSubmission.csv`
- `make_oof_stacker.py`
- `make_map_ranker_stack.py`
- `make_hierarchical_ranker.py`

### Step 2. Run the winning pipeline

```powershell
python make_oof_stacker.py
```

This is the only command needed for the final winning stacker family.

### Step 3. Wait for the full pipeline to finish

Approximate runtime on the original machine:

- about `13 to 16 minutes` for `make_oof_stacker.py`

This includes:

- data loading
- feature construction
- candidate generation
- base ranker training
- OOF stacker training
- inference
- submission writing

### Step 4. Confirm the output files were created

After the run completes, verify that these files exist:

- `submission_general_recall_stacker.csv`
- `submission_general_recall_stacker_blend.csv`

You can check with:

```powershell
Get-ChildItem submission_general_recall_stacker*.csv
```

## Internal Dependency Flow

The execution chain is:

- `make_oof_stacker.py`
- `make_map_ranker_stack.py`
- `make_hierarchical_ranker.py`

What each file does:

- `make_oof_stacker.py`
  - final runner
  - builds stacker features
  - trains the meta-ranker
  - writes the final stacker CSV files

- `make_map_ranker_stack.py`
  - core retrieve-and-rerank engine
  - loads the data
  - builds candidate tables
  - trains the main base rankers
  - returns shared training and scoring utilities

- `make_hierarchical_ranker.py`
  - hierarchy helper module
  - supports prefix models and shared utility functions used by the recall pipeline

If any of these three files are missing, the winning script should be considered incomplete.

## Smooth Run Recommendations

To maximize the chances of a clean run:

- keep the three Python scripts in the same directory
- keep the challenge CSV files in the same directory
- use a clean virtual environment
- install all required packages before the first run
- avoid renaming the data files
- do not move the scripts into different folders unless you also update imports
- allow the run to finish fully before checking the outputs

## Reproducibility Notes

Seeds are set in the pipeline, including for:

- `numpy`
- `torch`
- `KFold`
- `train_test_split`
- `XGBoost`
- `LightGBM`
- `ALS`

Main seed value:

- `42`

Important limitation:

- reruns may not be bitwise identical across all environments
- small output differences can occur because of GPU-related nondeterminism and top-k ranking tie behavior

This means:

- the same code path should reproduce the same model family
- the final CSV may be very close rather than perfectly identical

For code review, it is best to include both:

- the exact submitted winning artifact
- the generating code

## Expected Outputs

The main outputs of the winning script are:

- `submission_general_recall_stacker.csv`
- `submission_general_recall_stacker_blend.csv`

The stronger final competition submission from this family was:

- `submission_general_recall_stacker_blend.csv`

## Validation Checks After Running

After the pipeline finishes, verify the following:

1. each output CSV contains an `ID` column
2. each output CSV contains `occ_1` to `occ_5`
3. each row has exactly 5 predicted occupation codes
4. the occupation codes look like real challenge occupation codes, not encoded small integers
5. the output file opens normally in a spreadsheet or pandas

Example quick check in Python:

```powershell
@'
import pandas as pd
df = pd.read_csv("submission_general_recall_stacker_blend.csv")
print(df.head())
print(df.columns.tolist())
'@ | python -
```

## Troubleshooting

### Problem: `ModuleNotFoundError`

Cause:

- one of the dependent scripts is missing
- packages are not installed

Fix:

- verify that `make_map_ranker_stack.py` and `make_hierarchical_ranker.py` are in the same folder as `make_oof_stacker.py`
- verify that the required Python packages are installed

### Problem: Missing CSV file error

Cause:

- one or more challenge data files are not in the working directory

Fix:

- place `Train.csv`, `Test.csv`, `Skills.csv`, `Occupations.csv`, and `SampleSubmission.csv` in the same folder

### Problem: Run is slower than expected

Cause:

- no GPU available
- cold environment with no cached model files
- different package or hardware setup

Fix:

- allow additional time
- avoid running multiple heavy ML jobs at the same time
- use the same machine and environment when possible

### Problem: Output file differs slightly from the original winning artifact

Cause:

- seeded but not perfectly bitwise deterministic pipeline

Fix:

- treat this as expected behavior
- include the original winning artifact together with the code review package
- explain that reruns reproduce the same high-performing model family, not always the exact same CSV bytes

## Recommended Code Review Package

For the cleanest code review submission, include:

- `make_oof_stacker.py`
- `make_map_ranker_stack.py`
- `make_hierarchical_ranker.py`
- `requirements.txt`
- `SOLUTION_DOCUMENTATION.md`
- this runbook
- the exact submitted winning file `submission_general_recall_stacker_blend.csv`

## Final Notes

The correct reviewer-facing story is:

- `make_oof_stacker.py` is the final winning runner
- it depends on `make_map_ranker_stack.py`
- that file depends on `make_hierarchical_ranker.py`
- running the three together with the official challenge data reproduces the winning solution family

For the most reliable rerun, keep the environment simple, keep the files together, install the packages first, and run the pipeline from the project root.
