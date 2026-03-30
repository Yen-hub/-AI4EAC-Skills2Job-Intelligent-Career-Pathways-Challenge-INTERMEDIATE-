# AI4EAC Skills2Job Intelligent Career Pathways Challenge

This repository contains the final reviewed winning solution and the main ablation branches for the Zindi AI4EAC Skills2Job Intelligent Career Pathways Challenge.

The competition task is to predict the top 5 occupation codes from a set of 5 skills, ranked in confidence order and evaluated with `mAP@5`.

## Final Result

The final reviewed winning submission family is based on the out-of-fold stacker pipeline in [`winning_solution/make_oof_stacker.py`](winning_solution/make_oof_stacker.py).

Reviewed winning submission:

- submission file: `submission_general_recall_stacker_blend.csv`
- public score: `0.562135561`
- private score: `0.560219036`

Strong related models from the same family are documented in [`docs/TOP_MODELS.md`](docs/TOP_MODELS.md).

## Repository Structure

```text
.
|-- winning_solution/
|   |-- make_oof_stacker.py
|   |-- make_map_ranker_stack.py
|   |-- make_hierarchical_ranker.py
|   `-- README.md
|-- ablation_studies/
|   |-- README.md
|   `-- *.py
|-- docs/
|   |-- SOLUTION_DOCUMENTATION.md
|   |-- RUNNING_MAKE_OOF_STACKER.md
|   `-- TOP_MODELS.md
|-- requirements.txt
`-- .gitignore
```

## Data

The challenge datasets are intentionally not committed to this repository.

Download them from:

- [Zindi challenge data page](https://zindi.africa/competitions/skills2job-intelligent-career-pathways-challenge/data)

Required files:

- `Train.csv`
- `Test.csv`
- `Skills.csv`
- `Occupations.csv`
- `SampleSubmission.csv`

For the winning pipeline, place those CSV files inside the `winning_solution/` directory before running the code.

For the ablation scripts, place the same CSV files inside `ablation_studies/` if you want to rerun those experiments.

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Yen-hub/-AI4EAC-Skills2Job-Intelligent-Career-Pathways-Challenge-INTERMEDIATE-.git skills2job-winner
cd skills2job-winner
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Zindi data and place it locally

Copy the five challenge CSV files into:

- `winning_solution/`

### 5. Run the final winning pipeline

```bash
cd winning_solution
python make_oof_stacker.py
```

Expected outputs:

- `submission_general_recall_stacker.csv`
- `submission_general_recall_stacker_blend.csv`

## Winning Solution Summary

The final solution is a hybrid retrieve-and-rerank architecture with:

- multi-view candidate retrieval
- hierarchy-aware prefix features
- bagged `XGBoostRanker`
- `LightGBMRanker` as a second ranker
- a final out-of-fold XGBoost stacker

The winning script is not self-contained. The dependency chain is:

- `winning_solution/make_oof_stacker.py`
- `winning_solution/make_map_ranker_stack.py`
- `winning_solution/make_hierarchical_ranker.py`

All three files are required together for the final winning path.

## Reproducibility Notes

The pipeline is seeded and was built to be reproducible, but it is not perfectly bitwise deterministic in every environment.

Main reasons:

- GPU-based XGBoost behavior can vary slightly between reruns
- some ranking stages depend on top-k tie ordering

In practice, reruns reproduce the same strong model family and remain close in score, but exact CSV identity is not guaranteed across all environments.

## Ablation Studies

The [`ablation_studies/`](ablation_studies/) folder contains the main experimental branches explored during development, including:

- earlier kNN and metadata baselines
- pre-recall retrieve-rerank variants
- LightGBM and CatBoost ranking branches
- query-bagging experiments
- NMF and other retrieval baselines
- utility scripts for parameter sweeps and quick blends

See [`ablation_studies/README.md`](ablation_studies/README.md) for an index.

## Documentation

Detailed documentation is included here:

- [`docs/SOLUTION_DOCUMENTATION.md`](docs/SOLUTION_DOCUMENTATION.md)
- [`docs/RUNNING_MAKE_OOF_STACKER.md`](docs/RUNNING_MAKE_OOF_STACKER.md)
- [`docs/TOP_MODELS.md`](docs/TOP_MODELS.md)

## Environment

The winning solution was developed locally on Windows with Python 3.13.

Local winning environment notes:

- GPU used: `NVIDIA GeForce RTX 2050`
- torch build used locally: CUDA-enabled `2.7.1+cu118`

The repository requirements file pins the package versions used during the reviewed solution cycle.

## License and Data Use

This repository contains code and documentation only. Challenge data remains subject to the competition rules and the dataset license on Zindi.
