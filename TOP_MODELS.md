# Top Models

This file maps the four best public leaderboard submissions from this workspace to the exact script entry points kept in this repository.

## Reproducibility defaults

All four submissions below assume:

- challenge CSV files are in the repository root
- `USE_FAST_DENSE=1`
- `SECOND_RANKER=lgbm`
- `FINAL_CROSS_ENCODER_TOP_K=0`
- `OOF_FOLDS=3`
- `XGB_BAG_SEEDS=42,73,121`

## Version map

### 1. `submission_general_recall_blend.csv`

- Public score: `0.562516248`
- Rank family: recall-expanded retrieve-rerank stack
- Exact script: `make_map_ranker_stack.py`
- Exact version: default `RECALL_EXPANSION=1`
- Command:

```bash
python make_map_ranker_stack.py
```

- Output file of interest: `submission_general_recall_blend.csv`

### 2. `submission_general_recall_xgb_bag.csv`

- Public score: `0.561429897`
- Rank family: recall-expanded retrieve-rerank stack
- Exact script: `make_map_ranker_stack.py`
- Exact version: default `RECALL_EXPANSION=1`
- Command:

```bash
python make_map_ranker_stack.py
```

- Output file of interest: `submission_general_recall_xgb_bag.csv`

### 3. `submission_general_blend.csv`

- Public score: `0.559117920`
- Rank family: pre-recall generalizing stack
- Exact script: `make_map_ranker_stack_prerecall.py`
- Exact version: wrapper-pinned legacy mode with `RECALL_EXPANSION=0`
- Command:

```bash
python make_map_ranker_stack_prerecall.py
```

- Output file of interest: `submission_general_blend.csv`

### 4. `submission_general_xgb_bag.csv`

- Public score: `0.555747446`
- Rank family: pre-recall generalizing stack
- Exact script: `make_map_ranker_stack_prerecall.py`
- Exact version: wrapper-pinned legacy mode with `RECALL_EXPANSION=0`
- Command:

```bash
python make_map_ranker_stack_prerecall.py
```

- Output file of interest: `submission_general_xgb_bag.csv`

## Notes

- The recall-expanded family is the current mainline code path in `make_map_ranker_stack.py`.
- The older top-2 are preserved via `make_map_ranker_stack_prerecall.py` so they remain reproducible after later improvements.
- `make_hierarchical_ranker.py`, `make_knn_submissions.py`, and `make_lgbm_ranker.py` are included in this repo because they were important stepping stones and still provide reusable baselines and utilities, but they are not the current top-4 leaderboard files.
