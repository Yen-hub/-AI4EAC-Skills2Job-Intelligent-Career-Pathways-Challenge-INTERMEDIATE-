# Top Models

This file maps the most important leaderboard models from the project to the code paths in this repository.

## Final Reviewed Winner

### `submission_general_recall_stacker_blend.csv`

- role: final reviewed winning submission
- script: `winning_solution/make_oof_stacker.py`
- public score: `0.562135561`
- private score: `0.560219036`
- notes: final out-of-fold stacker blend built on the recall-expanded retrieval stack

## Strong Related Models

### `submission_general_recall_blend.csv`

- role: strongest public baseline from the same final solution family
- script: `winning_solution/make_map_ranker_stack.py`
- public score: `0.562516248`
- notes: recall-expanded retrieve-and-rerank blend without the final stacker layer

### `submission_general_recall_xgb_bag.csv`

- role: best simpler hedge from the recall-expanded family
- script: `winning_solution/make_map_ranker_stack.py`
- public score: `0.561429897`
- notes: bagged XGBoost ranker without the final stacker blend

### `submission_general_recall_stacker.csv`

- role: direct stacker output without the final blend
- script: `winning_solution/make_oof_stacker.py`
- public score: `0.560362116`
- notes: useful for comparing the effect of the final blended score

### `submission_general_blend.csv`

- role: strongest pre-recall family baseline
- script: `ablation_studies/make_map_ranker_stack_prerecall.py`
- public score: `0.559117920`
- notes: important historical checkpoint before the recall-expanded candidate work

## Default Runtime Assumptions

These models assume:

- challenge CSV files are placed locally next to the script being run
- `USE_FAST_DENSE=1`
- `SECOND_RANKER=lgbm`
- `FINAL_CROSS_ENCODER_TOP_K=0`
- `OOF_FOLDS=3`
- `XGB_BAG_SEEDS=42,73,121`

## Interpretation

The best public score and the final reviewed winner are not the same CSV.

- the best public score came from `submission_general_recall_blend.csv`
- the final reviewed winning submission was `submission_general_recall_stacker_blend.csv`

That distinction is important because the reviewed winner is the code path that survived the full competition and code-review process.
