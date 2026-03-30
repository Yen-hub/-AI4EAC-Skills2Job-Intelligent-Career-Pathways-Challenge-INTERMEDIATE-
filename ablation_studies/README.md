# Ablation Studies

This folder contains the main experimental branches explored before settling on the final winning stacker pipeline.

The datasets are not committed to git. To rerun any ablation script, place the challenge CSV files in this directory:

- `Train.csv`
- `Test.csv`
- `Skills.csv`
- `Occupations.csv`
- `SampleSubmission.csv`

Download link:

- [Zindi challenge data page](https://zindi.africa/competitions/skills2job-intelligent-career-pathways-challenge/data)

## Main Experiment Groups

### Early retrieval and ranking baselines

- `make_knn_submissions.py`
- `make_lgbm_ranker.py`
- `fast_ranker_v2.py`
- `fast_xgb_map.py`
- `xgb_rankmap_simple.py`

### Intermediate generalizing stacks

- `make_map_ranker_stack_prerecall.py`
- `make_catboost_ranker.py`
- `meta_stack.py`
- `quick_blend.py`
- `param_sweep.py`

### Alternative retrieval branches

- `make_nmf_collab.py`
- `make_querybag_ranker.py`
- `make_pecos_xlinear.py`
- `ultra_ranker_gpu.py`

### Utilities and experiment runners

- `quick_experiments.py`
- `run_all_pipelines.py`
- `test_knn_meta_weights.py`
- `test_xgb_rankmap.py`
- `ensemble_fixed_weights.py`

## Purpose

These scripts are included to show the solution evolution and the main ablation branches considered during the competition. They are not the recommended entry points for reproducing the final winning result.

For the final winning path, use the files in [`../winning_solution/`](../winning_solution/).
