# AI4EAC Skills2Job Intelligent Career Pathways Challenge

This repository documents an intermediate competition solution for the AI4EAC Skills2Job Intelligent Career Pathways Challenge on Zindi.

As of March 29, 2026, the best public score from this working solution family is **0.562516248** using a retrieval-heavy, out-of-fold trained ranking pipeline.

## Project status

This repository now contains a cleaned code-only snapshot of the four most important model pipelines from the working directory, plus documentation and lightweight dependency notes.

The strongest current family is based on:

- expanded candidate retrieval
- out-of-fold training
- XGBoost `rank:map`
- LightGBM secondary ranking
- normalized score blending

## Repository contents

This code snapshot intentionally keeps only the most important pipeline files:

- `make_map_ranker_stack.py` - strongest current retrieve-rerank stack
- `make_map_ranker_stack_prerecall.py` - legacy runner for the older top pre-recall family
- `make_hierarchical_ranker.py` - hierarchical prefix-aware reranker and shared utilities
- `make_knn_submissions.py` - early KNN and metadata blend family
- `make_lgbm_ranker.py` - alternative LambdaMART-style ranking pipeline
- `TOP_MODELS.md` - exact mapping from leaderboard scores to script entry points
- `requirements.txt` - lightweight dependency list
- `.gitignore` - excludes data files, submissions, logs, and local environment artifacts

## Challenge objective

The task is to predict the top 5 occupations (`occ_1` to `occ_5`) for each query row based on 5 input skills.

Each training example contains:

- 5 skills
- 5 target occupations

The competition metric is **mAP@5**, so the order of the predicted occupations matters.

## Data files used locally

The local working directory contains the following challenge files:

- `Train.csv`
- `Test.csv`
- `Skills.csv`
- `Occupations.csv`
- `SampleSubmission.csv`

These files are expected to live in the same directory as the experiment scripts when reproducing the runs locally.

## Competition rule alignment

This solution path was designed to stay within the competition rules:

- open-source languages and tools only
- only challenge-provided datasets used
- publicly available pretrained models allowed
- no AutoML

The current pipeline uses only public open-source packages such as:

- `pandas`
- `numpy`
- `scikit-learn`
- `scipy`
- `xgboost`
- `lightgbm`
- `implicit`
- `sentence-transformers`
- `catboost`
- `torch`

## Solution evolution

The modeling path moved through several stages:

1. Simple KNN and metadata blends
2. Hierarchical prefix-aware reranking
3. MAP-optimized XGBoost stack with OOF training
4. Generalizing retrieval ensemble with bagged XGBoost and LightGBM
5. Recall-expanded retrieval with graph walk, occupation-profile retrieval, and multi-view query-neighbor bagging

The key lesson from the experiment cycle was that **candidate recall became the main bottleneck** before ranking quality was fully saturated.

## Current best-performing approach

### Retrieval layer

The strongest family builds a broad candidate set from multiple retrieval sources:

- skill-ID TF-IDF query neighbors
- metadata TF-IDF query neighbors
- set-based query neighbor retrieval
- category and type signature neighbor retrieval
- implicit BM25 skill-to-occupation retrieval
- implicit ALS skill-to-occupation retrieval
- co-occurrence retrieval
- skill graph retrieval
- broader graph-walk expansion
- occupation-profile retrieval
- group and career profile retrieval
- semantic occupation text retrieval
- prefix, occupation-group, and career-area expansion

### Ranking layer

The reranking stage uses:

- bagged `XGBoostRanker` with `objective="rank:map"`
- `LightGBMRanker` as a second ranker
- groupwise score normalization
- blended final ranking

### Generalization strategy

To avoid overfitting the public leaderboard, the pipeline emphasizes:

- out-of-fold training features
- exact candidate ceiling audits
- holdout validation
- model-family diversity
- leaderboard use only as a final check

## Current public results

The strongest tested submissions in the current local workspace are:

| Submission file | Public score | Notes |
| --- | ---: | --- |
| `submission_general_recall_blend.csv` | `0.562516248` | Best current public score |
| `submission_general_recall_xgb_bag.csv` | `0.561429897` | Best private hedge candidate |
| `submission_general_blend.csv` | `0.559117920` | Strong earlier blend |
| `submission_general_xgb_bag.csv` | `0.555747446` | Strong earlier XGBoost bag |
| `submission_general_lgbm.csv` | `0.553834726` | Earlier LightGBM variant |
| `submission_general_recall_lgbm.csv` | `0.549498607` | Weaker standalone after recall expansion |

## Important validation findings

An unbiased out-of-fold candidate audit showed that expanding retrieval improved the candidate ceiling:

- previous mean `recall@90`: about `0.8237`
- improved mean `recall@90`: about `0.8527`
- previous proportion of queries with all 5 true occupations inside top-90 candidates: about `0.5595`
- improved proportion: about `0.5971`

This was important because it confirmed there was still real headroom without relying on leaderboard-only tuning.

## Local holdout signal

The strongest recall-expanded family produced the following holdout `MAP@5` values:

- XGBoost bag: `0.630709`
- LightGBM: `0.629708`
- Blend: `0.635368`

These numbers suggest the family is improving for the right reasons, even though public gains are naturally smaller because the leaderboard is only a slice of the full test set.

## Additional local workspace context

The original local workspace also contains extra exploratory scripts, logs, submission files, and audit outputs that are not included in this repo snapshot. The intention here is to keep the GitHub repository focused on the best reusable code paths rather than every experiment artifact.

## Reproducibility notes

To reproduce the strongest current family locally:

1. Place the challenge CSV files in the project root.
2. Install the required Python packages.
3. Run the main pipeline:

```bash
python make_map_ranker_stack.py
```

This produces submission files such as:

- `submission_general_recall_blend.csv`
- `submission_general_recall_xgb_bag.csv`
- `submission_general_recall_lgbm.csv`

## Suggested package install

Example package installation:

```bash
pip install pandas numpy scipy scikit-learn xgboost lightgbm implicit sentence-transformers catboost torch threadpoolctl
```

## What is still missing from this repository

This repository is being initialized with documentation first. The next cleanup steps would normally be:

- push the cleaned experiment scripts
- add a `requirements.txt` or environment file
- add a small runner script for the best public and private-safe pipelines
- add notes on code submission packaging for competition review

## Recommended final bets from the current family

Based on the tested submissions so far:

- **Best public defender:** `submission_general_recall_blend.csv`
- **Best private hedge:** `submission_general_recall_xgb_bag.csv`

## Notes

- The public leaderboard is useful, but it is not the training signal.
- Small public differences should not be overinterpreted.
- Private leaderboard safety comes from OOF validation, recall audits, and model diversity.
