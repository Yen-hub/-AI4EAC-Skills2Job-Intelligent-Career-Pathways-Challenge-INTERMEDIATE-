"""
FAST MAP RANKER: XGBoost rank:map without expensive cross-encoder
Goal: Quick iteration to validate rank:map improves form 0.368
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70, flush=True)
print("FAST XGBoost rank:map Pipeline", flush=True)
print("="*70, flush=True)

# Import from hierarchical ranker
from make_hierarchical_ranker import (
    OCC_COLS, SKILL_COLS, load_data, build_cooccurrence_artifacts,
    build_query_frame, aggregate_neighbor_scores, compute_cooccurrence_scores,
    apk, rank_dict, fit_prefix_models, align_categoricals
)
import xgboost as xgb
import torch
from sklearn.preprocessing import LabelEncoder

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Loading data...", flush=True)
train, test, skill_meta, occ_meta = load_data()
print(f"  Train: {train.shape}, Test: {test.shape}", flush=True)

# Build features (reuse hierarchical ranker logic but faster)
print("Building features...", flush=True)
artifacts = build_cooccurrence_artifacts(train, skill_meta)
train_rank_df, group_train, _ = build_query_frame(
    train, train, artifacts, occ_meta, label_available=True
)
test_rank_df, _, _ = build_query_frame(
    train, test, artifacts, occ_meta, label_available=False
)
print(f"  Train features: {train_rank_df.shape}, groups: {len(group_train)}", flush=True)
print(f"  Test features: {test_rank_df.shape}", flush=True)

# Prepare for XGBoost 
print("Preparing for XGBoost...", flush=True)

cat_cols = [
    "candidate_occ",
    "prefix2",
    "prefix4",
    "career_area",
    "occupation_group",
    "requirement_level",
    "license_required",
    "cert_required",
    "requires_training",
    "query_cat_signature",
    "query_type_signature",
    "query_subcat_signature",
    "skill_1",
    "skill_2",
    "skill_3",
    "skill_4",
    "skill_5",
    "skill_cat_1",
    "skill_cat_2",
    "skill_cat_3",
    "skill_cat_4",
    "skill_cat_5",
    "skill_type_1",
    "skill_type_2",
    "skill_type_3",
    "skill_type_4",
    "skill_type_5",
]

# Encode categoricals
align_categoricals(train_rank_df, test_rank_df, cat_cols)
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train_rank_df[col] = le.fit_transform(train_rank_df[col].astype(str))
    test_rank_df[col] = le.transform(test_rank_df[col].astype(str))

feature_cols = [c for c in train_rank_df.columns if c not in ["qid", "label"]]
print(f"  Features: {len(feature_cols)}", flush=True)

# Train XGBoost with rank:map
print("\nTraining XGBoost rank:map...", flush=True)
X_train = train_rank_df[feature_cols]
y_train = train_rank_df["label"].astype(int)

ranker = xgb.XGBRanker(
    objective="rank:map",
    eval_metric="map@5",
    tree_method="hist",
    device="cuda" if torch.cuda.is_available() else "cpu",
    n_estimators=200,
    learning_rate=0.08,
    max_depth=7,
    min_child_weight=1.0,
    subsample=0.85,
    colsample_bytree=0.8,
    reg_lambda=1.5,
    random_state=SEED,
    verbosity=0,
)

ranker.fit(X_train, y_train, group=group_train)
print("✓ Training complete", flush=True)

# Predict on test
print("Generating predictions...", flush=True)
X_test = test_rank_df[feature_cols]
test_rank_df["pred_score"] = ranker.predict(X_test)

# Extract top-5 per query
def top5_from_scores(ranked_df, score_col):
    outputs = []
    for qid, group_df in ranked_df.groupby("qid", sort=False):
        df = group_df.sort_values(score_col, ascending=False).head(5)
        occupations = df["candidate_occ"].tolist()
        outputs.append(occupations)
    return outputs

top5 = top5_from_scores(test_rank_df, "pred_score")

# Build submission
submission = pd.DataFrame({
    'ID': test['ID'].astype(str).tolist(),
    'occupation1': [occ[0] if len(occ) > 0 else "" for occ in top5],
    'occupation2': [occ[1] if len(occ) > 1 else "" for occ in top5],
    'occupation3': [occ[2] if len(occ) > 2 else "" for occ in top5],
    'occupation4': [occ[3] if len(occ) > 3 else "" for occ in top5],
    'occupation5': [occ[4] if len(occ) > 4 else "" for occ in top5],
})

output_path = "submission_fast_xgb_map.csv"
submission.to_csv(output_path, index=False)
print(f"\n✓ Saved: {output_path}", flush=True)
print(f"  Shape: {submission.shape}", flush=True)
print(f"  Sample:\n{submission.head()}", flush=True)

print("\n" + "="*70, flush=True)
print("COMPLETE - Ready to submit", flush=True)
print("="*70, flush=True)
