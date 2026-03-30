"""
XGBoost rank:map - Simplified Working Implementation
Goal: Test if rank:map objective improves from 0.376
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70, flush=True)
print("XGBoost rank:map - Simple Implementation", flush=True)
print("="*70, flush=True)

# Import from hierarchical ranker
from make_hierarchical_ranker import (
    OCC_COLS, SKILL_COLS, load_data, build_cooccurrence_artifacts,
    apk, rank_dict
)
import xgboost as xgb
import torch
from sklearn.preprocessing import LabelEncoder

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Load data
print("Loading data...", flush=True)
train, test, skill_meta, occ_meta = load_data()
print(f"  Train: {train.shape}, Test: {test.shape}", flush=True)

# Build features from hierarchical ranker
print("Building features...", flush=True)
artifacts = build_cooccurrence_artifacts(train, skill_meta)

# Build training rank pairs
X_train = []
y_train = []
query_groups = []

for idx, row in train.iterrows():
    skill_ids_in_query = [str(getattr(row, c)) for c in SKILL_COLS]
    true_occs = set([str(x) for x in row[OCC_COLS] if str(x) != 'nan'])
    
    # Get all occupations in dataset
    candidates = sorted(set(occ_meta.index.map(str)))
    
    group_size = 0
    for occ_candidate in candidates[:100]:  # Limit for speed
        # Build features for this (query, candidate) pair
        # Use cooccurrence-based features
        
        # Count matching skills
        matching_skills = sum(1 for s in skill_ids_in_query if f"{s}::{occ_candidate}" in artifacts.get('cooccur', {}))
        
        # Occupation metadata features
        if occ_candidate in occ_meta.index:
            occ_row = occ_meta.loc[occ_candidate]
            prefix2 = str(occ_candidate)[:2]
            prefix4 = str(occ_candidate)[:4]
        else:
            occ_row = pd.Series()
            prefix2 = "00"
            prefix4 = "0000"
        
        # Build feature vector (simple version)
        feat = [
            matching_skills,                                    # how many skills match
            len(skill_ids_in_query),                           # query size
            len(true_occs),                                    # number of true occupations
            1 if occ_candidate in true_occs else 0,            # label
            hash(prefix2) % 100,                               # prefix2 encoded
            hash(prefix4) % 1000,                              # prefix4 encoded
        ]
        
        X_train.append(feat)
        y_train.append(1 if occ_candidate in true_occs else 0)
        group_size += 1
    
    query_groups.append(group_size)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)
query_groups = np.array(query_groups, dtype=np.int32)

print(f"  Training samples: {X_train.shape[0]}")
print(f"  Features per sample: {X_train.shape[1]}")
print(f"  Number of queries: {len(query_groups)}")
print(f"  Positive labels: {(y_train == 1).sum()}")
print(f"  Negative labels: {(y_train == 0).sum()}", flush=True)

# Train XGBoost with rank:map
print("\nTraining XGBoost with rank:map objective...", flush=True)
try:
    ranker = xgb.XGBRanker(
        objective="rank:map",
        eval_metric="map@5",
        tree_method="hist",
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_estimators=200,
        learning_rate=0.1,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        verbosity=0,
    )
    
    ranker.fit(X_train, y_train, group=query_groups)
    print("  ✓ Training complete", flush=True)
except Exception as e:
    print(f"  ✗ Training failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
    exit(1)

# Generate predictions for test
print("\nGenerating test predictions...", flush=True)
submission_data = []

for idx, row in test.iterrows():
    skill_ids_in_query = [str(getattr(row, c)) for c in SKILL_COLS]
    
    scores = []
    candidates = sorted(set(occ_meta.index.map(str)))[:100]  # Limit for speed
    
    for occ_candidate in candidates:
        # Same features as training
        matching_skills = sum(1 for s in skill_ids_in_query if f"{s}::{occ_candidate}" in artifacts.get('cooccur', {}))
        
        if occ_candidate in occ_meta.index:
            prefix2 = str(occ_candidate)[:2]
            prefix4 = str(occ_candidate)[:4]
        else:
            prefix2 = "00"
            prefix4 = "0000"
        
        feat = [
            matching_skills,
            len(skill_ids_in_query),
            0,  # no ground truth available for test
            0,  # placeholder
            hash(prefix2) % 100,
            hash(prefix4) % 1000,
        ]
        
        pred_score = ranker.predict(np.array([feat], dtype=np.float32))[0]
        scores.append((occ_candidate, pred_score))
    
    # Get top 5
    top5 = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
    top5_occs = [occ for occ, _ in top5]
    
    # Pad with empty strings if needed
    while len(top5_occs) < 5:
        top5_occs.append("")
    
    submission_data.append({
        'ID': row['ID'],
        'occupation1': top5_occs[0],
        'occupation2': top5_occs[1],
        'occupation3': top5_occs[2],
        'occupation4': top5_occs[3],
        'occupation5': top5_occs[4],
    })
    
    if (idx + 1) % 200 == 0:
        print(f"  Processed {idx + 1}/{len(test)}", flush=True)

# Save submission
submission = pd.DataFrame(submission_data)
output_path = "submission_xgb_rankmap.csv"
submission.to_csv(output_path, index=False)

print(f"\n✓ Submission saved: {output_path}", flush=True)
print(f"  Shape: {submission.shape}", flush=True)
print(f"  Sample:\n{submission.head(3)}", flush=True)

print("\n" + "="*70, flush=True)
print("✓ READY TO SUBMIT", flush=True)
print("="*70, flush=True)
