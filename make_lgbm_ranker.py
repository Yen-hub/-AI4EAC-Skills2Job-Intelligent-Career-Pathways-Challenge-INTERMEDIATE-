"""
LGBMRanker - Alternative to XGBoost for ranking
Uses LambdaMART instead of rank:map - different objective, should give different predictions
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70, flush=True)
print("LGBMRanker - LambdaMART Learning-to-Rank Pipeline", flush=True)
print("="*70, flush=True)

# Import from hierarchical ranker
from make_hierarchical_ranker import (
    OCC_COLS, SKILL_COLS, load_data, build_cooccurrence_artifacts,
    build_query_frame, apk, rank_dict
)
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import torch

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Load data
print("Loading data...", flush=True)
train, test, skill_meta, occ_meta = load_data()
print(f"  Train: {train.shape}, Test: {test.shape}", flush=True)

# Build features using hierarchical ranker helper
print("Building features...", flush=True)
artifacts = build_cooccurrence_artifacts(train, skill_meta)

# Build ranking dataset (simplified feature set)
print("\nBuilding rank features...", flush=True)
X_train_data = []
y_train_data = []
group_train = []

# Get all occupation candidates
all_occs = sorted(set(occ_meta.index.map(str)))[:300]  # Limit for speed

for idx, row in train.iterrows():
    skill_ids = [str(getattr(row, c)) for c in SKILL_COLS]
    true_occs = set([str(x) for x in row[OCC_COLS] if str(x) != 'nan'])
    
    group_features = []
    group_labels = []
    
    for occ in all_occs:
        # Feature 1: cooccurrence score
        cooccur_score = sum(1 for s in skill_ids 
                          if f"{s}::{occ}" in artifacts.get('cooccur', {}))
        
        # Feature 2: from occupation metadata
        prefix2 = int(str(occ)[:2]) if len(str(occ)) >= 2 else 0
        prefix4 = int(str(occ)[:4]) if len(str(occ)) >= 4 else 0
        
        # Feature 3: query properties
        query_size = len(skill_ids)
        label_count = len(true_occs)
        
        # Feature 4: Is this a true occupation?
        is_true = 1 if occ in true_occs else 0
        
        feat = [
            cooccur_score,
            prefix2,
            prefix4,
            query_size,
            label_count,
            hash(occ) % 100,
        ]
        
        group_features.append(feat)
        group_labels.append(is_true)
    
    X_train_data.extend(group_features)
    y_train_data.extend(group_labels)
    group_train.append(len(group_features))
    
    if (idx + 1) % 500 == 0:
        print(f"  Processed {idx + 1}/{len(train)}", flush=True)

X_train = np.array(X_train_data, dtype=np.float32)
y_train = np.array(y_train_data, dtype=np.int32)
group_train = np.array(group_train, dtype=np.int32)

print(f"  Train features: {X_train.shape}", flush=True)
print(f"  Train labels: positive={sum(y_train)}, negative={len(y_train)-sum(y_train)}", flush=True)
print(f"  Groups: {len(group_train)}", flush=True)

# Train LGBMRanker
print("\nTraining LGBMRanker (LambdaMART)...", flush=True)
try:
    ranker = lgb.LGBMRanker(
        objective='rank_xendcg',  # XE-NDCG (similar to LambdaMART)
        metric='ndcg',
        num_leaves=63,
        learning_rate=0.05,
        n_estimators=200,
        verbose=-1,
        num_threads=1,
        random_state=SEED,
    )
    
    ranker.fit(
        X_train, y_train,
        group=group_train,
        eval_set=[(X_train, y_train)],
        eval_group=[group_train],
        eval_metric='ndcg@5',
    )
    print("  ✓ LGBMRanker training complete", flush=True)
    
except Exception as e:
    print(f"  Error: {e}", flush=True)
    import traceback
    traceback.print_exc()
    exit(1)

# Generate test predictions
print("\nGenerating test predictions...", flush=True)
submission_data = []

for idx, row in test.iterrows():
    skill_ids = [str(getattr(row, c)) for c in SKILL_COLS]
    
    # Build features for all candidates
    test_features = []
    candidate_occs = []
    
    for occ in all_occs:
        cooccur_score = sum(1 for s in skill_ids 
                          if f"{s}::{occ}" in artifacts.get('cooccur', {}))
        prefix2 = int(str(occ)[:2]) if len(str(occ)) >= 2 else 0
        prefix4 = int(str(occ)[:4]) if len(str(occ)) >= 4 else 0
        query_size = len(skill_ids)
        
        feat = [
            cooccur_score,
            prefix2,
            prefix4,
            query_size,
            0,  # no true labels for test
            hash(occ) % 100,
        ]
        
        test_features.append(feat)
        candidate_occs.append(occ)
    
    test_features = np.array(test_features, dtype=np.float32)
    
    # Get predictions
    scores = ranker.predict(test_features)
    
    # Get top-5
    top5_indices = np.argsort(-scores)[:5]
    top5_occs = [candidate_occs[i] for i in top5_indices]
    
    # Pad if needed
    while len(top5_occs) < 5:
        top5_occs.append("")
    
    submission_data.append({
        'ID': row['ID'],
        'occ_1': top5_occs[0],
        'occ_2': top5_occs[1],
        'occ_3': top5_occs[2],
        'occ_4': top5_occs[3],
        'occ_5': top5_occs[4],
    })
    
    if (idx + 1) % 200 == 0:
        print(f"  Predicted {idx + 1}/{len(test)}", flush=True)

# Save submission
submission = pd.DataFrame(submission_data)
output_path = "submission_lgbm_ranker.csv"
submission.to_csv(output_path, index=False)

print(f"\n✓ Submission saved: {output_path}", flush=True)
print(f"  Shape: {submission.shape}", flush=True)
print(f"  Sample:\n{submission.head(3)}", flush=True)

print("\n" + "="*70, flush=True)
print("✓ READY TO SUBMIT", flush=True)
print("="*70, flush=True)
