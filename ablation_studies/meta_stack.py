"""
META-LEARNER STACKING: Learn optimal blend weights from CV data
Uses: hier_lgbm, id_knn, meta_knn, blend_w030
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70, flush=True)
print("META-LEARNER STACKING", flush=True)
print("="*70, flush=True)

# Load training data to create meta-features
print("=" * 70, flush=True)
print("Creating meta-features from training data...", flush=True)
train_df = pd.read_csv("Train.csv")

# For each query, we know the 5 correct occupations
# Create a binary targets for meta-learner: is this occupation correct?
# Then feed it the rankings from each base model

submissions_paths = {
    'hier': 'submission_hier_lgbm.csv',
    'knn_id': 'submission_id_knn.csv',
    'knn_meta': 'submission_meta_knn.csv',
    'blend_w30': 'submission_blend_w030.csv',
}

submissions = {}
for name, path in submissions_paths.items():
    if Path(path).exists():
        df = pd.read_csv(path)
        submissions[name] = df.set_index('ID')
        print(f"✓ Loaded {name:15s}: {df.shape[0]} rows", flush=True)
    else:
        print(f"✗ Missing {name:15s}: {path}", flush=True)

if len(submissions) < 2:
    print("\n✗ Not enough submissions")
    exit(1)

# Create meta-features by ranking occupations across submissions
# For each (query, occupation) pair, assign ranks from each submission
X_meta = []
y_meta = []
query_ids = []

for query_id, row in train_df.iterrows():
    correct_occs = set([str(row['occ_1']), str(row['occ_2']), str(row['occ_3']),
                       str(row['occ_4']), str(row['occ_5'])])
    query_str_id = str(row['ID'])
    
    if query_str_id not in submissions['hier'].index:
        continue
    
    # Collect all ranked occupations from all models
    all_occs = set()
    for model_name in submissions:
        for i in range(1, 6):
            occ = str(submissions[model_name].loc[query_str_id, f'occ_{i}'])
            all_occs.add(occ)
    
    # Create features: for each occupation, what rank does each model give it?
    for occ in all_occs:
        features = []
        
        for model_name in submissions:
            # Find rank of this occupation in this model's predictions
            predictions = [str(submissions[model_name].loc[query_str_id, f'occ_{j}']) 
                          for j in range(1, 6)]
            try:
                rank = predictions.index(occ) + 1  # 1-based rank
            except ValueError:
                rank = 100  # Not in top-5, assign high rank
            
            features.append(rank)
        
        X_meta.append(features)
        y_meta.append(1 if occ in correct_occs else 0)
        query_ids.append(query_str_id)

X_meta = np.array(X_meta)
y_meta = np.array(y_meta)

print(f"\nMeta-features: X {X_meta.shape}, y {y_meta.shape}", flush=True)
print(f"  Positive samples: {y_meta.sum()} ({100*y_meta.mean():.1f}%)", flush=True)

# Train meta-learner
print("\nTraining meta-learner...", flush=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_meta)

meta_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
meta_model.fit(X_scaled, y_meta)

print("✓ Meta-learner trained", flush=True)
print(f"  Coefficients: {meta_model.coef_[0]}", flush=True)
print(f"  Intercept: {meta_model.intercept_[0]:.3f}", flush=True)

# Now apply meta-learner to test set
test_df = pd.read_csv("Test.csv")

print("\nGenerating predictions with meta-learner...", flush=True)
final_predictions = []

for query_id, row in test_df.iterrows():
    query_str_id = str(row['ID'])
    
    if query_str_id not in submissions['hier'].index:
        # Fallback to hier if missing
        preds = [submissions['hier'].loc[query_str_id, f'occ_{i}'] for i in range(1, 6)]
        final_predictions.append(preds)
        continue
    
    # Collect all occupations from all models
    all_occs_with_scores = {}  # occ -> score
    
    for occ_idx in range(1, 6):
        for model_name in submissions:
            occ = str(submissions[model_name].loc[query_str_id, f'occ_{occ_idx}'])
            
            # Get rank from each model for this occ
            ranks = []
            for other_model in submissions:
                preds = [str(submissions[other_model].loc[query_str_id, f'occ_{j}']) 
                        for j in range(1, 6)]
                try:
                    rank = preds.index(occ) + 1
                except ValueError:
                    rank = 100
                ranks.append(rank)
            
            # Score using meta-learner
            ranks_array = np.array(ranks).reshape(1, -1)
            ranks_scaled = scaler.transform(ranks_array)
            score = meta_model.predict_proba(ranks_scaled)[0, 1]
            
            if occ not in all_occs_with_scores:
                all_occs_with_scores[occ] = score
            else:
                all_occs_with_scores[occ] = max(all_occs_with_scores[occ], score)
    
    # Top 5 by meta-learner score
    top5 = sorted(all_occs_with_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    preds = [occ for occ, score in top5]
    final_predictions.append(preds)

# Build submission
submission = pd.DataFrame({
    'ID': test_df['ID'].astype(str).tolist(),
    'occ_1': [p[0] if len(p) > 0 else '' for p in final_predictions],
    'occ_2': [p[1] if len(p) > 1 else '' for p in final_predictions],
    'occ_3': [p[2] if len(p) > 2 else '' for p in final_predictions],
    'occ_4': [p[3] if len(p) > 3 else '' for p in final_predictions],
    'occ_5': [p[4] if len(p) > 4 else '' for p in final_predictions],
})

output_path = "submission_meta_stack.csv"
submission.to_csv(output_path, index=False)
print(f"\n✓ Saved: {output_path}", flush=True)

print("\n" + "="*70, flush=True)
print("META-STACKING COMPLETE", flush=True)
print("="*70 + "\n", flush=True)
