"""
PECOS XLinear for Extreme Multi-Label Ranking
Goal: Use PECOS XLinear to predict top-5 occupations from skill query
Advantages: Built for extreme multi-label, efficient sparse matrix handling, hierarchical support
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70, flush=True)
print("PECOS XLinear - Extreme Multi-Label Ranking", flush=True)
print("="*70, flush=True)

# Load data
print("Loading data...", flush=True)
train_df = pd.read_csv("Train.csv")
test_df = pd.read_csv("Test.csv")
skills_df = pd.read_csv("Skills.csv")
occupations_df = pd.read_csv("Occupations.csv")

print(f"  Train: {train_df.shape}, Test: {test_df.shape}", flush=True)
print(f"  Skills: {skills_df.shape}, Occupations: {occupations_df.shape}", flush=True)

# Build feature matrix for queries (sparse TF-IDF of skill text)
print("\nBuilding feature matrix...", flush=True)
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, vstack

# Map skill IDs to text descriptions
skill_text_map = dict(zip(skills_df['ID'], skills_df['NAME']))
occ_text_map = dict(zip(occupations_df['ID'], occupations_df['OCCUPATION_NAME']))

# Create skill text features for train
train_skill_texts = []
for _, row in train_df.iterrows():
    skills = [str(x) for x in [row['skill_1'], row['skill_2'], row['skill_3'], 
                               row['skill_4'], row['skill_5']] if str(x) != 'nan']
    skill_desc = " ".join([skill_text_map.get(s, s) for s in skills])
    train_skill_texts.append(skill_desc)

# Create skill text features for test
test_skill_texts = []
for _, row in test_df.iterrows():
    skills = [str(x) for x in [row['skill_1'], row['skill_2'], row['skill_3'], 
                               row['skill_4'], row['skill_5']] if str(x) != 'nan']
    skill_desc = " ".join([skill_text_map.get(s, s) for s in skills])
    test_skill_texts.append(skill_desc)

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=300, analyzer='char', ngram_range=(2,3), 
                        min_df=1, max_df=0.95)
X_train_tfidf = tfidf.fit_transform(train_skill_texts)
X_test_tfidf = tfidf.transform(test_skill_texts)

print(f"  Train features (TF-IDF): {X_train_tfidf.shape}", flush=True)
print(f"  Test features (TF-IDF): {X_test_tfidf.shape}", flush=True)

# Build label matrix (occupations as multi-hot)
print("\nBuilding label matrix...", flush=True)
occ_list = sorted(occupations_df['ID'].unique().astype(str))
occ_to_idx = {occ: i for i, occ in enumerate(occ_list)}

# Train labels
train_labels = []
for _, row in train_df.iterrows():
    occs = [str(x) for x in [row['occ_1'], row['occ_2'], row['occ_3'], 
                            row['occ_4'], row['occ_5']] if str(x) != 'nan']
    label_row = np.zeros(len(occ_list), dtype=np.int32)
    for occ in occs:
        if occ in occ_to_idx:
            label_row[occ_to_idx[occ]] = 1
    train_labels.append(label_row)

Y_train = csr_matrix(np.array(train_labels))
print(f"  Train labels: {Y_train.shape}", flush=True)
print(f"  Positive labels: {Y_train.nnz}", flush=True)

# Train PECOS XLinear
print("\nTraining PECOS XLinear...", flush=True)
try:
    from pecos.xmc import XLinearModel
    from pecos.xmc.xlinear.model import XLinearModel as XLModel
    
    # Use the correct PECOS API
    ranker_model = XLinearModel.train(
        X_train_tfidf,
        Y_train,
        max_leaf_size=100,
        shallow_tree=False,
        verbose=True,
    )
    print("  ✓ XLinear training complete", flush=True)
    
except Exception as e:
    print(f"  Note: {e}", flush=True)
    print("  Falling back to simpler PECOS classifier...", flush=True)
    from pecos.xmc import Indexer, XLinearModel
    
    # Simpler approach: train basic XLinear
    ranker_model = XLinearModel.train(
        X_train_tfidf,
        Y_train,
        C_eval_method='f1',
        verbose=True,
    )
    print("  ✓ XLinear training complete", flush=True)

# Predict on test
print("\nGenerating test predictions...", flush=True)
try:
    # Get prediction scores
    Y_pred_scores = ranker_model.predict(X_test_tfidf, top_k=5)
    print(f"  Predictions shape: {Y_pred_scores.shape}", flush=True)
    
    # Convert to occupation IDs
    pred_rows = []
    for i in range(Y_pred_scores.shape[0]):
        # Get indices of top-5 predictions
        indices = Y_pred_scores[i].indices[:5]
        
        occupations = []
        for idx in indices:
            if idx < len(occ_list):
                occupations.append(occ_list[idx])
            else:
                occupations.append("")
        
        # Pad to 5
        while len(occupations) < 5:
            occupations.append("")
        
        pred_rows.append({
            'ID': test_df.iloc[i]['ID'],
            'occupation1': occupations[0],
            'occupation2': occupations[1],
            'occupation3': occupations[2],
            'occupation4': occupations[3],
            'occupation5': occupations[4],
        })
    
    submission = pd.DataFrame(pred_rows)
    
except Exception as e:
    print(f"  Error in prediction: {e}", flush=True)
    import traceback
    traceback.print_exc()
    
    # Fallback: use simple nearest neighbor based on feature similarity
    print("  Falling back to similarity-based ranking...", flush=True)
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarities = cosine_similarity(X_test_tfidf, X_train_tfidf)
    
    pred_rows = []
    for i in range(len(test_df)):
        # Find most similar training samples
        sim_idx = np.argsort(-similarities[i])[:10]
        
        # Get occupations from similar samples
        occ_counter = {}
        for j in sim_idx:
            occs = [str(x) for x in train_df.iloc[j][['occ_1', 'occ_2', 'occ_3', 
                                                        'occ_4', 'occ_5']] 
                   if str(x) != 'nan']
            for occ in occs:
                occ_counter[occ] = occ_counter.get(occ, 0) + 1
        
        # Get top-5
        top5 = sorted(occ_counter.items(), key=lambda x: x[1], reverse=True)[:5]
        occupations = [occ for occ, _ in top5]
        while len(occupations) < 5:
            occupations.append("")
        
        pred_rows.append({
            'ID': test_df.iloc[i]['ID'],
            'occupation1': occupations[0],
            'occupation2': occupations[1],
            'occupation3': occupations[2],
            'occupation4': occupations[3],
            'occupation5': occupations[4],
        })
    
    submission = pd.DataFrame(pred_rows)

# Save submission
output_path = "submission_pecos_xlinear.csv"
submission.to_csv(output_path, index=False)

print(f"\n✓ Submission saved: {output_path}", flush=True)
print(f"  Shape: {submission.shape}", flush=True)
print(f"  Sample:\n{submission.head(3)}", flush=True)

print("\n" + "="*70, flush=True)
print("✓ READY TO SUBMIT", flush=True)
print("="*70, flush=True)
