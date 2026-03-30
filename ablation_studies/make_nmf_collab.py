"""
Collaborative Filtering via Matrix Factorization
Treats queries as "pseudo-users" with skill features, occupations as "items"
Uses implicit feedback (co-occurrence matrix) with NMF or ALS-like decomposition
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70, flush=True)
print("Matrix Factorization - Collaborative Filtering", flush=True)
print("="*70, flush=True)

# Load data
print("Loading data...", flush=True)
train_df = pd.read_csv("Train.csv")
test_df = pd.read_csv("Test.csv")
skills_df = pd.read_csv("Skills.csv")
occupations_df = pd.read_csv("Occupations.csv")

print(f"  Train: {train_df.shape}, Test: {test_df.shape}", flush=True)

# Build co-occurrence matrix: skill → occupation
print("\nBuilding skill-occupation co-occurrence matrix...", flush=True)
from scipy.sparse import csr_matrix, lil_matrix

# Get unique skills and occupations from training data
SKILL_COLS = [f"skill_{i}" for i in range(1, 6)]
OCC_COLS = [f"occ_{i}" for i in range(1, 6)]

unique_skills = set()
unique_occs = set()

for _, row in train_df.iterrows():
    for col in SKILL_COLS:
        s = str(row[col])
        if s != 'nan':
            unique_skills.add(s)
    for col in OCC_COLS:
        o = str(row[col])
        if o != 'nan':
            unique_occs.add(o)

# Also include test skills
for _, row in test_df.iterrows():
    for col in SKILL_COLS:
        s = str(row[col])
        if s != 'nan':
            unique_skills.add(s)

skills = sorted(unique_skills)
occs = sorted(unique_occs)

skill_to_idx = {s: i for i, s in enumerate(skills)}
occ_to_idx = {o: i for i, o in enumerate(occs)}

# Build co-occurrence in lil format (efficient for construction)
cooccur = lil_matrix((len(skills), len(occs)), dtype=np.float32)

# Count co-occurrences in training data
for _, row in train_df.iterrows():
    skills_in_query = [str(x) for x in row[SKILL_COLS] if str(x) != 'nan']
    occs_in_row = [str(x) for x in row[OCC_COLS] if str(x) != 'nan']
    
    for s in skills_in_query:
        if s in skill_to_idx:
            for o in occs_in_row:
                if o in occ_to_idx:
                    cooccur[skill_to_idx[s], occ_to_idx[o]] += 1.0

# Convert to CSR for efficient operations
cooccur_csr = cooccur.tocsr()
print(f"  Co-occurrence matrix: {cooccur_csr.shape}, NNZ={cooccur_csr.nnz}", flush=True)

# Apply TF-IDF weighting
print("\nApplying TF-IDF weighting...", flush=True)
from sklearn.preprocessing import normalize

# Row normalize (skill normalization)
cooccur_tfidf = normalize(cooccur_csr, norm='l2', axis=1)
print(f"  TF-IDF matrix: {cooccur_tfidf.shape}, density={cooccur_tfidf.nnz/cooccur_tfidf.shape[0]/cooccur_tfidf.shape[1]:.4f}", flush=True)

# Matrix factorization using NMF
print("\nTraining NMF factorization...", flush=True)
from sklearn.decomposition import NMF

# Factorize: skills × occupations → skills × components × components × occupations
n_components = 80

try:
    nmf = NMF(n_components=n_components, init='random', random_state=42, max_iter=200, 
              verbose=0, alpha_W=0.01, l1_ratio=0.0)
    W = nmf.fit_transform(cooccur_csr)  # skill embeddings
    H = nmf.components_  # occupation embeddings
    
    print(f"  ✓ NMF complete: W={W.shape}, H={H.shape}", flush=True)
    
    # Compute reconstructed scores
    recon = W @ H
    
except Exception as e:
    print(f"  NMF failed: {e}, using SVD instead", flush=True)
    from sklearn.decomposition import TruncatedSVD
    
    svd = TruncatedSVD(n_components=80, random_state=42)
    W = svd.fit_transform(cooccur_tfidf)
    H = svd.components_
    recon = W @ H
    
    print(f"  ✓ SVD complete: W={W.shape}, H={H.shape}", flush=True)

# Generate training predictions (for validation)
print("\nGenerating training performance check...", flush=True)
train_correct = 0
train_total = 0

for idx, row in train_df.iterrows():
    skills_in_query = [str(x) for x in row[SKILL_COLS] if str(x) != 'nan']
    true_occs = set(str(x) for x in row[OCC_COLS] if str(x) != 'nan')
    
    # Average reconstruction for skills in query
    skill_indices = [skill_to_idx[s] for s in skills_in_query if s in skill_to_idx]
    
    if skill_indices:
        avg_scores = np.mean(recon[skill_indices], axis=0)
        top5_idx = np.argsort(-avg_scores)[:5]
        top5_occs = set(occs[i] for i in top5_idx)
        
        # Count matches
        matches = len(true_occs & top5_occs)
        train_correct += matches
        train_total += len(true_occs)

train_accuracy = train_correct / max(1, train_total)
print(f"  Training recall@5: {train_accuracy:.4f}", flush=True)

# Generate test predictions
print("\nGenerating test predictions...", flush=True)
submission_data = []

for idx, row in test_df.iterrows():
    skills_in_query = [str(x) for x in row[SKILL_COLS] if str(x) != 'nan']
    
    # Average reconstruction for skills in query
    skill_indices = [skill_to_idx[s] for s in skills_in_query if s in skill_to_idx]
    
    if skill_indices:
        avg_scores = np.mean(recon[skill_indices], axis=0)
    else:
        # Fallback: use mean of all occupations
        avg_scores = np.ones(len(occs))
    
    # Get top-5
    top5_idx = np.argsort(-avg_scores)[:5]
    top5_occs = [occs[i] for i in top5_idx]
    
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
        print(f"  Predicted {idx + 1}/{len(test_df)}", flush=True)

# Save submission
submission = pd.DataFrame(submission_data)
output_path = "submission_nmf_collab.csv"
submission.to_csv(output_path, index=False)

print(f"\n✓ Submission saved: {output_path}", flush=True)
print(f"  Shape: {submission.shape}", flush=True)
print(f"  Sample:\n{submission.head(3)}", flush=True)

print("\n" + "="*70, flush=True)
print("✓ READY TO SUBMIT", flush=True)
print("="*70, flush=True)
