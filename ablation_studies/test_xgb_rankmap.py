"""
MINIMAL TEST: Does XGBoost rank:map work?
Just test with 500 training samples to get instant feedback
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70, flush=True)
print("TEST: XGBoost rank:map objective", flush=True)
print("="*70, flush=True)

# Load data
print("Loading data...", flush=True)
train_df = pd.read_csv("Train.csv")
test_df = pd.read_csv("Test.csv")
skills_df = pd.read_csv("Skills.csv")
occupations_df = pd.read_csv("Occupations.csv")

print(f"  Train: {train_df.shape}, Test: {test_df.shape}", flush=True)

# QUICK: Use only first 500 train samples
train_df = train_df.iloc[:500].copy()
print(f"  Using subset: {train_df.shape}", flush=True)

# Build features (simplified)
print("Building features...", flush=True)
skills_text = skills_df.set_index('ID')['NAME'].to_dict()
occ_text = occupations_df.set_index('ID')['OCCUPATION_NAME'].to_dict()

# TF-IDF on occupation titles
occ_titles = [occ_text.get(oid, f"occ_{oid}") for oid in occupations_df['ID'].values]
tfidf = TfidfVectorizer(max_features=50, analyzer='char', ngram_range=(2,3))
occ_tfidf = tfidf.fit_transform(occ_titles)

# Create train features
X_train = []
y_train = []
groups_train = []

for qid, group in train_df.groupby('QUERY_ID', sort=False):
    skills = group['SKILL_ID'].values[0:5]
    skill_titles = [skills_text.get(s, f"skill_{s}") for s in skills]
    skill_text = " ".join(skill_titles)
    
    for _, row in group.iterrows():
        occ_id = row['OCC_ID']
        label = row['LABEL']
        
        # Get OCC tfidf
        occ_idx = (occupations_df['OCC_ID'] == occ_id).argmax()
        feat = occ_tfidf[occ_idx].toarray().flatten()
        
        X_train.append(feat)
        y_train.append(label)
    
    groups_train.append(len(group))

X_train = np.array(X_train)
y_train = np.array(y_train, dtype=int)
groups_train = np.array(groups_train)

print(f"  Features shape: {X_train.shape}", flush=True)
print(f"  Groups: {len(groups_train)}, total: {sum(groups_train)}", flush=True)
print(f"  Labels (0/1): {(y_train==0).sum()}/{(y_train==1).sum()}", flush=True)

# Train XGBoost with rank:map
print("\nTraining XGBoost rank:map...", flush=True)
try:
    ranker = xgb.XGBRanker(
        objective="rank:map",
        eval_metric="map@5",
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        tree_method='hist',
        random_state=42,
        verbosity=0,
    )
    
    ranker.fit(
        X_train, 
        y_train,
        group=groups_train,
        eval_set=[(X_train, y_train)],
        eval_group=[groups_train],
    )
    
    print("✓ Training successful!", flush=True)
    
    # Get predictions
    scores = ranker.predict(X_train[:100])  # Just first 100 for quick test
    print(f"✓ Predictions work: shape {scores.shape}, range [{scores.min():.3f}, {scores.max():.3f}]", flush=True)
    
    print("\n" + "="*70, flush=True)
    print("✓✓✓ XGBoost rank:map IS WORKING! ✓✓✓", flush=True)
    print("="*70, flush=True)
    
except Exception as e:
    print(f"✗ FAILED: {type(e).__name__}: {e}", flush=True)
    print("\nTraceback:", flush=True)
    import traceback
    traceback.print_exc()

print("\nDone.", flush=True)
