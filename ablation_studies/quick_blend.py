"""
BLENDING: Combine top submissions in different ratios
Fastest way to iterate toward 0.7
"""
import pandas as pd
import numpy as np
from pathlib import Path

print("\n" + "="*70, flush=True)
print("ENSEMBLE BLENDING - Testing weight combinations", flush=True)
print("="*70, flush=True)

# Load base submissions
submissions = {
    'hier': 'submission_hier_lgbm.csv',
    'knn_id': 'submission_id_knn.csv',
    'knn_meta': 'submission_meta_knn.csv',
    'blend_w30': 'submission_blend_w030.csv',
}

data = {}
for name, path in submissions.items():
    if Path(path).exists():
        df = pd.read_csv(path)
        data[name] = df
        print(f"✓ Loaded {name:15s}: {df.shape}", flush=True)
    else:
        print(f"✗ Missing {name}", flush=True)

if len(data) < 2:
    print("\n✗ Not enough submissions to blend", flush=True)
    exit(1)

# Assume format: ID, occupation1, occupation2, occupation3, occupation4, occupation5
# We'll blend the occupation predictions

def blend_submissions(subs: dict, weights: dict) -> pd.DataFrame:
    """Average occupation predictions with given weights"""
    result = subs[list(weights.keys())[0]].copy()
    
    # Convert occupations to arrays for averaging
    for col in ['occupation1', 'occupation2', 'occupation3', 'occupation4', 'occupation5']:
        # For simplicity, we'll take majority vote among blended sources
        # (real blending would need richer scoring)
        pass
    
    return result

# Simple approach: rotate between best submissions
blends = [
    ('hier_blw30', {'hier': 0.6, 'blend_w30': 0.4}),
    ('hier_knn_id', {'hier': 0.5, 'knn_id': 0.5}),
    ('knn_meta_knn_id', {'knn_meta': 0.5, 'knn_id': 0.5}),
    ('all_equal', {'hier': 0.25, 'knn_id': 0.25, 'knn_meta': 0.25, 'blend_w30': 0.25}),
]

for blend_name, weights in blends:
    print(f"\n📊 Blend: {blend_name}", flush=True)
    print(f"   Weights: {weights}", flush=True)
    
    # Check all weights have data
    available = all(key in data for key in weights.keys())
    if not available:
        missing = [k for k in weights.keys() if k not in data]
        print(f"   ✗ Missing: {missing}", flush=True)
        continue
    
    # Simple voting blend: for each position, take most common occupation
    result_rows = []
    for idx in range(len(data[list(weights.keys())[0]])):
        row_id = data[list(weights.keys())[0]].iloc[idx]['ID']
        
        # Collect all predictions
        all_preds = []
        for source_name, weight in weights.items():
            for col in [f'occupation{i}' for i in range(1, 6)]:
                if col in data[source_name].columns:
                    occ = data[source_name].iloc[idx][col]
                    all_preds.append((occ, weight))
        
        # Score each occupation by total weight
        occ_scores = {}
        for occ, weight in all_preds:
            occ_scores[occ] = occ_scores.get(occ, 0) + weight
        
        # Top 5 by weight
        top5 = sorted(occ_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        result_rows.append({
            'ID': row_id,
            'occupation1': top5[0][0] if len(top5) > 0 else '',
            'occupation2': top5[1][0] if len(top5) > 1 else '',
            'occupation3': top5[2][0] if len(top5) > 2 else '',
            'occupation4': top5[3][0] if len(top5) > 3 else '',
            'occupation5': top5[4][0] if len(top5) > 4 else '',
        })
    
    result = pd.DataFrame(result_rows)
    out_path = f"submission_blend_{blend_name}.csv"
    result.to_csv(out_path, index=False)
    print(f"   ✓ Created: {out_path}", flush=True)

print("\n" + "="*70, flush=True)
print("Blend submissions created. Ready to submit.", flush=True)
print("="*70 + "\n", flush=True)
