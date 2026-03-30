"""
SIMPLE ENSEMBLE: Test different fixed weights on best submissions
"""
import pandas as pd
import numpy as np
from pathlib import Path

print("\n" + "="*70, flush=True)
print("FIXED-WEIGHT ENSEMBLE", flush=True)
print("="*70 + "\n", flush=True)

# Load submissions
subs = {
    'hier': pd.read_csv('submission_hier_lgbm.csv'),
    'knn_id': pd.read_csv('submission_id_knn.csv'),
    'knn_meta': pd.read_csv('submission_meta_knn.csv'),
}

for name in subs:
    print(f"✓ Loaded {name:12s}: {subs[name].shape}", flush=True)

# Try different weight combinations
weight_configs = [
    ('hier_60_id_20_meta_20', {'hier': 0.6, 'knn_id': 0.2, 'knn_meta': 0.2}),
    ('hier_50_id_30_meta_20', {'hier': 0.5, 'knn_id': 0.3, 'knn_meta': 0.2}),
    ('hier_50_id_25_meta_25', {'hier': 0.5, 'knn_id': 0.25, 'knn_meta': 0.25}),
    ('hier_40_id_40_meta_20', {'hier': 0.4, 'knn_id': 0.4, 'knn_meta': 0.2}),
    ('hier_knn_id_40_meta_20', {'hier': 0.4, 'knn_id': 0.4, 'knn_meta': 0.2}),
    ('equal_33_33_33', {'hier': 0.33, 'knn_id': 0.33, 'knn_meta': 0.34}),
]

for config_name, weights in weight_configs:
    print(f"\n🧮 Config: {config_name}", flush=True)
    print(f"   Weights: {weights}", flush=True)
    
    # Weighted vote for each position
    results = []
    for row_idx in range(len(subs['hier'])):
        row_id = subs['hier'].iloc[row_idx]['ID']
        
        # Collect all predictions with weights
        weighted_preds = {}
        
        for source, weight in weights.items():
            for pos in range(1, 6):
                col_name = f'occ_{pos}'
                occ = str(subs[source].iloc[row_idx][col_name])
                
                if occ not in weighted_preds:
                    weighted_preds[occ] = 0
                weighted_preds[occ] += weight
        
        # Get top-5
        top5 = sorted(weighted_preds.items(), key=lambda x: x[1], reverse=True)[:5]
        
        results.append({
            'ID': row_id,
            'occ_1': top5[0][0] if len(top5) > 0 else '',
            'occ_2': top5[1][0] if len(top5) > 1 else '',
            'occ_3': top5[2][0] if len(top5) > 2 else '',
            'occ_4': top5[3][0] if len(top5) > 3 else '',
            'occ_5': top5[4][0] if len(top5) > 4 else '',
        })
    
    result_df = pd.DataFrame(results)
    output_path = f"submission_ensemble_{config_name}.csv"
    result_df.to_csv(output_path, index=False)
    print(f"   💾 Saved: {output_path}", flush=True)

print("\n" + "="*70, flush=True)
print("ENSEMBLES CREATED", flush=True)
print("="*70 + "\n", flush=True)
