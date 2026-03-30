"""
PARAMETER SWEEP: Test different hierarchical ranker configs on holdout CV
Quick validation before submitting to leaderboard
"""
import subprocess
import os
import sys
from pathlib import Path
import tempfile
import shutil
import time

print("\n" + "="*70, flush=True)
print("PARAMETER SWEEP FOR HIERARCHICAL RANKER", flush=True)
print("="*70 + "\n", flush=True)

# Create parameter variations
variations = [
    ("lr_0.06_est_600", {
        'learning_rate': 0.06,
        'n_estimators': 600,
    }),
    ("lr_0.08_est_400", {
        'learning_rate': 0.08,
        'n_estimators': 400,
    }),
    ("lr_0.05_est_700", {
        'learning_rate': 0.05,
        'n_estimators': 700,
    }),
    ("lr_0.04_est_500_BASELINE", {
        'learning_rate': 0.04,  # Current
        'n_estimators': 500,    # Current
    }),
]

print(f"Testing {len(variations)} configurations on holdout CV...\n", flush=True)

results = {}
original_script = Path("make_hierarchical_ranker.py")
original_content = original_script.read_text()

for config_name, params in variations:
    print(f"⏱️  [{config_name}]", flush=True)
    
    # Create modified script
    modified_content = original_content
    
    # Replace parameters
    modified_content = modified_content.replace(
        '        learning_rate=0.04,',
        f'        learning_rate={params["learning_rate"]},',
        1
    )
    modified_content = modified_content.replace(
        '        n_estimators=500,',
        f'        n_estimators={params["n_estimators"]},',
        1
    )
    
    # Create temp file
    temp_script = Path(f"_temp_{config_name}.py")
    temp_script.write_text(modified_content)
    
    try:
        # Run with limited verbosity
        result = subprocess.run(
            [sys.executable, str(temp_script)],
            capture_output=True,
            text=True,
            timeout=1200,  # 20 min timeout
        )
        
        output = result.stdout + result.stderr
        
        # Extract MAP@5 score
        cv_score = None
        for line in output.split('\n'):
            if 'holdout MAP@5' in line:
                try:
                    cv_score = float(line.split(':')[-1].strip())
                except:
                    pass
                break
        
        if cv_score:
            results[config_name] = cv_score
            print(f"   ✓ CV Score: {cv_score:.6f}", flush=True)
        else:
            print(f"   ✗ No CV score found in output", flush=True)
            results[config_name] = None
    
    except subprocess.TimeoutExpired:
        print(f"   ✗ TIMEOUT (20 min exceeded)", flush=True)
        results[config_name] = None
    except Exception as e:
        print(f"   ✗ ERROR: {e}", flush=True)
        results[config_name] = None
    finally:
        # Clean up
        try:
            temp_script.unlink()
        except:
            pass

# Summary
print("\n" + "="*70, flush=True)
print("RESULTS SUMMARY", flush=True)
print("="*70 + "\n", flush=True)

valid_results = {k: v for k, v in results.items() if v is not None}

if valid_results:
    sorted_results = sorted(valid_results.items(), key=lambda x: x[1], reverse=True)
    
    for config_name, score in sorted_results:
        improvement = " (BEST!)" if score == max(valid_results.values()) else ""
        print(f"  {config_name:30s}  {score:.6f}{improvement}", flush=True)
    
    best_config, best_score = sorted_results[0]
    print(f"\n🏆 BEST: {best_config} with {best_score:.6f}", flush=True)
    
    # Check if improvement over baseline
    baseline_score = valid_results.get("lr_0.04_est_500_BASELINE", 0)
    if baseline_score > 0:
        improvement_pct = 100 * (best_score - baseline_score) / baseline_score
        print(f"   Improvement over baseline: {improvement_pct:+.2f}%", flush=True)
else:
    print("✗ No valid results", flush=True)

print("\n" + "="*70 + "\n", flush=True)
