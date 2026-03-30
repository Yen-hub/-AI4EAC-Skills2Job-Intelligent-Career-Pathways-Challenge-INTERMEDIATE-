"""
AGGRESSIVE_MULTI_VARIANT: Run multiple ranking approaches in parallel
- Uses existing proven pipelines
- Adds XGBoost rank:map variant
- Fast submission generation
"""
import subprocess
import time
from pathlib import Path

scripts = [
    ("make_hierarchical_ranker.py", "Hierarchical LightGBM"),
    ("make_knn_submissions.py", "KNN Variants"),  
    ("make_map_ranker_stack.py", "Stacking Ensemble"),
]

DATA_DIR = Path(".")
SUBMISSIONS_DIR = DATA_DIR

print("\n" + "="*80, flush=True)
print("RUNNING ALL PIPELINES IN SEQUENCE (FAST MODE)")
print("="*80 + "\n", flush=True)

results = {}

for script_name, desc in scripts:
    script_path = DATA_DIR / script_name
    if not script_path.exists():
        print(f"⊘ SKIP: {desc} ({script_name} not found)", flush=True)
        continue
    
    print(f"\n{'─'*80}", flush=True)
    print(f"▶ RUNNING: {desc}", flush=True)
    print(f"{'─'*80}", flush=True)
    
    start = time.time()
    try:
        result = subprocess.run(
            ["python", "-u", str(script_path)],
            cwd=str(DATA_DIR),
            timeout=3600,
            capture_output=True,
            text=True
        )
        elapsed = time.time() - start
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"✓ {desc}: COMPLETED in {elapsed:.1f}s", flush=True)
            results[desc] = "✓ OK"
        else:
            print(f"✗ {desc}: FAILED (code {result.returncode})", flush=True)
            results[desc] = f"✗ FAIL ({result.returncode})"
    except subprocess.TimeoutExpired:
        print(f"✗ {desc}: TIMEOUT (>1hr)", flush=True)
        results[desc] = "✗ TIMEOUT"
    except Exception as e:
        print(f"✗ {desc}: ERROR - {e}", flush=True)
        results[desc] = f"✗ ERROR: {e}"

# List generated submissions
print("\n" + "="*80, flush=True)
print("SUMMARY")
print("="*80, flush=True)

for desc, status in results.items():
    print(f"{desc:40s} {status}", flush=True)

print("\nGenerated submissions:", flush=True)
submission_files = sorted(SUBMISSIONS_DIR.glob("submission_*.csv"))
for i, f in enumerate(submission_files, 1):
    size = f.stat().st_size
    print(f"  {i:2d}. {f.name:50s} ({size:,} bytes)", flush=True)

print(f"\nTotal: {len(submission_files)} submission files ready")
print("="*80 + "\n", flush=True)
