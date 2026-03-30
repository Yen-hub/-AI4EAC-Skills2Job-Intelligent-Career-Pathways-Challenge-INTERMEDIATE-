"""
QUICK_EXPERIMENTS: Test multiple metric configurations fast
Modifies make_hierarchical_ranker to try:
- metric="map@5" → Direct AP optimization
- Different learning_rates  
- Different n_estimators
"""
import sys
import subprocess
from pathlib import Path
import tempfile
import shutil

def create_variant(orig_script: Path, metric: str, lr: float, n_est: int, name: str) -> Path:
    """Create a modified version of the script with different parameters"""
    content = orig_script.read_text()
    
    # Replace the metric
    content = content.replace(
        'metric="ndcg",',
        f'metric="{metric}",',
        1
    )
    
    # Replace learning_rate
    content = content.replace(
        '        learning_rate=0.04,',
        f'        learning_rate={lr},',
        1
    )
    
    # Replace n_estimators
    content = content.replace(
        '        n_estimators=500,',
        f'        n_estimators={n_est},',
        1
    )
    
    # Replace output filename
    content = content.replace(
        'submission_hier_lgbm.csv',
        f'{name}.csv',
        1
    )
    content = content.replace(
        'submission_hier_lgbm_tail.csv',
        f'{name}_tail.csv',
        1
    )
    
    # Write to temp file
    temp_path = orig_script.parent / f"_temp_{name}.py"
    temp_path.write_text(content)
    
    return temp_path

def run_variant(script_path: Path, name: str, cwd: Path) -> bool:
    """Run a variant and return success"""
    print(f"\n{'='*70}", flush=True)
    print(f"Testing: {name}", flush=True)
    print(f"{'='*70}", flush=True)
    
    try:
        result = subprocess.run(
            ["python", "-u", str(script_path)],
            cwd=str(cwd),
            timeout=600,
            capture_output=False,
        )
        success = result.returncode == 0
        status = "✓ OK" if success else f"✗ FAIL ({result.returncode})"
        print(f"{name}: {status}", flush=True)
        return success
    except subprocess.TimeoutExpired:
        print(f"{name}: ✗ TIMEOUT", flush=True)
        return False
    except Exception as e:
        print(f"{name}: ✗ ERROR - {e}", flush=True)
        return False

def main():
    data_dir = Path(".")
    orig_script = data_dir / "make_hierarchical_ranker.py"
    
    if not orig_script.exists():
        print(f"Error: {orig_script} not found")
        return
    
    variants = [
        # (metric, learning_rate, n_estimators, name)
        ("map@5", 0.05, 400, "submission_hier__map_v1"),
        ("map@5", 0.06, 350, "submission_hier__map_v2"),
        ("ndcg", 0.05, 400, "submission_hier__ndcg_v2"),  # Better ndcg config
    ]
    
    print("\n" + "="*70, flush=True)
    print("RAPID PARAMETER SWEEP (3 variants)")
    print("="*70, flush=True)
    
    results = {}
    temp_scripts = []
    
    try:
        for metric, lr, n_est, name in variants:
            print(f"\n📝 Creating variant: {name}")
            script = create_variant(orig_script, metric, lr, n_est, name)
            temp_scripts.append(script)
            
            print(f"▶ Running variant...", flush=True)
            success = run_variant(script, name, data_dir)
            results[name] = "✓ OK" if success else "✗ FAIL"
    
    finally:
        # Cleanup temp scripts
        for script in temp_scripts:
            try:
                script.unlink()
            except:
                pass
    
    # Summary
    print("\n" + "="*70, flush=True)
    print("RESULTS SUMMARY")
    print("="*70, flush=True)
    for name, status in results.items():
        print(f"  {name:45s} {status}", flush=True)
    
    # List submissions
    print("\n📦 Generated submissions:", flush=True)
    submissions = sorted(data_dir.glob("submission_hier__*.csv"))
    for f in submissions:
        size_kb = f.stat().st_size / 1024
        print(f"   {f.name:50s} ({size_kb:.1f} KB)", flush=True)

if __name__ == "__main__":
    main()
