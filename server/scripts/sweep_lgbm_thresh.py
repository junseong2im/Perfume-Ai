# -*- coding: utf-8 -*-
"""
LightGBM Consensus Threshold Sweep
===================================
Runs V16 training 4 times with different LGBM_CONSENSUS_THRESH values:
  - 0.00 (no filter, equivalent to V15)
  - 0.50 (loose filter)
  - 0.65 (moderate filter)
  - 0.75 (strict filter, original V16)

Results are logged to weights/sweep_results.json
"""

import subprocess
import sys
import os
import json
import time
import re

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

SCRIPT_PATH = os.path.join('scripts', 'retrain_cos09_v16.py')
RESULTS_FILE = os.path.join('weights', 'sweep_results.json')
THRESHOLDS = [0.00, 0.50, 0.65, 0.75]

def set_threshold(thresh):
    """Replace LGBM_CONSENSUS_THRESH value in v16 script."""
    with open(SCRIPT_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the threshold line
    new_content = re.sub(
        r'LGBM_CONSENSUS_THRESH\s*=\s*[\d.]+',
        f'LGBM_CONSENSUS_THRESH = {thresh}',
        content
    )
    
    with open(SCRIPT_PATH, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"  Set LGBM_CONSENSUS_THRESH = {thresh}")

def parse_results(output):
    """Parse the final results from V16 output."""
    results = {}
    
    # Parse ensemble score
    for line in output.split('\n'):
        line = line.strip()
        # Match lines like "  Ensemble            : 0.8074 <<< BEST"
        m = re.match(r'(\S[\w+\s]*\S)\s*:\s*([\d.]+)', line)
        if m:
            name = m.group(1).strip()
            score = float(m.group(2))
            results[name] = score
        
        # Parse pseudo-label count
        m2 = re.match(r'Pseudo-labeled:\s*(\d+)\s*\(lgbm_filtered=(\d+)\)', line)
        if m2:
            results['pseudo_labeled'] = int(m2.group(1))
            results['lgbm_filtered'] = int(m2.group(2))
        
        # Parse V16 result line
        m3 = re.match(r'V16 result:\s*([\d.]+)', line)
        if m3:
            results['v16_result'] = float(m3.group(1))
    
    return results

def run_experiment(thresh_idx, thresh):
    """Run one V16 training experiment with the given threshold."""
    print(f"\n{'='*70}")
    print(f"  SWEEP EXPERIMENT {thresh_idx+1}/4: LGBM_CONSENSUS_THRESH = {thresh}")
    print(f"{'='*70}")
    
    set_threshold(thresh)
    
    t0 = time.time()
    
    # Run V16 script and capture output
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    proc = subprocess.run(
        [sys.executable, SCRIPT_PATH],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env,
    )
    
    elapsed = time.time() - t0
    
    # Print last 30 lines of output
    output_lines = proc.stdout.split('\n') if proc.stdout else []
    print(f"\n  --- Last 30 lines of output (thresh={thresh}) ---")
    for line in output_lines[-30:]:
        print(f"  {line}")
    
    if proc.returncode != 0:
        print(f"  *** ERROR: Exit code {proc.returncode} ***")
        if proc.stderr:
            stderr_lines = proc.stderr.split('\n')
            for line in stderr_lines[-10:]:
                print(f"  STDERR: {line}")
        return {
            'threshold': thresh,
            'error': True,
            'exit_code': proc.returncode,
            'elapsed_s': elapsed,
            'elapsed_min': round(elapsed / 60, 1),
        }
    
    # Parse results from output
    parsed = parse_results(proc.stdout or '')
    
    result = {
        'threshold': thresh,
        'error': False,
        'exit_code': 0,
        'elapsed_s': elapsed,
        'elapsed_min': round(elapsed / 60, 1),
        'parsed_results': parsed,
    }
    
    # Save model checkpoint with threshold suffix
    import shutil
    v16_path = os.path.join('weights', 'odor_gnn_v16.pt')
    main_path = os.path.join('weights', 'odor_gnn.pt')
    
    suffix = f"thresh_{thresh:.2f}"
    
    # Copy whichever was saved
    for src in [main_path, v16_path]:
        if os.path.exists(src):
            dst = os.path.join('weights', f'odor_gnn_v16_{suffix}.pt')
            shutil.copy2(src, dst)
            print(f"  Saved checkpoint: {dst}")
            break
    
    return result

def main():
    os.makedirs('weights', exist_ok=True)
    
    all_results = []
    
    print(f"Starting LightGBM Consensus Threshold Sweep")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Estimated total time: ~6 hours (4 x ~1.5h)")
    
    t_total = time.time()
    
    for i, thresh in enumerate(THRESHOLDS):
        result = run_experiment(i, thresh)
        all_results.append(result)
        
        # Save intermediate results after each experiment
        with open(RESULTS_FILE, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n  Experiment {i+1}/4 done. Saved to {RESULTS_FILE}")
    
    total_elapsed = time.time() - t_total
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"  SWEEP COMPLETE — FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Threshold':>10s} | {'Pseudo#':>8s} | {'Filtered':>8s} | {'Ensemble':>10s} | {'Time':>8s}")
    print(f"  {'-'*10} | {'-'*8} | {'-'*8} | {'-'*10} | {'-'*8}")
    
    best_score = 0
    best_thresh = None
    
    for r in all_results:
        elapsed_m = r.get('elapsed_min', r.get('elapsed_s', 0) / 60)
        if r.get('error'):
            print(f"  {r['threshold']:>10.2f} | {'ERROR':>8s} | {'---':>8s} | {'---':>10s} | {elapsed_m:>5.1f}min")
            continue
        
        p = r.get('parsed_results', {})
        pseudo = str(p.get('pseudo_labeled', '?'))
        filtered = str(p.get('lgbm_filtered', '?'))
        ensemble = p.get('v16_result', p.get('Ensemble', 0))
        
        if isinstance(ensemble, (int, float)) and ensemble > best_score:
            best_score = ensemble
            best_thresh = r['threshold']
        
        ens_str = f"{ensemble:.4f}" if isinstance(ensemble, (int, float)) else str(ensemble)
        print(f"  {r['threshold']:>10.2f} | {pseudo:>8s} | {filtered:>8s} | {ens_str:>10s} | {elapsed_m:>5.1f}min")
    
    print(f"\n  Best threshold: {best_thresh} (CosSim={best_score:.4f})")
    print(f"  Total time: {total_elapsed/60:.1f}min ({total_elapsed/3600:.1f}h)")
    print(f"  Results saved to: {RESULTS_FILE}")
    print(f"{'='*70}")
    
    # Restore original threshold
    set_threshold(0.75)
    print("  Restored LGBM_CONSENSUS_THRESH = 0.75")

if __name__ == '__main__':
    main()
