import os
import json
import pandas as pd

print("=== AUDITING EXPERIMENT DIRECTORIES ===")
exp_dir = 'model_training/artifacts/experiments'
records = []
for d in sorted(os.listdir(exp_dir)):
    p = os.path.join(exp_dir, d)
    if not os.path.isdir(p):
        continue
    config_file = os.path.join(p, 'config.json')
    run_file = os.path.join(p, 'run.json')
    val_file = os.path.join(p, 'validation_metrics.json')
    cal_file = os.path.join(p, 'calibration.json')
    done_file = os.path.join(p, 'DONE.json')
    fail_file = os.path.join(p, 'FAILURE.json')
    eval_dir = os.path.join(p, 'evaluations')
    
    cfg = json.load(open(config_file)) if os.path.exists(config_file) else {}
    run = json.load(open(run_file)) if os.path.exists(run_file) else {}
    val = json.load(open(val_file)) if os.path.exists(val_file) else {}
    cal = json.load(open(cal_file)) if os.path.exists(cal_file) else {}
    done = json.load(open(done_file)) if os.path.exists(done_file) else {}
    
    # Check evaluation folder for test metrics
    test_eval = {}
    if os.path.exists(eval_dir):
        for ef in os.listdir(eval_dir):
            ep = os.path.join(eval_dir, ef)
            if os.path.isdir(ep):
                metrics_f = os.path.join(ep, 'metrics.json')
                if os.path.exists(metrics_f):
                    test_eval[ef] = json.load(open(metrics_f))
    
    records.append({
        'dir': d,
        'has_best_pt': os.path.exists(os.path.join(p, 'best.pt')),
        'has_done': os.path.exists(done_file),
        'has_failure': os.path.exists(fail_file),
        'model_type': cfg.get('model', {}).get('type') or cfg.get('model', {}).get('name'),
        'seed': cfg.get('training', {}).get('seed') or cfg.get('seed'),
        'split_fp': run.get('split', {}).get('split_fingerprint'),
        'split_cohort': run.get('split', {}).get('cohort'),
        'val_ap': val.get('average_precision'),
        'cal_threshold': cal.get('threshold'),
        'test_evals': list(test_eval.keys())
    })

print(f"Audited {len(records)} directories.")
for r in records[:15]:
    print(r)
print("...")
for r in records[-10:]:
    print(r)

