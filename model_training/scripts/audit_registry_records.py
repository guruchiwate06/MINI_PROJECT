import json
import os
import glob

with open('frontend/public/data/experiment_registry.json') as f:
    registry = json.load(f)

print(f"Total entries in experiment_registry.json: {len(registry)}")

results = []
for idx, exp in enumerate(registry):
    eid = exp.get('id')
    method = exp.get('method')
    arch = exp.get('architecture')
    fold = exp.get('fold')
    seed = exp.get('seed')
    split_strat = exp.get('split_strategy')
    status = exp.get('status')
    
    # Check directory on disk
    exp_dir = os.path.join('model_training/artifacts/experiments', eid)
    track_dir = os.path.join('model_training/artifacts/track_ot', eid)
    
    actual_dir = None
    if os.path.exists(exp_dir):
        actual_dir = exp_dir
    elif os.path.exists(track_dir):
        actual_dir = track_dir
        
    ckpt_path = None
    split_fp = None
    split_cohort = None
    eval_partition = None
    has_test_eval = False
    
    if actual_dir:
        # Check checkpoint
        for cp in ['best.pt', 'last.pt', 'model.pkl']:
            if os.path.exists(os.path.join(actual_dir, cp)):
                ckpt_path = os.path.join(actual_dir, cp)
                break
        # Read run.json
        run_file = os.path.join(actual_dir, 'run.json')
        if os.path.exists(run_file):
            try:
                run_data = json.load(open(run_file))
                split_fp = run_data.get('split', {}).get('split_fingerprint')
                split_cohort = run_data.get('split', {}).get('cohort')
            except Exception:
                pass
        # Check test_clean
        test_ready = os.path.join(actual_dir, 'evaluations', 'test_clean', 'READY.json')
        if os.path.exists(test_ready):
            has_test_eval = True
            eval_partition = 'test'
        elif os.path.exists(os.path.join(actual_dir, 'evaluations', 'test_clean')):
            has_test_eval = True
            eval_partition = 'test'
        else:
            eval_partition = 'validation' if 'track_ot' in eid else 'unassigned'
    elif 'baseline' in eid:
        actual_dir = 'Phase 1 Benchmark Report'
        ckpt_path = 'model_training/reports/vibration_benchmark/summary.csv'
        eval_partition = 'test'
        
    # Classify record nature
    # e.g., 'fold-experiment (seed variation)', 'independent baseline', etc.
    if 'baseline' in eid:
        rec_type = 'Phase 1 Baseline / Statistical Comparison'
    elif eid == 'track_ot_pp_fold1_full_s42':
        rec_type = 'Multimodal Transformer (Exploratory / Incomplete Training)'
    elif eid == 'tcn_combined_seed42':
        rec_type = 'Independent Combined-Cohort Model'
    else:
        rec_type = f"Fold Experiment (Fold {fold}, Seed {seed})"

    results.append({
        'index': idx + 1,
        'id': eid,
        'method': method,
        'architecture': arch,
        'fold': fold,
        'seed': seed,
        'split_strategy': split_strat,
        'split_fingerprint': split_fp,
        'split_cohort': split_cohort,
        'actual_dir': actual_dir,
        'checkpoint_path': ckpt_path,
        'eval_partition': eval_partition,
        'record_type': rec_type,
        'status': status,
        'f1': exp.get('f1'),
        'average_precision': exp.get('average_precision'),
        'validation_ap': exp.get('validation_ap')
    })

print(f"Audited {len(results)} records.")
# Count categories
cat_counts = {}
for r in results:
    t = r['record_type']
    cat_counts[t] = cat_counts.get(t, 0) + 1
for k, v in cat_counts.items():
    print(f"  {k}: {v}")

with open('model_training/scripts/reconciliation_output.json', 'w') as out:
    json.dump(results, out, indent=2)
print("Saved reconciliation output.")

