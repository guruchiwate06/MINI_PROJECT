import json
from pathlib import Path

exp_dirs = [
    'fold1_cnn_s42',
    'fold2_cnn_s42',
    'fold3_cnn_s42',
    'fold1_tcn_s42',
    'fold2_tcn_s42',
    'fold3_tcn_s42',
    'fold1_rf_s42',
    'fold2_rf_s42',
    'fold3_rf_s42',
]

for ed in exp_dirs:
    p = Path('model_training/artifacts/experiments') / ed
    print(f"\n==================== {ed} ====================")
    # Check run.json
    run_f = p / 'run.json'
    if run_f.exists():
        r = json.load(open(run_f))
        print("Run manifest:", r.get('split_hash') or r.get('manifest_hash') or r.get('split_fingerprint'))
        print("Run config keys:", list(r.keys()))
        if 'config' in r:
            print("Config:", {k: v for k, v in r['config'].items() if 'site' in k or 'fold' in k or 'threshold' in k or 'split' in k})
    
    # Check validation_metrics.json
    vm_f = p / 'validation_metrics.json'
    if vm_f.exists():
        vm = json.load(open(vm_f))
        print("Validation metrics:", vm)
        
    # Check evaluations/test_clean/metrics.json
    tm_f = p / 'evaluations' / 'test_clean' / 'metrics.json'
    if tm_f.exists():
        tm = json.load(open(tm_f))
        print("Test metrics keys / threshold:", tm.get('calibrated_threshold'), tm.get('threshold'), "AP:", tm.get('average_precision') or tm.get('pr_auc'))

