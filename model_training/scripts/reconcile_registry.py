import json
import os

with open('frontend/public/data/experiment_registry.json') as f:
    registry = json.load(f)

print(f"Total registry entries: {len(registry)}")

categories = {}
methods = {}
splits = {}
missing_checkpoints = []

for i, exp in enumerate(registry):
    eid = exp.get('id')
    method = exp.get('method')
    arch = exp.get('architecture')
    fold = exp.get('fold')
    seed = exp.get('seed')
    split = exp.get('split_strategy')
    status = exp.get('status')
    ckpt = exp.get('checkpoint_path')
    eval_partition = exp.get('evaluation_partition')
    split_fp = exp.get('split_fingerprint')
    
    ckpt_exists = os.path.exists(ckpt) if ckpt else False
    if not ckpt_exists:
        missing_checkpoints.append((eid, ckpt))
        
    print(f"[{i+1:02d}] ID: {eid} | Arch: {arch} | Method: {method} | Fold: {fold} | Seed: {seed} | Status: {status} | EvalPart: {eval_partition} | SplitFP: {str(split_fp)[:10]}... | Ckpt: {'EXISTS' if ckpt_exists else 'MISSING'} ({ckpt})")

print(f"\nMissing checkpoints count: {len(missing_checkpoints)}")
for m in missing_checkpoints:
    print(f"  {m[0]}: {m[1]}")

