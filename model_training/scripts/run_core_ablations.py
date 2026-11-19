"""Sequential execution of Core Ablation Study on RTX 5070.

Runs the four critical ablation models on Fold 1, Seed 42:
1. vibration: Vibration-only sequence model (no vision tokens).
2. generic_ot: Standard unconstrained OT (no physical band mask, no geometry).
3. no_drift: Physically constrained OT without learned drift parameter.
4. no_cross_run: OT without cross-run cycle and recurrence consistency losses.

Evaluates each on the held-out test cohort after validation calibration,
and generates the final comparative ablation matrix against full TRACK-OT++.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from railvivid.common import read_json, write_json, write_csv, utc_now
from track_ot.config import load_config
from track_ot.runner import train, evaluate, run_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ABLATION_CONFIGS = [
    root / "artifacts" / "plans" / "core_ablations_fold1" / "configs" / "track_ot_pp_fold1_vibration_s42.json",
    root / "artifacts" / "plans" / "core_ablations_fold1" / "configs" / "track_ot_pp_fold1_generic_ot_s42.json",
    root / "artifacts" / "plans" / "core_ablations_fold1" / "configs" / "track_ot_pp_fold1_no_drift_s42.json",
    root / "artifacts" / "plans" / "core_ablations_fold1" / "configs" / "track_ot_pp_fold1_no_cross_run_s42.json",
]

FULL_MODEL_RUN = root / "artifacts" / "track_ot" / "track_ot_pp_fold1_full_recovered_s42"


def main():
    print(f"[{utc_now()}] Starting Core Ablation Campaign on RTX 5070 (Fold 1, Seed 42)...")
    results = []

    # 1. Load full model metrics
    full_metrics = read_json(FULL_MODEL_RUN / "evaluations" / "test_clean" / "metrics.json")
    results.append({
        "variant": "Full TRACK-OT++",
        "description": "Full multimodal architecture with learned drift, physical band, and cycle consistency",
        "average_precision": full_metrics["average_precision"],
        "roc_auc": full_metrics["roc_auc"],
        "f1": full_metrics["f1"],
        "event_recall": full_metrics["events"]["event_recall"],
        "false_alarms_per_km": full_metrics["events"]["false_alarms_per_km"],
        "status": "COMPLETED"
    })

    # 2. Train and evaluate each ablation variant sequentially
    for cfg_path in ABLATION_CONFIGS:
        c = load_config(cfg_path)
        name = c["experiment"]["name"]
        variant = name.replace("track_ot_pp_fold1_", "").replace("_s42", "")
        folder = run_path(c)
        
        print(f"\n==================================================")
        print(f"[{utc_now()}] Job: {name} (Variant: {variant})")
        print(f"==================================================")

        # Train if not already done
        done_file = folder / "DONE.json"
        if done_file.exists():
            print(f"[{utc_now()}] {name} is already trained. Found {done_file}")
        else:
            t0 = time.perf_counter()
            train(c, on_training_device=True)
            elapsed = time.perf_counter() - t0
            print(f"[{utc_now()}] {name} training finished in {elapsed/60:.2f} minutes.")

        # Evaluate on held-out test set if not already evaluated
        eval_ready = folder / "evaluations" / "test_clean" / "READY.json"
        if eval_ready.exists():
            print(f"[{utc_now()}] {name} test evaluation already complete.")
        else:
            print(f"[{utc_now()}] Evaluating {name} on held-out test set...")
            evaluate(c, partition="test", device="cuda")

        # Record test metrics
        test_metrics = read_json(folder / "evaluations" / "test_clean" / "metrics.json")
        results.append({
            "variant": variant,
            "description": c["experiment"]["notes"],
            "average_precision": test_metrics["average_precision"],
            "roc_auc": test_metrics["roc_auc"],
            "f1": test_metrics["f1"],
            "event_recall": test_metrics["events"]["event_recall"],
            "false_alarms_per_km": test_metrics["events"]["false_alarms_per_km"],
            "status": "COMPLETED"
        })

    # 3. Save comparative summary
    import pandas as pd
    out_dir = root / "reports" / "ablation_study"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "ablation_summary.csv", index=False)
    
    print("\n" + "="*60)
    print("FINAL ABLATION STUDY RESULTS:")
    print("="*60)
    print(df[["variant", "average_precision", "roc_auc", "f1", "event_recall", "false_alarms_per_km"]].to_string())
    print("="*60)


if __name__ == "__main__":
    main()
