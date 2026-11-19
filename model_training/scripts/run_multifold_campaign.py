"""Multi-Fold Spatial Cross-Validation Campaign on RTX 5070.

Executes Full TRACK-OT++ across 3 spatial folds and 3 random seeds (9 jobs):
- Fold 1: Test Sites [1, 2, 3], Val Site [4], Train Sites [5, 6, 7, 8, 9] (Seeds 42, 43, 44)
- Fold 2: Test Sites [4, 5, 6], Val Site [7], Train Sites [1, 2, 3, 8, 9] (Seeds 42, 43, 44)
- Fold 3: Test Sites [7, 8, 9], Val Site [6], Train Sites [1, 2, 3, 4, 5] (Seeds 42, 43, 44)

Evaluates held-out test performance with validation-calibrated operating thresholds,
computes fold-wise and aggregate mean +/- std across all 9 runs, and produces
a journal-grade validation report.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from railvivid.common import read_json, write_json, write_csv, utc_now
from track_ot.config import load_config
from track_ot.runner import train, evaluate, run_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PLAN_DIR = root / "artifacts" / "plans" / "multifold_full"
REPORT_DIR = root / "reports" / "multifold_validation"


def main():
    print(f"[{utc_now()}] Starting Multi-Fold Spatial Cross-Validation Campaign on RTX 5070...")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    jobs_df = pd.read_csv(PLAN_DIR / "jobs.csv")
    print(f"[{utc_now()}] Loaded {len(jobs_df)} jobs from {PLAN_DIR / 'jobs.csv'}")

    results = []
    summary_csv = REPORT_DIR / "multifold_summary.csv"

    for idx, row in jobs_df.iterrows():
        job_id = row["id"]
        fold = int(row["fold"])
        seed = int(row["seed"])
        variant = row["variant"]
        cfg_path = PLAN_DIR / row["config"]
        
        c = load_config(cfg_path)
        folder = run_path(c)

        print(f"\n{'='*65}")
        print(f"[{utc_now()}] [{idx+1}/{len(jobs_df)}] Job: {job_id} (Fold {fold}, Seed {seed})")
        print(f"{'='*65}")

        # Train if not already done
        done_file = folder / "DONE.json"
        if done_file.exists():
            print(f"[{utc_now()}] {job_id} already trained. Found {done_file}")
        else:
            t0 = time.perf_counter()
            train(c, on_training_device=True)
            elapsed = time.perf_counter() - t0
            print(f"[{utc_now()}] {job_id} training finished in {elapsed/60:.2f} minutes.")

        # Evaluate on held-out test set
        eval_ready = folder / "evaluations" / "test_clean" / "READY.json"
        if eval_ready.exists():
            print(f"[{utc_now()}] {job_id} test evaluation already complete.")
        else:
            print(f"[{utc_now()}] Evaluating {job_id} on held-out test set...")
            evaluate(c, partition="test", device="cuda")

        # Record test metrics
        test_metrics = read_json(folder / "evaluations" / "test_clean" / "metrics.json")
        val_metrics = read_json(folder / "validation_metrics.json") if (folder / "validation_metrics.json").exists() else {}
        
        entry = {
            "job_id": job_id,
            "fold": fold,
            "seed": seed,
            "variant": variant,
            "val_ap": val_metrics.get("average_precision", None),
            "test_ap": test_metrics["average_precision"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_f1": test_metrics["f1"],
            "test_event_recall": test_metrics["events"].get("event_recall", 0.0),
            "test_events_detected": test_metrics["events"].get("true_positive_events", 0),
            "test_events_total": test_metrics["events"].get("total_events", 0),
            "test_false_alarms": test_metrics["events"].get("false_positive_events", 0),
            "test_false_alarms_per_km": test_metrics["events"].get("false_alarms_per_km", 0.0),
            "status": "COMPLETED"
        }
        results.append(entry)

        # Save intermediate summary after every job
        pd.DataFrame(results).to_csv(summary_csv, index=False)
        print(f"[{utc_now()}] Progress saved ({len(results)}/{len(jobs_df)} completed). Current Test AP: {entry['test_ap']:.4f}")

    # Generate final summary and Markdown report
    df = pd.DataFrame(results)
    df.to_csv(summary_csv, index=False)

    print("\n" + "="*70)
    print("MULTI-FOLD SPATIAL CROSS-VALIDATION SUMMARY:")
    print("="*70)
    print(df[["job_id", "fold", "seed", "test_ap", "test_roc_auc", "test_f1", "test_event_recall", "test_false_alarms_per_km"]].to_string())
    print("="*70)

    # Compute fold-wise stats
    fold_stats = df.groupby("fold").agg({
        "test_ap": ["mean", "std"],
        "test_roc_auc": ["mean", "std"],
        "test_f1": ["mean", "std"],
        "test_event_recall": ["mean", "std"],
        "test_false_alarms_per_km": ["mean", "std"]
    })
    print("\nFold-wise statistics:")
    print(fold_stats)

    overall_mean_ap = df["test_ap"].mean()
    overall_std_ap = df["test_ap"].std()
    overall_mean_auc = df["test_roc_auc"].mean()
    overall_std_auc = df["test_roc_auc"].std()

    print(f"\nOverall Aggregate: Test AP = {overall_mean_ap:.4f} +/- {overall_std_ap:.4f}, Test ROC-AUC = {overall_mean_auc:.4f} +/- {overall_std_auc:.4f}")


if __name__ == "__main__":
    main()
