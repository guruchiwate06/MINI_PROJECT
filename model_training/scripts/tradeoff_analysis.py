"""Tradeoff Analysis: Operating-point sweeps calibrated strictly on validation data.

1. Evaluates all models (TRACK-OT++, CNN, RF, TCN, SSL-TCN) on Fold 1, Seed 42.
2. Selects operating thresholds STRICTLY from validation data across a sweep of FA budgets
   (e.g., 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0 FA/km) and Max-F1.
3. Applies thresholds unchanged to the held-out test predictions.
4. Generates comparative tables and plotting data (FROC and PR curves).
5. Zero retraining required.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score

from railvivid.metrics import select_threshold, event_metrics, classification_metrics

root = Path(__file__).resolve().parent.parent
artifacts = root / "artifacts"
sites = pd.read_csv(root / "artifacts" / "railvivid_v1" / "sites.csv")

MODELS = {
    "TRACK-OT++": artifacts / "track_ot" / "track_ot_pp_fold1_full_recovered_s42",
    "CNN": artifacts / "experiments" / "fold1_cnn_s42",
    "Random Forest": artifacts / "experiments" / "fold1_rf_s42",
    "TCN": artifacts / "experiments" / "fold1_tcn_s42",
    "SSL-TCN": artifacts / "experiments" / "fold1_ssl_tcn_s42",
}

BUDGETS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0]


def run_tradeoff_analysis():
    records = []
    pr_curves = {}
    roc_curves = {}

    for name, exp_path in MODELS.items():
        val_path = exp_path / "validation_predictions.csv"
        test_path = exp_path / "evaluations" / "test_clean" / "predictions.csv"
        
        if not val_path.exists() or not test_path.exists():
            print(f"Skipping {name}: missing prediction artifacts")
            continue
            
        val = pd.read_csv(val_path)
        test = pd.read_csv(test_path)
        
        # Test ground truth
        test_valid = test.loc[test.label.isin([0, 1])].copy()
        y_test = test_valid.label.to_numpy(dtype=int)
        p_test = test_valid.probability.to_numpy(dtype=float)
        
        # Window-level summary
        ap = average_precision_score(y_test, p_test)
        auc = roc_auc_score(y_test, p_test)
        
        # Precision-Recall & ROC data
        prec, rec, _ = precision_recall_curve(y_test, p_test)
        fpr, tpr, _ = roc_curve(y_test, p_test)
        pr_curves[name] = {"precision": prec.tolist(), "recall": rec.tolist(), "ap": float(ap)}
        roc_curves[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "roc_auc": float(auc)}
        
        # 1. Sweep validation FA budgets
        for b in BUDGETS:
            try:
                th = select_threshold(val, sites, {
                    "threshold": "false_alarm_budget",
                    "false_alarm_budget_per_km": b,
                    "event_tolerance_m": 10.0,
                    "event_merge_gap_m": 5.0
                })
                
                # Validation performance at this threshold
                ev_val, _ = event_metrics(val, sites, th, 10.0, 5.0)
                
                # Apply UNCHANGED to held-out test
                ev_test, _ = event_metrics(test, sites, th, 10.0, 5.0)
                cls_test = classification_metrics(test.label, test.probability, th)
                
                records.append({
                    "model": name,
                    "policy": f"FA_budget_{b}",
                    "val_budget_fa_km": b,
                    "selected_threshold": th,
                    "val_event_recall": ev_val["event_recall"],
                    "val_actual_fa_km": ev_val["false_alarms_per_km"],
                    "test_event_recall": ev_test["event_recall"],
                    "test_tp_events": ev_test["true_positive_events"],
                    "test_fp_events": ev_test["false_positive_events"],
                    "test_fn_events": ev_test["false_negative_events"],
                    "test_actual_fa_km": ev_test["false_alarms_per_km"],
                    "test_event_precision": ev_test["event_precision"],
                    "test_localization_mae_m": ev_test["localization_mae_m"],
                    "test_window_precision": cls_test["precision"],
                    "test_window_recall": cls_test["recall"],
                    "test_window_f1": cls_test["f1"],
                    "ap": ap,
                    "roc_auc": auc
                })
            except Exception as e:
                # E.g. if no threshold satisfies the budget on validation
                pass

        # 2. Add Max-F1 validation policy
        try:
            th_f1 = select_threshold(val, sites, {"threshold": "max_f1", "event_tolerance_m": 10.0, "event_merge_gap_m": 5.0})
            ev_val_f1, _ = event_metrics(val, sites, th_f1, 10.0, 5.0)
            ev_test_f1, _ = event_metrics(test, sites, th_f1, 10.0, 5.0)
            cls_test_f1 = classification_metrics(test.label, test.probability, th_f1)
            records.append({
                "model": name,
                "policy": "val_max_f1",
                "val_budget_fa_km": None,
                "selected_threshold": th_f1,
                "val_event_recall": ev_val_f1["event_recall"],
                "val_actual_fa_km": ev_val_f1["false_alarms_per_km"],
                "test_event_recall": ev_test_f1["event_recall"],
                "test_tp_events": ev_test_f1["true_positive_events"],
                "test_fp_events": ev_test_f1["false_positive_events"],
                "test_fn_events": ev_test_f1["false_negative_events"],
                "test_actual_fa_km": ev_test_f1["false_alarms_per_km"],
                "test_event_precision": ev_test_f1["event_precision"],
                "test_localization_mae_m": ev_test_f1["localization_mae_m"],
                "test_window_precision": cls_test_f1["precision"],
                "test_window_recall": cls_test_f1["recall"],
                "test_window_f1": cls_test_f1["f1"],
                "ap": ap,
                "roc_auc": auc
            })
        except Exception:
            pass

    df = pd.DataFrame(records)
    out_dir = root / "reports" / "tradeoff_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "tradeoff_sweep.csv", index=False)
    
    with open(out_dir / "pr_roc_curves.json", "w") as f:
        json.dump({"pr_curves": pr_curves, "roc_curves": roc_curves}, f, indent=2)

    print(f"Tradeoff analysis complete. Saved {len(df)} operating point records to {out_dir}")
    return df


if __name__ == "__main__":
    run_tradeoff_analysis()
