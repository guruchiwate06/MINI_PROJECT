"""Generate complete publication-grade Tradeoff Analysis report and plots.

Addresses all reviewer questions:
1. Complete FROC curve (Event Recall vs. False Alarms / km) across all models.
2. Comparison at matched operating points (e.g. at ~1 FA/km, ~6 FA/km, ~11 FA/km, ~23 FA/km).
3. Window-level PR and ROC curves.
4. Site-by-site traversal analysis (Sites 1, 2, 3) explaining why certain events were missed at tau=0.886.
5. Strict distinction between validation-calibrated operating points and empirical test curves.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score

from railvivid.metrics import select_threshold, event_metrics, classification_metrics

root = Path(__file__).resolve().parent.parent
sites = pd.read_csv(root / "artifacts" / "railvivid_v1" / "sites.csv")
reports_dir = root / "reports" / "tradeoff_analysis"
reports_dir.mkdir(parents=True, exist_ok=True)

MODELS = {
    "TRACK-OT++": root / "artifacts" / "track_ot" / "track_ot_pp_fold1_full_recovered_s42",
    "CNN": root / "artifacts" / "experiments" / "fold1_cnn_s42",
    "Random Forest": root / "artifacts" / "experiments" / "fold1_rf_s42",
    "TCN": root / "artifacts" / "experiments" / "fold1_tcn_s42",
    "SSL-TCN": root / "artifacts" / "experiments" / "fold1_ssl_tcn_s42",
}

COLORS = {
    "TRACK-OT++": "#1f77b4", # blue
    "CNN": "#ff7f0e",        # orange
    "Random Forest": "#2ca02c", # green
    "TCN": "#d62728",        # red
    "SSL-TCN": "#9467bd",    # purple
}


def analyze():
    print("Loading model predictions...")
    data = {}
    for name, path in MODELS.items():
        val = pd.read_csv(path / "validation_predictions.csv")
        test = pd.read_csv(path / "evaluations" / "test_clean" / "predictions.csv")
        data[name] = {"val": val, "test": test}

    # 1. EMPIRICAL FROC SWEEPS (Test Event Recall vs. False Alarms/km)
    print("Generating FROC sweeps across thresholds...")
    froc_records = []
    
    thresholds = np.unique(np.concatenate([
        np.linspace(0.01, 0.99, 150),
        np.logspace(-4, -0.01, 50)
    ]))
    
    for name, d in data.items():
        test_df = d["test"]
        for th in thresholds:
            ev, _ = event_metrics(test_df, sites, th, 10.0, 5.0)
            froc_records.append({
                "model": name,
                "threshold": float(th),
                "recall": float(ev["event_recall"]),
                "fa_per_km": float(ev["false_alarms_per_km"]),
                "tp": int(ev["true_positive_events"]),
                "fp": int(ev["false_positive_events"]),
                "fn": int(ev["false_negative_events"])
            })

    df_froc = pd.DataFrame(froc_records)
    df_froc.to_csv(reports_dir / "froc_full_sweep.csv", index=False)

    # 2. VALIDATION-CALIBRATED OPERATING POINTS
    # Here we select thresholds using ONLY validation data at various FA budgets and Max-F1
    print("Generating validation-calibrated operating comparisons...")
    calibrated_records = []
    budgets = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0]

    for name, d in data.items():
        val_df = d["val"]
        test_df = d["test"]
        
        # Max-F1 on validation
        try:
            th_f1 = select_threshold(val_df, sites, {"threshold": "max_f1", "event_tolerance_m": 10.0, "event_merge_gap_m": 5.0})
            ev_test, _ = event_metrics(test_df, sites, th_f1, 10.0, 5.0)
            calibrated_records.append({
                "model": name,
                "calibration_policy": "Val Max-F1",
                "threshold": th_f1,
                "test_event_recall": ev_test["event_recall"],
                "test_actual_fa_km": ev_test["false_alarms_per_km"],
                "tp": ev_test["true_positive_events"],
                "fp": ev_test["false_positive_events"],
                "fn": ev_test["false_negative_events"]
            })
        except Exception as e:
            pass

        # FA Budget policies
        for b in [1.0, 5.0, 10.0, 20.0]:
            try:
                th = select_threshold(val_df, sites, {
                    "threshold": "false_alarm_budget",
                    "false_alarm_budget_per_km": b,
                    "event_tolerance_m": 10.0,
                    "event_merge_gap_m": 5.0
                })
                ev_test, _ = event_metrics(test_df, sites, th, 10.0, 5.0)
                calibrated_records.append({
                    "model": name,
                    "calibration_policy": f"Val Budget <= {b} FA/km",
                    "threshold": th,
                    "test_event_recall": ev_test["event_recall"],
                    "test_actual_fa_km": ev_test["false_alarms_per_km"],
                    "tp": ev_test["true_positive_events"],
                    "fp": ev_test["false_positive_events"],
                    "fn": ev_test["false_negative_events"]
                })
            except Exception:
                pass

    df_cal = pd.DataFrame(calibrated_records).drop_duplicates()
    df_cal.to_csv(reports_dir / "validation_calibrated_points.csv", index=False)

    # 3. MATCHED OPERATING POINT COMPARISON
    # What is the event recall of each model when matching actual test false alarm rates?
    print("Computing matched-operating-point comparison...")
    matched_targets = [0.875, 3.5, 6.125, 11.375, 23.625]
    matched_records = []
    
    for target_fa in matched_targets:
        for name, d in data.items():
            sub = df_froc[df_froc.model == name]
            # find threshold where test FA/km <= target_fa + 0.1 with highest recall
            eligible = sub[sub.fa_per_km <= target_fa + 0.1]
            if not eligible.empty:
                best_op = eligible.sort_values(by=["recall", "fa_per_km"], ascending=[False, True]).iloc[0]
                matched_records.append({
                    "target_fa_km": target_fa,
                    "model": name,
                    "threshold": best_op["threshold"],
                    "event_recall": best_op["recall"],
                    "achieved_fa_km": best_op["fa_per_km"],
                    "tp": best_op["tp"],
                    "fp": best_op["fp"],
                    "fn": best_op["fn"]
                })
    df_matched = pd.DataFrame(matched_records)
    df_matched.to_csv(reports_dir / "matched_fa_comparison.csv", index=False)

    # 4. SITE-BY-SITE DETECTION BREAKDOWN
    print("Analyzing per-site traversals on test set...")
    site_records = []
    track_ot_test = data["TRACK-OT++"]["test"]
    
    # Check predictions around the 3 test sites (Site 1: 0m/177.5m, Site 2: 70.0m, Site 3: 130.0m)
    test_sites = sites[sites.id.isin([1, 2, 3])]
    for _, s in test_sites.iterrows():
        s_chain = s["chainage_m"]
        s_id = int(s["id"])
        # Query windows within +-10m
        near = track_ot_test[abs(track_ot_test.chainage_m - s_chain) <= 10.0]
        for run_id, g in near.groupby("run_id"):
            max_p = g.probability.max()
            mean_p = g.probability.mean()
            site_records.append({
                "site_id": s_id,
                "site_type": s["type"],
                "chainage_m": s_chain,
                "run_id": run_id,
                "max_probability": max_p,
                "mean_probability": mean_p,
                "detected_at_0p886": max_p >= 0.886147,
                "detected_at_0p865": max_p >= 0.865455,
                "detected_at_0p80": max_p >= 0.80,
                "detected_at_0p70": max_p >= 0.70,
                "detected_at_0p50": max_p >= 0.50,
            })
    df_sites = pd.DataFrame(site_records)
    df_sites.to_csv(reports_dir / "site_traversal_analysis.csv", index=False)

    # 5. GENERATE PUBLICATION-QUALITY FIGURES
    print("Rendering publication figures...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # A) FROC Curve
    for name, sub in df_froc.groupby("model"):
        # Sort by FA/km for monotonic curve
        sub_s = sub.sort_values("fa_per_km")
        axes[0].plot(sub_s.fa_per_km, sub_s.recall, label=name, color=COLORS[name], linewidth=2)
    
    # Highlight the operating points from the original benchmark report
    axes[0].scatter([0.875], [0.111], color=COLORS["TRACK-OT++"], s=100, zorder=5, label="TRACK-OT++ (@ $\\tau=0.886$)")
    axes[0].scatter([11.375], [0.444], color=COLORS["CNN"], s=100, zorder=5, label="CNN Baseline")
    axes[0].scatter([23.625], [0.667], color=COLORS["Random Forest"], s=100, zorder=5, label="RF Baseline")
    axes[0].set_xlim(0, 30)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlabel("False Alarms / km", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Event Recall (Site Traversals)", fontsize=11, fontweight="bold")
    axes[0].set_title("Free-Response ROC (FROC) Curve", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="lower right", fontsize=9)

    # B) Precision-Recall Curve (Window-Level)
    for name, d in data.items():
        test_valid = d["test"].loc[d["test"].label.isin([0, 1])]
        y = test_valid.label.to_numpy(dtype=int)
        p = test_valid.probability.to_numpy(dtype=float)
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        axes[1].plot(rec, prec, label=f"{name} (AP={ap:.4f})", color=COLORS[name], linewidth=2)
    axes[1].set_xlim(0, 1.0)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Recall (Window Level)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Precision (Window Level)", fontsize=11, fontweight="bold")
    axes[1].set_title("Precision–Recall Curve (Fold 1)", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right", fontsize=9)

    # C) ROC Curve (Window-Level)
    for name, d in data.items():
        test_valid = d["test"].loc[d["test"].label.isin([0, 1])]
        y = test_valid.label.to_numpy(dtype=int)
        p = test_valid.probability.to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        axes[2].plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", color=COLORS[name], linewidth=2)
    axes[2].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[2].set_xlim(0, 1.0)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("True Positive Rate", fontsize=11, fontweight="bold")
    axes[2].set_title("ROC Curve (Fold 1)", fontsize=12, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    plot_path = reports_dir / "tradeoff_curves.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved publication figures to {plot_path}")

    # Copy to Brain Artifacts directory
    brain_artifact = Path("C:/Users/Thrill3r/.gemini/antigravity/brain/776dc837-85e2-4c94-82fd-0b036fa4dd07/tradeoff_curves.png")
    import shutil
    shutil.copy(plot_path, brain_artifact)
    print("Copied artifact to brain directory.")


if __name__ == "__main__":
    analyze()
