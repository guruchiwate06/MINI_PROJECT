import glob
import os
import json
import pandas as pd

print("Searching all experiments for Site 6 (TP=14, FP=44, FN=8) and Site 9 (TP=18, FP=26, FN=1)...")

# 1. Search subgroups.csv in all experiments
for p in sorted(glob.glob('model_training/artifacts/experiments/**/subgroups.csv', recursive=True)):
    df = pd.read_csv(p)
    if 'group' in df.columns:
        s6 = df[(df['group'] == 'site_id') & (df['value'].astype(str) == '6')]
        s9 = df[(df['group'] == 'site_id') & (df['value'].astype(str) == '9')]
        if not s6.empty:
            r = s6.iloc[0]
            if abs(r['f1'] - 0.35) < 0.02 or abs(r['precision'] - 0.241) < 0.02 or abs(r['recall'] - 0.636) < 0.02:
                print(f"FOUND Site 6 match in {p}:")
                print(r.to_dict())
        if not s9.empty:
            r = s9.iloc[0]
            if abs(r['f1'] - 0.5714) < 0.02 or abs(r['precision'] - 0.409) < 0.02 or abs(r['recall'] - 0.947) < 0.02:
                print(f"FOUND Site 9 match in {p}:")
                print(r.to_dict())

# 2. Check all predictions.csv and validation_predictions.csv
for p in sorted(glob.glob('model_training/artifacts/experiments/**/*prediction*.csv', recursive=True)):
    try:
        df = pd.read_csv(p)
        if 'site_id' in df.columns and 'label' in df.columns:
            # check column for prediction
            pred_col = 'prediction' if 'prediction' in df.columns else ('pred' if 'pred' in df.columns else None)
            prob_col = 'probability' if 'probability' in df.columns else None
            if pred_col:
                for target_site in [6, 9]:
                    sub = df[(df['site_id'] == target_site) & (df['label'].isin([0, 1]))]
                    if len(sub) > 0:
                        y_t = sub['label'].to_numpy()
                        y_p = sub[pred_col].to_numpy()
                        tp = int(((y_t == 1) & (y_p == 1)).sum())
                        fp = int(((y_t == 0) & (y_p == 1)).sum())
                        fn = int(((y_t == 1) & (y_p == 0)).sum())
                        tn = int(((y_t == 0) & (y_p == 0)).sum())
                        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                        if (target_site == 6 and abs(f1 - 0.35) < 0.02) or (target_site == 9 and abs(f1 - 0.5714) < 0.02):
                            print(f"MATCH in {p} for Site {target_site}: TP={tp}, FP={fp}, FN={fn}, TN={tn}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
    except Exception as e:
        pass

