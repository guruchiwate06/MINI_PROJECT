import json, os
from pathlib import Path
import pandas as pd
import numpy as np

# Search all validation_predictions.csv or evaluate thresholds
for root, dirs, files in os.walk('model_training/artifacts'):
    for f in files:
        if f == 'validation_predictions.csv' or f.endswith('_validation_predictions.csv'):
            p = os.path.join(root, f)
            try:
                df = pd.read_csv(p)
                # check columns
                # target or label, proba or score
                target_col = [c for c in df.columns if c in ['label', 'target', 'y_true', 'is_defect']]
                prob_col = [c for c in df.columns if c in ['probability', 'proba', 'score', 'prob', 'y_pred', 'pred']]
                if not target_col or not prob_col:
                    continue
                y_true = df[target_col[0]].values
                y_prob = df[prob_col[0]].values
                
                # Check at various thresholds
                for t in np.linspace(0.1, 0.9, 81):
                    y_pred = (y_prob >= t).astype(int)
                    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
                    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
                    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
                    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
                    if tp == 14 and fn == 8:
                        if fp == 44 and tn == 386:
                            print(f"EXACT HIT: {p} at threshold {t:.4f} -> TP={tp}, FP={fp}, FN={fn}, TN={tn}")
                        else:
                            # close hit
                            prec = tp / (tp + fp) if (tp+fp) > 0 else 0
                            rec = tp / (tp + fn)
                            f1 = 2 * prec * rec / (prec + rec) if (prec+rec) > 0 else 0
                            if abs(prec - 0.241) < 0.01 or abs(f1 - 0.3500) < 0.01:
                                print(f"CLOSE HIT: {p} at threshold {t:.4f} -> TP={tp}, FP={fp}, FN={fn}, TN={tn}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
            except Exception as e:
                pass

