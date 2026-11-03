import os
import pandas as pd
import numpy as np

for root, dirs, files in os.walk('.'):
    if '.venv' in root or 'node_modules' in root or '.git' in root:
        continue
    for f in files:
        if f.endswith('.csv'):
            p = os.path.join(root, f)
            try:
                # read first line
                with open(p, 'r', encoding='utf-8', errors='ignore') as fh:
                    header = fh.readline()
                if not any(k in header for k in ['prob', 'score', 'pred']):
                    continue
                df = pd.read_csv(p)
                target_col = [c for c in df.columns if c in ['label', 'target', 'y_true', 'is_defect']]
                prob_col = [c for c in df.columns if any(k in c for k in ['prob', 'score', 'pred'])]
                if not target_col or not prob_col:
                    continue
                y_true = df[target_col[0]].values
                y_prob = df[prob_col[0]].values
                # Check lengths
                if len(y_true) in [452, 493, 498, 539, 421, 419]:
                    for t in np.linspace(0.01, 0.99, 100):
                        y_pred = (y_prob >= t).astype(int)
                        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
                        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
                        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
                        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
                        if tp == 14 and fn == 8:
                            prec = tp / (tp + fp) if (tp+fp) > 0 else 0
                            rec = tp / (tp + fn)
                            f1 = 2 * prec * rec / (prec + rec) if (prec+rec) > 0 else 0
                            if fp == 44 or (abs(prec - 0.241) < 0.005 and abs(f1 - 0.3500) < 0.005):
                                print(f"MATCH: {p} at T={t:.4f} -> TP={tp}, FP={fp}, FN={fn}, TN={tn}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
                                break
            except Exception:
                pass

