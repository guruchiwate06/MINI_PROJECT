import pandas as pd
import glob
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

windows = pd.read_csv('model_training/artifacts/railvivid_v1/windows.csv')[['run_id', 'start_idx', 'site_id', 'label']]

for p in sorted(glob.glob('model_training/research_audit/evidence/rf_predictions_*.csv')):
    df = pd.read_csv(p)
    merged = pd.merge(df, windows, on=['run_id', 'start_idx'], suffixes=('_pred', '_win'))
    # filter binary labels
    merged = merged[merged['label_pred'].isin([0, 1])]
    
    for s in [6, 9]:
        sub = merged[merged['site_id'] == s]
        if len(sub) > 0 and (sub['label_pred'] == 1).sum() > 0:
            y_t = sub['label_pred'].to_numpy()
            y_p = sub['prediction'].to_numpy()
            score = sub['score'].to_numpy()
            
            tp = int(((y_t == 1) & (y_p == 1)).sum())
            fp = int(((y_t == 0) & (y_p == 1)).sum())
            fn = int(((y_t == 1) & (y_p == 0)).sum())
            tn = int(((y_t == 0) & (y_p == 0)).sum())
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            ap = average_precision_score(y_t, score) if len(set(y_t)) > 1 else 0
            
            p_name = p.split('\\')[-1]
            print(f"[{p_name}] Site {s}: n={len(sub)}, pos={int(y_t.sum())}, TP={tp}, FP={fp}, FN={fn}, TN={tn} | Prec={prec*100:.1f}%, Rec={rec*100:.1f}%, F1={f1:.4f}, AP={ap:.4f}")

