import pandas as pd
import glob

for p in glob.glob('model_training/research_audit/evidence/rf_predictions_*.csv'):
    df = pd.read_csv(p)
    print(f"=== {p} ===")
    print("Columns:", list(df.columns))
    if 'site_id' in df.columns:
        for s in [6, 9]:
            sub = df[df['site_id'] == s]
            if not sub.empty:
                y = sub['label']
                # check if there's predicted label or probability
                prob_col = 'probability' if 'probability' in sub.columns else ('predicted_prob' if 'predicted_prob' in sub.columns else None)
                pred_col = 'prediction' if 'prediction' in sub.columns else ('pred' if 'pred' in sub.columns else None)
                print(f"  Site {s}: total={len(sub)}, pos={int((y==1).sum())}, neg={int((y==0).sum())}")
                if pred_col:
                    tp = int(((y==1) & (sub[pred_col]==1)).sum())
                    fp = int(((y==0) & (sub[pred_col]==1)).sum())
                    fn = int(((y==1) & (sub[pred_col]==0)).sum())
                    tn = int(((y==0) & (sub[pred_col]==0)).sum())
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                    print(f"    TP={tp}, FP={fp}, FN={fn}, TN={tn} -> Prec={prec*100:.2f}%, Rec={rec*100:.2f}%, F1={f1:.4f}")

