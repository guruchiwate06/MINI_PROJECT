import glob
import json
import pandas as pd
import os

print("=== CHECKING ALL SUBGROUPS FOR SITE 6 AND SITE 9 ===")
for p in sorted(glob.glob('model_training/artifacts/experiments/*/evaluations/test_clean/subgroups.csv')):
    df = pd.read_csv(p)
    exp = os.path.normpath(p).split(os.sep)[3]
    if 'group' in df.columns and 'value' in df.columns:
        s6 = df[(df['group'] == 'site_id') & (df['value'].astype(str) == '6')]
        s9 = df[(df['group'] == 'site_id') & (df['value'].astype(str) == '9')]
        if not s6.empty:
            r = s6.iloc[0]
            print(f"Site 6 in {exp}: Prec={r['precision']:.4f}, Rec={r['recall']:.4f}, F1={r['f1']:.4f}, AP={r['average_precision']:.4f}, Thresh={r['threshold']:.4f}, Obs={r['observations']}, Pos={r['positive']}")
        if not s9.empty:
            r = s9.iloc[0]
            print(f"Site 9 in {exp}: Prec={r['precision']:.4f}, Rec={r['recall']:.4f}, F1={r['f1']:.4f}, AP={r['average_precision']:.4f}, Thresh={r['threshold']:.4f}, Obs={r['observations']}, Pos={r['positive']}")

