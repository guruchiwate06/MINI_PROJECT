import os
import json

exp_dir = 'model_training/artifacts/experiments'
for d in ['fold1_cnn_s42', 'fold2_cnn_s42', 'fold3_cnn_s42', 'tcn_combined_seed42']:
    p = os.path.join(exp_dir, d, 'evaluations', 'test_clean')
    if os.path.exists(p):
        print(f"=== {d}/evaluations/test_clean ===")
        print("Files:", os.listdir(p))
        mf = os.path.join(p, 'metrics.json')
        if os.path.exists(mf):
            print("Metrics:", json.load(open(mf)))
        sg = os.path.join(p, 'subgroups.csv')
        if os.path.exists(sg):
            import pandas as pd
            df_sg = pd.read_csv(sg)
            print("Subgroups cols:", list(df_sg.columns))
            print(df_sg.head())

