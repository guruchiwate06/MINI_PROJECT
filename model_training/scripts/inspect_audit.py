import os
import json
import pandas as pd

splits_dir = 'model_training/artifacts/railvivid_v1/splits'
for d in os.listdir(splits_dir):
    p = os.path.join(splits_dir, d)
    if os.path.isdir(p):
        print(f"=== Split dir: {d} ===")
        files = os.listdir(p)
        print("  Files:", files)
        mf = os.path.join(p, 'manifest.csv')
        if os.path.exists(mf):
            df = pd.read_csv(mf)
            print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")
            partition_col = 'partition' if 'partition' in df.columns else ('split' if 'split' in df.columns else None)
            if partition_col:
                print(f"  Partition column: {partition_col}")
                print("  Counts by partition:", df[partition_col].value_counts().to_dict())
                if 'site_id' in df.columns:
                    for part, grp in df.groupby(partition_col):
                        sites = sorted([int(x) for x in grp['site_id'].dropna().unique()])
                        print(f"    Partition '{part}': {len(grp)} rows, sites: {sites}, anomaly_rate: {grp['label'].mean():.4f}")
            if 'spatial_partition' in df.columns:
                print("  spatial_partition counts:", df['spatial_partition'].value_counts().to_dict())
        for f in files:
            if f.endswith('.json'):
                with open(os.path.join(p, f)) as jf:
                    data = json.load(jf)
                    print(f"  JSON {f}: {data}")
        print()

