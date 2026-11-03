import os
import json
import pandas as pd

sg1 = pd.read_csv('model_training/artifacts/experiments/fold1_cnn_s42/evaluations/test_clean/subgroups.csv')
sg2 = pd.read_csv('model_training/artifacts/experiments/fold2_cnn_s42/evaluations/test_clean/subgroups.csv')
sg3 = pd.read_csv('model_training/artifacts/experiments/fold3_cnn_s42/evaluations/test_clean/subgroups.csv')

with open('frontend/public/data/spatial_holdout_site_breakdown.json') as f:
    site_breakdown = json.load(f)

print("Checking Fold 1 sites (1, 2, 3):")
print(sg1[sg1['group'] == 'site_id'][['value', 'observations', 'positive', 'precision', 'recall', 'f1', 'average_precision', 'threshold']])

print("Checking Fold 2 sites (4, 5, 6):")
print(sg2[sg2['group'] == 'site_id'][['value', 'observations', 'positive', 'precision', 'recall', 'f1', 'average_precision', 'threshold']])

print("Checking Fold 3 sites (7, 8, 9):")
print(sg3[sg3['group'] == 'site_id'][['value', 'observations', 'positive', 'precision', 'recall', 'f1', 'average_precision', 'threshold']])

