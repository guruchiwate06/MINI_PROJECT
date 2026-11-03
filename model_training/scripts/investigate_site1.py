import pandas as pd

pred1 = pd.read_csv('model_training/artifacts/experiments/fold1_cnn_s42/evaluations/test_clean/predictions.csv')
print("Predictions columns:", list(pred1.columns))
s1 = pred1[pred1['site_id'] == 1]
print(f"Site 1 total rows: {len(s1)}")
print(f"Site 1 label counts:\n{s1['label'].value_counts()}")
print(f"Site 1 score summary for label=0:\n{s1[s1['label']==0]['probability'].describe()}")
print(f"Site 1 score summary for label=1:\n{s1[s1['label']==1]['probability'].describe()}")
print(f"Threshold = 0.80606")
print(f"FP count (label==0 and prob >= thresh): {len(s1[(s1['label']==0) & (s1['probability'] >= 0.80606)])}")

