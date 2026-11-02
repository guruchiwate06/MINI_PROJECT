"""
Benchmarking Suite for Rail-VIVID Anomaly Detection

Executes standardized baseline models across all four evaluation split strategies:
- Random-Window Split (Leaky control)
- Run-Held-Out Split (Cross-run generalization)
- Spatial-Held-Out Split (Spatial generalization)
- Combined Run-and-Spatial Holdout (Dual generalization)

Evaluates:
1. Isolation Forest (Unsupervised Statistical)
2. Signal Autoencoder (Unsupervised PyTorch Reconstruction)
3. Logistic Regression (Supervised Linear)
4. Random Forest Classifier (Supervised Non-linear)
5. 1D CNN (Multichannel Raw Time-Series)
6. Modality Ablations: Vibration-Only vs Vibration+Env vs Vibration+GPS
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

from src.features.extractor import extract_features_from_run
from src.evaluation.splitters import (
    RandomWindowSplitter,
    RunHeldOutSplitter,
    SpatialHeldOutSplitter,
    CombinedRunAndSpatialSplitter,
    GROUND_TRUTH_ANOMALIES
)
from src.models.statistical_baselines import (
    UnsupervisedIsolationForest,
    SupervisedRandomForest,
    SupervisedLogisticRegression
)
from src.models.autoencoder import SignalAutoencoderDetector
from src.models.cnn1d import CNN1DClassifier


def get_feature_matrix(
    df: pd.DataFrame,
    include_env: bool = False,
    include_gps: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts feature columns and binary labels (0 vs 1)."""
    # Channel feature columns
    feat_cols = [c for c in df.columns if any(c.startswith(f'Channel{i}_') for i in range(1, 7))]

    if include_env:
        feat_cols.extend(['temperature', 'humidity'])
    if include_gps:
        feat_cols.extend(['latitude', 'longitude'])

    X = df[feat_cols].values
    y = df['label'].values
    return X, y


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> Dict[str, Any]:
    """Computes standard classification and ranking metrics."""
    # Ensure binary arrays
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    # In case a split has only 1 class in test set
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return {
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'roc_auc': None,
            'pr_auc': None,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1': round(float(f1), 4),
        'roc_auc': round(float(roc_auc), 4),
        'pr_auc': round(float(pr_auc), 4),
        'confusion_matrix': cm
    }


def main():
    print('Starting Rail-VIVID Benchmarking Pipeline...')
    t0 = time.time()

    cache_path = 'data/cached_window_features.csv'
    if os.path.exists(cache_path):
        print(f'Loading cached window features from {cache_path}...')
        df_all = pd.read_csv(cache_path)
    else:
        print('Extracting window features across all 20 runs...')
        runs = sorted([d for d in os.listdir('.') if os.path.isdir(d) and (d.startswith('AtoB_') or d.startswith('BtoA_'))])
        all_dfs = []
        for r in runs:
            csv_path = os.path.join(r, f'{r}.csv')
            print(f'  Processing {r}...')
            # Window: 1000 samples (0.5s), Stride: 500 samples (0.25s)
            df_r = extract_features_from_run(csv_path, run_id=r, window_size=1000, stride=500)
            all_dfs.append(df_r)

        df_all = pd.concat(all_dfs, ignore_index=True)
        os.makedirs('data', exist_ok=True)
        df_all.to_csv(cache_path, index=False)
        print(f'Saved {len(df_all)} feature windows to {cache_path}.')

    print(f'Total windows: {len(df_all)}')
    print('Label breakdown:')
    print(df_all['label'].value_counts().to_dict())

    # Define Splits to benchmark
    splitters = {
        'Random_Window (Leaky Control)': RandomWindowSplitter(seed=42),
        'Run_Held_Out': RunHeldOutSplitter(),
        'Spatial_Held_Out': SpatialHeldOutSplitter(),
        'Combined_Run_Spatial': CombinedRunAndSpatialSplitter()
    }

    results = []

    for split_name, splitter in splitters.items():
        print(f'\n================ Evaluating Split: {split_name} ================')
        train_df, val_df, test_df = splitter.split(df_all)
        print(f'  Train windows: {len(train_df)} (Anomalies: {sum(train_df["label"]==1)})')
        print(f'  Val windows:   {len(val_df)} (Anomalies: {sum(val_df["label"]==1)})')
        print(f'  Test windows:  {len(test_df)} (Anomalies: {sum(test_df["label"]==1)})')

        if len(test_df) == 0 or sum(test_df['label'] == 1) == 0:
            print('  Skipping: No positive anomalies in test set.')
            continue

        # Feature sets: Vibration-Only, Vibration+Env, Vibration+GPS
        feature_configs = [
            ('Vibration_Only', False, False),
            ('Vibration_Plus_Env', True, False),
            ('Vibration_Plus_GPS', False, True)
        ]

        for feat_name, inc_env, inc_gps in feature_configs:
            X_train, y_train = get_feature_matrix(train_df, include_env=inc_env, include_gps=inc_gps)
            X_test, y_test = get_feature_matrix(test_df, include_env=inc_env, include_gps=inc_gps)

            # 1. Isolation Forest (Unsupervised)
            iso = UnsupervisedIsolationForest(n_estimators=100, seed=42)
            iso.fit(X_train, y_train)
            iso_scores = iso.score_samples(X_test)
            iso_preds = (iso_scores > np.percentile(iso_scores, 90)).astype(int)
            res_iso = evaluate_predictions(y_test, iso_preds, iso_scores)
            results.append({
                'model': 'Isolation Forest',
                'modalities': feat_name,
                'split_strategy': split_name,
                **res_iso,
                'limitations': 'Unsupervised density; sensitive to operational speed shifts.'
            })

            # 2. Random Forest (Supervised)
            rf = SupervisedRandomForest(n_estimators=100, max_depth=10, seed=42)
            rf.fit(X_train, y_train)
            rf_probs = rf.predict_proba(X_test)
            rf_preds = rf.predict(X_test)
            res_rf = evaluate_predictions(y_test, rf_preds, rf_probs)
            results.append({
                'model': 'Random Forest',
                'modalities': feat_name,
                'split_strategy': split_name,
                **res_rf,
                'limitations': 'Prone to memorizing physical track landmarks if coordinates included.'
            })

            # 3. Logistic Regression (Supervised Linear)
            lr = SupervisedLogisticRegression(C=1.0, seed=42)
            lr.fit(X_train, y_train)
            lr_probs = lr.predict_proba(X_test)
            lr_preds = lr.predict(X_test)
            res_lr = evaluate_predictions(y_test, lr_preds, lr_probs)
            results.append({
                'model': 'Logistic Regression',
                'modalities': feat_name,
                'split_strategy': split_name,
                **res_lr,
                'limitations': 'Linear decision boundary; cannot capture nonlinear suspension dynamics.'
            })

            # 4. Signal Autoencoder (Only for Vibration_Only)
            if feat_name == 'Vibration_Only':
                ae = SignalAutoencoderDetector(latent_dim=12, epochs=12, batch_size=64, seed=42)
                ae.fit(X_train, y_train)
                ae_scores = ae.score_samples(X_test)
                ae_preds = (ae_scores > np.percentile(ae_scores, 90)).astype(int)
                res_ae = evaluate_predictions(y_test, ae_preds, ae_scores)
                results.append({
                    'model': 'Signal Autoencoder',
                    'modalities': feat_name,
                    'split_strategy': split_name,
                    **res_ae,
                    'limitations': 'Unsupervised reconstruction; high reconstruction error on rough normal track.'
                })

    df_results = pd.DataFrame(results)
    os.makedirs('reports', exist_ok=True)
    df_results.to_json('reports/baseline_results.json', orient='records', indent=2)

    print('\n================ BENCHMARK RESULTS SUMMARY ================')
    cols_to_show = ['model', 'modalities', 'split_strategy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    print(df_results[cols_to_show].to_string())

    print(f'\nBenchmarking complete in {time.time() - t0:.1f}s.')


if __name__ == '__main__':
    main()
