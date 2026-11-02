import pytest
import numpy as np
import pandas as pd
from src.evaluation.splitters import (
    assign_spatial_labels,
    RandomWindowSplitter,
    RunHeldOutSplitter,
    SpatialHeldOutSplitter,
    CombinedRunAndSpatialSplitter,
    GROUND_TRUTH_ANOMALIES
)

def create_synthetic_windows():
    # Synthetic windows simulating runs along the track, explicitly hitting anomaly points
    records = []
    runs = ['AtoB_20_1', 'AtoB_20_2', 'AtoB_20_3', 'AtoB_20_4', 'BtoA_25_4']
    
    # Points along track including exact anomaly coords
    anom_coords = [(a['lat'], a['lon']) for a in GROUND_TRUTH_ANOMALIES]
    inter_coords = [(40.2330, -77.8800), (40.2370, -77.8920)]  # clear normal points
    all_coords = anom_coords + inter_coords

    for r in runs:
        for i, (lat, lon) in enumerate(all_coords):
            records.append({
                'run_id': r,
                'window_idx': i,
                'latitude': lat,
                'longitude': lon,
                'feature_mean': np.random.randn(),
                'feature_rms': np.random.rand()
            })
    df = pd.DataFrame(records)
    return assign_spatial_labels(df, anomaly_radius_m=8.0, buffer_radius_m=20.0)

def test_spatial_labeling():
    df = create_synthetic_windows()
    assert 'label' in df.columns
    assert 'dist_to_nearest_anomaly_m' in df.columns
    assert 'nearest_anomaly_id' in df.columns
    # Check that labels contain -1, 0, 1
    unique_labels = set(df['label'].unique())
    assert 1 in unique_labels
    assert 0 in unique_labels

def test_run_held_out_splitter():
    df = create_synthetic_windows()
    test_runs = ['AtoB_20_4', 'BtoA_25_4']
    val_runs = ['AtoB_20_3']
    splitter = RunHeldOutSplitter(test_runs=test_runs, val_runs=val_runs)
    train_df, val_df, test_df = splitter.split(df)

    # Verify no test runs in train
    assert not set(train_df['run_id']).intersection(set(test_runs))
    assert not set(train_df['run_id']).intersection(set(val_runs))
    assert set(test_df['run_id']).issubset(set(test_runs))
    assert set(val_df['run_id']).issubset(set(val_runs))

def test_spatial_held_out_splitter():
    df = create_synthetic_windows()
    test_anoms = [7, 8, 9]
    val_anoms = [6]
    splitter = SpatialHeldOutSplitter(test_anomaly_ids=test_anoms, val_anomaly_ids=val_anoms)
    train_df, val_df, test_df = splitter.split(df)

    # Test anomalies must never appear in train
    train_anom_ids = set(train_df[train_df['label'] == 1]['nearest_anomaly_id'])
    test_anom_ids = set(test_df[test_df['label'] == 1]['nearest_anomaly_id'])
    val_anom_ids = set(val_df[val_df['label'] == 1]['nearest_anomaly_id'])

    assert not train_anom_ids.intersection(set(test_anoms))
    assert not train_anom_ids.intersection(set(val_anoms))
    assert test_anom_ids.issubset(set(test_anoms))
    assert val_anom_ids.issubset(set(val_anoms))

def test_combined_splitter():
    df = create_synthetic_windows()
    splitter = CombinedRunAndSpatialSplitter()
    train_df, val_df, test_df = splitter.split(df)
    assert len(train_df) > 0
    assert len(test_df) > 0
