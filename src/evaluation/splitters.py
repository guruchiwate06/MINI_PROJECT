"""
Evaluation Splitters for Rail-VIVID Anomaly Detection

This module implements leakage-resistant evaluation protocols:
1. RandomWindowSplitter: Leaky baseline (optimistic control; allows overlap across split).
2. BlockGroupedWindowSplitter: Leakage-free temporal block split (prevents window overlap leakage).
3. RunHeldOutSplitter: Holds out complete runs (evaluates cross-run generalization).
4. SpatialHeldOutSplitter: Dynamically partitions track sectors based on anomaly clusters.
5. CombinedRunAndSpatialSplitter: Holds out both unseen runs and unseen spatial anomaly sites.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set


# Standard ground-truth anomalies from Ma et al. (2026), Table 2
GROUND_TRUTH_ANOMALIES = [
    {'id': 1, 'type': 'Joints', 'size_cm': 2.0, 'lat': 40.23553, 'lon': -77.88898},
    {'id': 2, 'type': 'Joints', 'size_cm': 2.0, 'lat': 40.23512, 'lon': -77.88770},
    {'id': 3, 'type': 'Joints', 'size_cm': 2.4, 'lat': 40.23493, 'lon': -77.88711},
    {'id': 4, 'type': 'Divergence', 'size_cm': np.nan, 'lat': 40.23461, 'lon': -77.88612},
    {'id': 5, 'type': 'Convergence', 'size_cm': np.nan, 'lat': 40.23425, 'lon': -77.88485},
    {'id': 6, 'type': 'Joints', 'size_cm': 2.0, 'lat': 40.23388, 'lon': -77.88320},
    {'id': 7, 'type': 'Joints', 'size_cm': 2.5, 'lat': 40.23242, 'lon': -77.87795},
    {'id': 8, 'type': 'Joints', 'size_cm': 2.3, 'lat': 40.23205, 'lon': -77.87651},
    {'id': 9, 'type': 'Joints', 'size_cm': 3.0, 'lat': 40.23194, 'lon': -77.87508},
]


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance between two GPS coordinates in meters."""
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * R * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def assign_spatial_labels(
    df: pd.DataFrame,
    anomaly_radius_m: float = 6.0,
    buffer_radius_m: float = 18.0,
    anomalies: List[Dict] = GROUND_TRUTH_ANOMALIES
) -> pd.DataFrame:
    """
    Labels windows based on distance to known physical anomalies:
      - 1: Definite Anomaly (within anomaly_radius_m of an anomaly)
      - -1: Buffer / Transition Zone (within buffer_radius_m, excluded from training to prevent leakage)
      - 0: Conservative Normal Track (> buffer_radius_m from all anomalies)
    Also annotates `nearest_anomaly_id` and `dist_to_nearest_anomaly_m`.
    """
    df = df.copy()
    min_dists = np.full(len(df), np.inf)
    nearest_ids = np.zeros(len(df), dtype=int)

    lat_arr = df['latitude'].values if 'latitude' in df.columns else df['Latitude'].values
    lon_arr = df['longitude'].values if 'longitude' in df.columns else df['Longitude'].values

    for anom in anomalies:
        dists = np.array([haversine_distance(lat, lon, anom['lat'], anom['lon'])
                          for lat, lon in zip(lat_arr, lon_arr)])
        closer = dists < min_dists
        min_dists[closer] = dists[closer]
        nearest_ids[closer] = anom['id']

    labels = np.zeros(len(df), dtype=int)
    labels[min_dists <= anomaly_radius_m] = 1
    labels[(min_dists > anomaly_radius_m) & (min_dists <= buffer_radius_m)] = -1

    df['dist_to_nearest_anomaly_m'] = min_dists
    df['nearest_anomaly_id'] = nearest_ids
    df['label'] = labels
    return df


class RandomWindowSplitter:
    """
    Splits windows purely at random.
    WARNING: Highly susceptible to temporal correlation leakage due to sliding window overlap!
    Provided strictly as an optimistic upper-bound reference baseline.
    """
    def __init__(self, val_ratio: float = 0.15, test_ratio: float = 0.20, seed: int = 42):
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        valid_df = df[df['label'] != -1].copy()
        n = len(valid_df)
        np.random.seed(self.seed)
        shuffled_indices = np.random.permutation(n)

        n_test = int(n * self.test_ratio)
        n_val = int(n * self.val_ratio)

        test_idx = shuffled_indices[:n_test]
        val_idx = shuffled_indices[n_test:n_test + n_val]
        train_idx = shuffled_indices[n_test + n_val:]

        return (
            valid_df.iloc[train_idx].copy(),
            valid_df.iloc[val_idx].copy(),
            valid_df.iloc[test_idx].copy()
        )


class BlockGroupedWindowSplitter:
    """
    Groups contiguous windows into blocks (e.g. 20 consecutive windows = 5 seconds)
    before splitting, preventing adjacent overlapping windows from leaking between train and test.
    """
    def __init__(self, block_size: int = 20, val_ratio: float = 0.15, test_ratio: float = 0.20, seed: int = 42):
        self.block_size = block_size
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        valid_df = df[df['label'] != -1].copy()
        run_col = 'run_id' if 'run_id' in valid_df.columns else 'run'
        
        # Assign block ID per run
        valid_df['block_id'] = valid_df.groupby(run_col).cumcount() // self.block_size
        unique_blocks = valid_df[[run_col, 'block_id']].drop_duplicates().values
        
        np.random.seed(self.seed)
        shuffled_blocks = np.random.permutation(len(unique_blocks))
        n_test = int(len(unique_blocks) * self.test_ratio)
        n_val = int(len(unique_blocks) * self.val_ratio)
        
        test_block_tuples = set(map(tuple, unique_blocks[shuffled_blocks[:n_test]]))
        val_block_tuples = set(map(tuple, unique_blocks[shuffled_blocks[n_test:n_test + n_val]]))
        
        row_tuples = list(zip(valid_df[run_col], valid_df['block_id']))
        test_mask = [t in test_block_tuples for t in row_tuples]
        val_mask = [t in val_block_tuples for t in row_tuples]
        train_mask = [not (t or v) for t, v in zip(test_mask, val_mask)]
        
        return (
            valid_df[train_mask].drop(columns=['block_id']).copy(),
            valid_df[val_mask].drop(columns=['block_id']).copy(),
            valid_df[test_mask].drop(columns=['block_id']).copy()
        )


class RunHeldOutSplitter:
    """
    Holds out complete runs. Training occurs on a subset of runs; validation and
    testing evaluate cross-run temporal generalization across speeds/directions.
    """
    def __init__(
        self,
        test_runs: List[str] = ['AtoB_20_4', 'AtoB_22.5_4', 'BtoA_25_4', 'BtoA_27.5_4'],
        val_runs: List[str] = ['AtoB_20_3', 'BtoA_22.5_3', 'BtoA_27.5_3']
    ):
        self.test_runs = set(test_runs)
        self.val_runs = set(val_runs)

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        run_col = 'run_id' if 'run_id' in df.columns else 'run'
        valid_df = df[df['label'] != -1].copy()

        train_mask = ~valid_df[run_col].isin(self.test_runs | self.val_runs)
        val_mask = valid_df[run_col].isin(self.val_runs)
        test_mask = valid_df[run_col].isin(self.test_runs)

        return (
            valid_df[train_mask].copy(),
            valid_df[val_mask].copy(),
            valid_df[test_mask].copy()
        )


class SpatialHeldOutSplitter:
    """
    Holds out entire physical anomaly locations and track sectors.
    Dynamically partitions both anomalous and normal track windows based on
    proximity to anomaly clusters, eliminating arbitrary latitude hardcoding.
    """
    def __init__(
        self,
        test_anomaly_ids: List[int] = [7, 8, 9],
        val_anomaly_ids: List[int] = [6],
        train_anomaly_ids: Optional[List[int]] = None
    ):
        self.test_anomaly_ids = set(test_anomaly_ids)
        self.val_anomaly_ids = set(val_anomaly_ids)
        if train_anomaly_ids is None:
            all_ids = set(a['id'] for a in GROUND_TRUTH_ANOMALIES)
            self.train_anomaly_ids = all_ids - self.test_anomaly_ids - self.val_anomaly_ids
        else:
            self.train_anomaly_ids = set(train_anomaly_ids)

        self.anom_map = {a['id']: a for a in GROUND_TRUTH_ANOMALIES}

    def _min_dist_to_ids(self, lats: np.ndarray, lons: np.ndarray, ids: Set[int]) -> np.ndarray:
        if not ids:
            return np.full(len(lats), np.inf)
        coords = [(self.anom_map[i]['lat'], self.anom_map[i]['lon']) for i in ids if i in self.anom_map]
        if not coords:
            return np.full(len(lats), np.inf)
        dists = [np.array([haversine_distance(lat, lon, c[0], c[1]) for lat, lon in zip(lats, lons)])
                 for c in coords]
        return np.min(dists, axis=0)

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        valid_df = df[df['label'] != -1].copy()
        lats = valid_df['latitude'].values if 'latitude' in valid_df.columns else valid_df['Latitude'].values
        lons = valid_df['longitude'].values if 'longitude' in valid_df.columns else valid_df['Longitude'].values

        # Anomaly windows partitioned strictly by anomaly ID
        test_anom_mask = (valid_df['label'] == 1) & (valid_df['nearest_anomaly_id'].isin(self.test_anomaly_ids))
        val_anom_mask = (valid_df['label'] == 1) & (valid_df['nearest_anomaly_id'].isin(self.val_anomaly_ids))
        train_anom_mask = (valid_df['label'] == 1) & (valid_df['nearest_anomaly_id'].isin(self.train_anomaly_ids))

        # Dynamic Voronoi partitioning for normal track (label == 0)
        norm_mask = (valid_df['label'] == 0)
        dist_to_train = self._min_dist_to_ids(lats, lons, self.train_anomaly_ids)
        dist_to_val = self._min_dist_to_ids(lats, lons, self.val_anomaly_ids)
        dist_to_test = self._min_dist_to_ids(lats, lons, self.test_anomaly_ids)

        # Assign normal window to nearest anomaly cluster
        test_norm_mask = norm_mask & (dist_to_test <= dist_to_train) & (dist_to_test <= dist_to_val)
        val_norm_mask = norm_mask & (dist_to_val < dist_to_test) & (dist_to_val <= dist_to_train)
        train_norm_mask = norm_mask & (dist_to_train < dist_to_test) & (dist_to_train < dist_to_val)

        train_df = valid_df[train_anom_mask | train_norm_mask].copy()
        val_df = valid_df[val_anom_mask | val_norm_mask].copy()
        test_df = valid_df[test_anom_mask | test_norm_mask].copy()

        return train_df, val_df, test_df


class CombinedRunAndSpatialSplitter:
    """
    Combined holdout: Unseen runs AND unseen physical anomaly sites.
    Provides the most rigorous real-world deployment benchmark.
    """
    def __init__(
        self,
        test_runs: List[str] = ['AtoB_20_4', 'AtoB_22.5_4', 'BtoA_25_4', 'BtoA_27.5_4'],
        val_runs: List[str] = ['AtoB_20_3', 'BtoA_22.5_3', 'BtoA_27.5_3'],
        test_anomaly_ids: List[int] = [7, 8, 9],
        val_anomaly_ids: List[int] = [6]
    ):
        self.spatial_splitter = SpatialHeldOutSplitter(test_anomaly_ids, val_anomaly_ids)
        self.run_splitter = RunHeldOutSplitter(test_runs, val_runs)

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_sp, val_sp, test_sp = self.spatial_splitter.split(df)
        run_col = 'run_id' if 'run_id' in df.columns else 'run'

        train_df = train_sp[~train_sp[run_col].isin(self.run_splitter.test_runs | self.run_splitter.val_runs)].copy()
        val_df = val_sp[val_sp[run_col].isin(self.run_splitter.val_runs)].copy()
        test_df = test_sp[test_sp[run_col].isin(self.run_splitter.test_runs)].copy()

        return train_df, val_df, test_df
