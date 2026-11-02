"""
Feature Extraction Pipeline for Rail-VIVID Vibration Time-Series

Extracts windowed time-domain and frequency-domain features from 6-channel
accelerometer records with GPS spatial grounding and boundary buffering.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal
from typing import Dict, List, Tuple, Optional
from src.evaluation.splitters import assign_spatial_labels


def compute_window_features(window_data: np.ndarray, fs: float = 2000.0) -> Dict[str, float]:
    """
    Computes statistical and spectral features for a single 1D signal window.
    Features:
      - mean, std, rms, peak_to_peak, skewness, kurtosis, crest_factor
      - band_energy_low (0-50 Hz), band_energy_mid (50-200 Hz), band_energy_high (200-500 Hz)
      - spectral_centroid
    """
    features = {}
    mean_val = np.mean(window_data)
    std_val = np.std(window_data)
    rms_val = np.sqrt(np.mean(window_data ** 2))
    peak_val = np.max(np.abs(window_data))
    ptp_val = np.ptp(window_data)

    features['mean'] = float(mean_val)
    features['std'] = float(std_val)
    features['rms'] = float(rms_val)
    features['peak_to_peak'] = float(ptp_val)
    features['skewness'] = float(stats.skew(window_data))
    features['kurtosis'] = float(stats.kurtosis(window_data))
    features['crest_factor'] = float(peak_val / (rms_val + 1e-8))

    # Frequency-domain features via FFT / Welch
    freqs, psd = signal.welch(window_data, fs=fs, nperseg=min(len(window_data), 256))
    total_energy = np.sum(psd) + 1e-12

    # Band energies
    low_mask = (freqs >= 0) & (freqs < 50)
    mid_mask = (freqs >= 50) & (freqs < 200)
    high_mask = (freqs >= 200) & (freqs <= 500)

    features['band_energy_low'] = float(np.sum(psd[low_mask]) / total_energy)
    features['band_energy_mid'] = float(np.sum(psd[mid_mask]) / total_energy)
    features['band_energy_high'] = float(np.sum(psd[high_mask]) / total_energy)

    # Spectral centroid
    features['spectral_centroid'] = float(np.sum(freqs * psd) / total_energy)

    return features


def extract_features_from_run(
    csv_path: str,
    run_id: str,
    window_size: int = 1000,  # 0.5 seconds at 2 kHz
    stride: int = 500,        # 0.25 seconds (50% overlap)
    channels: List[str] = ['Channel1', 'Channel2', 'Channel3', 'Channel4', 'Channel5', 'Channel6'],
    fs: float = 2000.0,
    subsample_stride: Optional[int] = None
) -> pd.DataFrame:
    """
    Processes a single run CSV into a structured feature DataFrame with window-level features.
    """
    usecols = ['Timestamp', 'Latitude', 'Longitude', 'Temperature', 'Humidity'] + channels
    df_raw = pd.read_csv(csv_path, usecols=usecols)

    if subsample_stride is not None and subsample_stride > 1:
        df_raw = df_raw.iloc[::subsample_stride].reset_index(drop=True)
        effective_fs = fs / subsample_stride
        effective_win = window_size // subsample_stride
        effective_stride = stride // subsample_stride
    else:
        effective_fs = fs
        effective_win = window_size
        effective_stride = stride

    n_samples = len(df_raw)
    records = []

    for start_idx in range(0, n_samples - effective_win + 1, effective_stride):
        end_idx = start_idx + effective_win
        mid_idx = start_idx + effective_win // 2

        win_meta = {
            'run_id': run_id,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'timestamp': df_raw['Timestamp'].iloc[mid_idx],
            'latitude': df_raw['Latitude'].iloc[mid_idx],
            'longitude': df_raw['Longitude'].iloc[mid_idx],
            'temperature': df_raw['Temperature'].iloc[mid_idx],
            'humidity': df_raw['Humidity'].iloc[mid_idx],
        }

        # Compute features for each channel
        for ch in channels:
            ch_data = df_raw[ch].iloc[start_idx:end_idx].values
            ch_feats = compute_window_features(ch_data, fs=effective_fs)
            for feat_name, feat_val in ch_feats.items():
                win_meta[f'{ch}_{feat_name}'] = feat_val

        records.append(win_meta)

    df_feats = pd.DataFrame(records)
    df_labeled = assign_spatial_labels(df_feats, anomaly_radius_m=6.0, buffer_radius_m=18.0)
    return df_labeled

