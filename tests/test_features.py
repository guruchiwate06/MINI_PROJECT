import pytest
import numpy as np
from src.features.extractor import compute_window_features

def test_compute_window_features():
    # Generate 1000 samples of a synthetic sine wave + noise
    fs = 2000.0
    t = np.arange(1000) / fs
    signal_data = np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(1000)

    feats = compute_window_features(signal_data, fs=fs)

    # Check that all expected keys exist
    expected_keys = [
        'mean', 'std', 'rms', 'peak_to_peak', 'skewness', 'kurtosis', 'crest_factor',
        'band_energy_low', 'band_energy_mid', 'band_energy_high', 'spectral_centroid'
    ]
    for k in expected_keys:
        assert k in feats
        assert np.isfinite(feats[k])

    # Basic physical sanity checks
    assert feats['rms'] > 0
    assert feats['peak_to_peak'] > 0
    assert feats['crest_factor'] >= 1.0
    assert 0 <= feats['band_energy_low'] <= 1.0
    assert 0 <= feats['band_energy_mid'] <= 1.0
    assert 0 <= feats['band_energy_high'] <= 1.0

