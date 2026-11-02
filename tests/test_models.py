import pytest
import numpy as np
from src.models.statistical_baselines import (
    UnsupervisedIsolationForest,
    SupervisedRandomForest,
    SupervisedLogisticRegression
)
from src.models.autoencoder import SignalAutoencoderDetector
from src.models.cnn1d import CNN1DClassifier


def test_leak_free_threshold_calibration():
    # Synthetic train, val, and test data
    np.random.seed(42)
    X_train = np.random.randn(100, 8)
    y_train = np.random.choice([0, 1], size=100, p=[0.9, 0.1])

    X_val = np.random.randn(40, 8)
    y_val = np.random.choice([0, 1], size=40, p=[0.8, 0.2])

    X_test = np.random.randn(50, 8)
    y_test = np.random.choice([0, 1], size=50, p=[0.8, 0.2])

    # 1. Isolation Forest: threshold calibrated on val, evaluated on test
    iso = UnsupervisedIsolationForest(n_estimators=20)
    iso.fit(X_train, y_train)
    thresh = iso.calibrate_threshold(X_val, y_val)
    preds = iso.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1})
    assert np.isfinite(thresh)

    # 2. Random Forest: threshold tuned on val, evaluated on test
    rf = SupervisedRandomForest(n_estimators=10, max_depth=3)
    rf.fit(X_train, y_train)
    thresh_rf = rf.calibrate_threshold(X_val, y_val)
    preds_rf = rf.predict(X_test)
    assert len(preds_rf) == len(X_test)
    assert 0.0 <= thresh_rf <= 1.0

    # 3. Logistic Regression
    lr = SupervisedLogisticRegression()
    lr.fit(X_train, y_train)
    thresh_lr = lr.calibrate_threshold(X_val, y_val)
    preds_lr = lr.predict(X_test)
    assert len(preds_lr) == len(X_test)

    # 4. Signal Autoencoder: threshold calibrated on val
    ae = SignalAutoencoderDetector(latent_dim=4, epochs=2, batch_size=16)
    ae.fit(X_train, y_train)
    thresh_ae = ae.calibrate_threshold(X_val, y_val)
    preds_ae = ae.predict(X_test)
    assert len(preds_ae) == len(X_test)
    assert thresh_ae > 0
