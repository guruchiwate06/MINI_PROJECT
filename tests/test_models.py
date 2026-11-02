import pytest
import numpy as np
from src.models.statistical_baselines import (
    UnsupervisedIsolationForest,
    SupervisedRandomForest,
    SupervisedLogisticRegression
)
from src.models.autoencoder import SignalAutoencoderDetector
from src.models.cnn1d import CNN1DClassifier


def test_statistical_models():
    # 100 samples, 10 features
    X = np.random.randn(100, 10)
    y = np.random.choice([0, 1], size=100, p=[0.8, 0.2])

    # Isolation Forest
    iso = UnsupervisedIsolationForest(n_estimators=10)
    iso.fit(X, y)
    scores = iso.score_samples(X)
    assert len(scores) == 100
    assert np.all(np.isfinite(scores))

    # Random Forest
    rf = SupervisedRandomForest(n_estimators=10, max_depth=3)
    rf.fit(X, y)
    probs = rf.predict_proba(X)
    assert len(probs) == 100
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    # Logistic Regression
    lr = SupervisedLogisticRegression()
    lr.fit(X, y)
    lr_probs = lr.predict_proba(X)
    assert len(lr_probs) == 100


def test_deep_models():
    # 40 samples, 10 features
    X = np.random.randn(40, 10)
    y = np.random.choice([0, 1], size=40, p=[0.75, 0.25])

    # Signal Autoencoder
    ae = SignalAutoencoderDetector(latent_dim=4, epochs=2, batch_size=16)
    ae.fit(X, y)
    recon_scores = ae.score_samples(X)
    assert len(recon_scores) == 40
    assert np.all(recon_scores >= 0)

    # 1D CNN: 20 samples, 6 channels, 100 time points
    X_cnn = np.random.randn(20, 6, 100)
    y_cnn = np.random.choice([0, 1], size=20)
    cnn = CNN1DClassifier(in_channels=6, epochs=2, batch_size=10)
    cnn.fit(X_cnn, y_cnn)
    cnn_probs = cnn.predict_proba(X_cnn)
    assert len(cnn_probs) == 20
    assert np.all((cnn_probs >= 0.0) & (cnn_probs <= 1.0))

