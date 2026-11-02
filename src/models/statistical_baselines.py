"""
Statistical Feature-Based Anomaly Detectors and Classifiers for Rail-VIVID
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple, Optional


class UnsupervisedIsolationForest:
    """
    Unsupervised statistical anomaly detector using Isolation Forest.
    Trained strictly on training normal data (or unlabelled training mix).
    """
    def __init__(self, n_estimators: int = 150, contamination: float = 0.05, seed: int = 42):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=seed,
            n_jobs=-1
        )

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None):
        # If labels provided, fit solely on normal class (y == 0) to learn normal track distribution
        if y_train is not None and np.any(y_train == 0):
            X_fit = X_train[y_train == 0]
        else:
            X_fit = X_train
        X_scaled = self.scaler.fit_transform(X_fit)
        self.model.fit(X_scaled)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Returns anomaly score (higher score = more anomalous)."""
        X_scaled = self.scaler.transform(X)
        # decision_function returns negative anomaly score (lower is more anomalous)
        # Negate it so higher score = higher probability of defect
        return -self.model.decision_function(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)  # -1 for anomaly, 1 for inlier
        return (preds == -1).astype(int)


class SupervisedRandomForest:
    """
    Supervised Random Forest baseline for binary defect classification.
    """
    def __init__(self, n_estimators: int = 150, max_depth: int = 12, seed: int = 42):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight='balanced',
            random_state=seed,
            n_jobs=-1
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class SupervisedLogisticRegression:
    """
    L2-regularized Logistic Regression baseline.
    """
    def __init__(self, C: float = 1.0, seed: int = 42):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=C,
            class_weight='balanced',
            max_iter=1000,
            random_state=seed
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

