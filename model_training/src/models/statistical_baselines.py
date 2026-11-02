"""
Statistical Feature-Based Anomaly Detectors and Classifiers for Rail-VIVID

Provides leak-free threshold calibration using validation or normal training data only.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from typing import Dict, Any, Tuple, Optional


class UnsupervisedIsolationForest:
    """
    Unsupervised statistical anomaly detector using Isolation Forest.
    Trained strictly on normal training data (or unlabelled training mix).
    Threshold is calibrated strictly on validation or training data.
    """
    def __init__(self, n_estimators: int = 150, contamination: float = 0.05, seed: int = 42):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=seed,
            n_jobs=-1
        )
        self.threshold_ = 0.0

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None):
        # If labels provided, fit solely on normal class (y == 0)
        if y_train is not None and np.any(y_train == 0):
            X_fit = X_train[y_train == 0]
        else:
            X_fit = X_train
        X_scaled = self.scaler.fit_transform(X_fit)
        self.model.fit(X_scaled)
        # Default threshold based on training set scores (e.g. 95th percentile of normal train scores)
        train_scores = self.score_samples(X_fit)
        self.threshold_ = float(np.percentile(train_scores, 100.0 * (1.0 - self.contamination)))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Returns anomaly score (higher score = more anomalous)."""
        X_scaled = self.scaler.transform(X)
        return -self.model.decision_function(X_scaled)

    def calibrate_threshold(self, X_val: np.ndarray, y_val: Optional[np.ndarray] = None) -> float:
        """
        Calibrates decision threshold on validation data without touching test data.
        If validation labels exist with at least one positive, maximizes validation F1.
        Otherwise, uses the (1 - contamination) percentile of validation scores.
        """
        val_scores = self.score_samples(X_val)
        if y_val is not None and np.sum(y_val == 1) > 0 and np.sum(y_val == 0) > 0:
            candidate_thresholds = np.percentile(val_scores, np.linspace(70, 99, 30))
            best_f1 = -1.0
            best_thresh = self.threshold_
            for thresh in candidate_thresholds:
                preds = (val_scores >= thresh).astype(int)
                score = f1_score(y_val, preds, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_thresh = float(thresh)
            self.threshold_ = best_thresh
        else:
            self.threshold_ = float(np.percentile(val_scores, 100.0 * (1.0 - self.contamination)))
        return self.threshold_

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score_samples(X)
        return (scores >= self.threshold_).astype(int)


class SupervisedRandomForest:
    """
    Supervised Random Forest baseline with validation-based threshold optimization.
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
        self.threshold_ = 0.5

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.threshold_ = 0.5
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        # Handle cases where model only learned 1 class
        if len(self.model.classes_) == 1:
            if self.model.classes_[0] == 1:
                return np.ones(len(X))
            else:
                return np.zeros(len(X))
        return self.model.predict_proba(X_scaled)[:, 1]

    def calibrate_threshold(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Optimizes classification threshold on validation set to maximize F1."""
        val_probs = self.predict_proba(X_val)
        if np.sum(y_val == 1) > 0 and np.sum(y_val == 0) > 0:
            best_f1 = -1.0
            best_thresh = 0.5
            for thresh in np.linspace(0.1, 0.9, 33):
                preds = (val_probs >= thresh).astype(int)
                score = f1_score(y_val, preds, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_thresh = float(thresh)
            self.threshold_ = best_thresh
        else:
            self.threshold_ = 0.5
        return self.threshold_

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= self.threshold_).astype(int)


class SupervisedLogisticRegression:
    """
    L2-regularized Logistic Regression baseline with validation-based threshold calibration.
    """
    def __init__(self, C: float = 1.0, seed: int = 42):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=C,
            class_weight='balanced',
            max_iter=1000,
            random_state=seed
        )
        self.threshold_ = 0.5

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.threshold_ = 0.5
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        if len(self.model.classes_) == 1:
            return np.zeros(len(X)) if self.model.classes_[0] == 0 else np.ones(len(X))
        return self.model.predict_proba(X_scaled)[:, 1]

    def calibrate_threshold(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        val_probs = self.predict_proba(X_val)
        if np.sum(y_val == 1) > 0 and np.sum(y_val == 0) > 0:
            best_f1 = -1.0
            best_thresh = 0.5
            for thresh in np.linspace(0.1, 0.9, 33):
                preds = (val_probs >= thresh).astype(int)
                score = f1_score(y_val, preds, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_thresh = float(thresh)
            self.threshold_ = best_thresh
        else:
            self.threshold_ = 0.5
        return self.threshold_

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= self.threshold_).astype(int)
