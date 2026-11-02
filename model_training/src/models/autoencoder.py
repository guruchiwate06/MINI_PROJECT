"""
PyTorch Reconstruction Autoencoder for Vibration Feature Anomaly Detection

Provides leak-free threshold calibration on validation or training data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from typing import Optional


class SignalReconstructionAutoencoder(nn.Module):
    """
    MLP Autoencoder trained exclusively on healthy/normal track feature vectors.
    Anomalies are detected via high reconstruction mean squared error.
    """
    def __init__(self, input_dim: int, latent_dim: int = 12):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, latent_dim),
            nn.LeakyReLU(0.2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


class SignalAutoencoderDetector:
    def __init__(self, latent_dim: int = 12, epochs: int = 15, lr: float = 1e-3, batch_size: int = 64, seed: int = 42):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.scaler = StandardScaler()
        self.model = None
        self.threshold_ = 0.0

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Train exclusively on normal instances (y == 0)
        if y_train is not None and np.any(y_train == 0):
            X_fit = X_train[y_train == 0]
        else:
            X_fit = X_train

        X_scaled = self.scaler.fit_transform(X_fit)
        input_dim = X_scaled.shape[1]
        self.model = SignalReconstructionAutoencoder(input_dim, self.latent_dim)
        self.model.train()

        tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(tensor_x)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        for epoch in range(self.epochs):
            for (batch_x,) in loader:
                optimizer.zero_grad()
                recon = self.model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()

        # Set default threshold on training normal reconstructions (e.g. 95th percentile)
        train_recon_errors = self.score_samples(X_fit)
        self.threshold_ = float(np.percentile(train_recon_errors, 95.0))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Computes reconstruction error (MSE per sample) as anomaly score."""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            recon = self.model(tensor_x)
            mse = torch.mean((recon - tensor_x) ** 2, dim=1).cpu().numpy()
        return mse

    def calibrate_threshold(self, X_val: np.ndarray, y_val: Optional[np.ndarray] = None) -> float:
        """Calibrates threshold on validation data without touching test data."""
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
            self.threshold_ = float(np.percentile(val_scores, 95.0))
        return self.threshold_

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score_samples(X)
        return (scores >= self.threshold_).astype(int)
