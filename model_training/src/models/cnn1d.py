"""
PyTorch 1D Temporal Convolutional Neural Network for Raw Multichannel Vibration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional


class Vibration1DCNN(nn.Module):
    """
    Lightweight 1D CNN processing raw multichannel vibration windows (B x channels x length).
    Optimized for rapid CPU training.
    """
    def __init__(self, in_channels: int = 6, num_classes: int = 2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),

            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),

            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_block(x)
        feat = feat.view(feat.size(0), -1)
        logits = self.classifier(feat)
        return logits


class CNN1DClassifier:
    def __init__(self, in_channels: int = 6, epochs: int = 10, lr: float = 1e-3, batch_size: int = 64, seed: int = 42):
        self.in_channels = in_channels
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.model = Vibration1DCNN(in_channels=self.in_channels)
        self.model.train()

        tensor_x = torch.tensor(X_train, dtype=torch.float32)
        tensor_y = torch.tensor(y_train, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Handle class imbalance via weighted cross-entropy
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        weight_pos = neg_count / max(pos_count, 1)
        weights = torch.tensor([1.0, float(weight_pos)], dtype=torch.float32)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(weight=weights)

        for epoch in range(self.epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        tensor_x = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(tensor_x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
