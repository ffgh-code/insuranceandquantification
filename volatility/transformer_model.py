"""Lightweight single-layer attention Transformer for volatility prediction.

Simplified Transformer using PyTorch with one encoder layer,
designed as a baseline comparison against GARCH and LSTM.
"""

from __future__ import annotations
import logging
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class AttentionTransformer:
    """Single-layer Transformer with multi-head attention for volatility."""

    def __init__(self, seq_len: int = 60, d_model: int = 32, nhead: int = 4,
                 epochs: int = 80, lr: float = 0.001):
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.epochs = epochs
        self.lr = lr
        self._model = None
        self._train_loss = []
        self._val_loss = []

    def _build_model(self, input_dim: int):
        import torch
        import torch.nn as nn
        import math

        class TransformerTS(nn.Module):
            def __init__(self, input_dim, d_model, nhead, seq_len):
                super().__init__()
                self.input_proj = nn.Linear(input_dim, d_model)
                self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.1)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1,
                    dim_feedforward=d_model * 4,
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
                self.output = nn.Linear(d_model, 1)

            def forward(self, x):
                x = self.input_proj(x) + self.pos_encoder[:, :x.size(1), :]
                x = self.transformer(x)
                return self.output(x[:, -1, :]).squeeze()

        self._model = TransformerTS(input_dim, self.d_model, self.nhead, self.seq_len)

    def fit(self, features: np.ndarray, target: np.ndarray,
            validation_split: float = 0.2, verbose: bool = True) -> dict:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        self._build_model(features.shape[-1])
        X, y = self._create_sequences(features, target)

        split = int(len(X) * (1 - validation_split))
        X_tr, X_val = torch.FloatTensor(X[:split]), torch.FloatTensor(X[split:])
        y_tr, y_val = torch.FloatTensor(y[:split]), torch.FloatTensor(y[split:])

        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self._model.parameters(), lr=self.lr)

        for ep in range(self.epochs):
            self._model.train()
            for Xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self._model(Xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
            self._model.eval()
            with torch.no_grad():
                tr_loss = criterion(self._model(X_tr), y_tr).item()
                val_loss = criterion(self._model(X_val), y_val).item()
            self._train_loss.append(tr_loss)
            self._val_loss.append(val_loss)
            if verbose and (ep + 1) % 20 == 0:
                logger.info("Epoch %d/%d train=%.6f val=%.6f", ep + 1, self.epochs, tr_loss, val_loss)

        return {"train_loss": self._train_loss, "val_loss": self._val_loss,
                "final_val_loss": self._val_loss[-1] if self._val_loss else None}

    def _create_sequences(self, features: np.ndarray, target: np.ndarray):
        Xs, ys = [], []
        for i in range(len(features) - self.seq_len):
            Xs.append(features[i:i + self.seq_len])
            ys.append(target[i + self.seq_len])
        return np.array(Xs), np.array(ys)

    def predict(self, features: np.ndarray) -> np.ndarray:
        import torch
        if self._model is None:
            raise RuntimeError("Model not trained")
        self._model.eval()
        X, _ = self._create_sequences(features, np.zeros(len(features)))
        with torch.no_grad():
            return self._model(torch.FloatTensor(X)).numpy()
