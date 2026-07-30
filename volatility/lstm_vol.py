"""LSTM-based volatility prediction with sentiment features.

Implements recurrent neural network models for volatility forecasting,
with optional sentiment features as exogenous inputs.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LSTMVolatility:
    """LSTM neural network for volatility prediction.

    Builds and trains LSTM models using historical volatility and
    optional sentiment features for forecasting.

    Uses PyTorch for implementation with a clean training API.
    """

    def __init__(
        self,
        sequence_length: int = 60,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._feature_scaler = None
        self._target_scaler = None
        self._train_losses = []
        self._val_losses = []

    def _build_model(self, input_size: int):
        """Build LSTM model architecture."""
        import torch
        import torch.nn as nn

        class VolatilityLSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True,
                )
                self.dropout = nn.Dropout(dropout)
                self.linear = nn.Linear(hidden_size, 1)

            def forward(self, x):
                lstm_out, (h_n, c_n) = self.lstm(x)
                # Use the last hidden state
                out = self.dropout(lstm_out[:, -1, :])
                out = self.linear(out)
                return out.squeeze()

        self._model = VolatilityLSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

    def _prepare_sequences(
        self, features: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequence windows for LSTM training."""
        X, y = [], []
        for i in range(len(features) - self.sequence_length):
            X.append(features[i : i + self.sequence_length])
            y.append(target[i + self.sequence_length])
        return np.array(X), np.array(y)

    def _scale_data(
        self, features: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features and target to zero mean, unit variance."""
        from sklearn.preprocessing import StandardScaler

        self._feature_scaler = StandardScaler()
        features_scaled = self._feature_scaler.fit_transform(features)

        self._target_scaler = StandardScaler()
        target_scaled = self._target_scaler.fit_transform(target.reshape(-1, 1)).ravel()

        return features_scaled, target_scaled

    def fit(
        self,
        volatility_series: pd.Series,
        sentiment_series: Optional[pd.Series] = None,
        validation_split: float = 0.2,
        verbose: bool = True,
    ) -> dict:
        """Train the LSTM model on volatility data.

        Args:
            volatility_series: Historical volatility values (target).
            sentiment_series: Optional sentiment scores (additional feature).
            validation_split: Fraction of data for validation.
            verbose: Print training progress.

        Returns:
            Dict with training history and model info.
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        # Prepare feature matrix
        features = []
        features.append(volatility_series.values[:-1])  # Lagged vol
        features.append(
            np.gradient(volatility_series.values)  # Vol change
        )
        features.append(
            volatility_series.rolling(5).mean().values[:-1]  # Short MA
        )
        features.append(
            volatility_series.rolling(21).mean().values[:-1]  # Long MA
        )

        if sentiment_series is not None:
            # Align sentiment with vol index
            aligned = pd.concat(
                [volatility_series, sentiment_series], axis=1
            ).dropna()
            sentiment_aligned = aligned.iloc[:, 1].values
            features.append(sentiment_aligned)

        features = np.column_stack(features)
        target = volatility_series.values[1:]  # Next period vol

        # Remove NaN rows
        valid = ~np.isnan(features).any(axis=1) & ~np.isnan(target)
        features, target = features[valid], target[valid]

        # Scale
        features_scaled, target_scaled = self._scale_data(features, target)

        # Create sequences
        X, y = self._prepare_sequences(features_scaled, target_scaled)
        input_size = features_scaled.shape[1]

        # Build model
        self._build_model(input_size)

        # Train/val split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val)
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self._model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )

        self._train_losses = []
        self._val_losses = []

        for epoch in range(self.epochs):
            # Training
            self._model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                y_pred = self._model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            self._train_losses.append(avg_train_loss)

            # Validation
            self._model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    y_pred = self._model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            self._val_losses.append(avg_val_loss)

            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            if verbose and (epoch + 1) % 20 == 0:
                logger.info(
                    "Epoch %d/%d - Train Loss: %.6f, Val Loss: %.6f, LR: %.6f",
                    epoch + 1,
                    self.epochs,
                    avg_train_loss,
                    avg_val_loss,
                    current_lr,
                )

        return {
            "train_losses": self._train_losses,
            "val_losses": self._val_losses,
            "final_train_loss": self._train_losses[-1] if self._train_losses else None,
            "final_val_loss": self._val_losses[-1] if self._val_losses else None,
            "input_size": input_size,
            "n_train_samples": len(X_train),
            "n_val_samples": len(X_val),
        }

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate volatility predictions.

        Args:
            features: Feature array with same shape as training data.

        Returns:
            Array of predicted volatility values (original scale).
        """
        import torch

        if self._model is None or self._feature_scaler is None:
            raise RuntimeError("Model must be trained before prediction.")

        self._model.eval()
        features_scaled = self._feature_scaler.transform(features)
        sequences = []
        for i in range(len(features_scaled) - self.sequence_length + 1):
            sequences.append(features_scaled[i : i + self.sequence_length])
        X = np.array(sequences)

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self._model(X_tensor).cpu().numpy()

        if self._target_scaler is not None:
            predictions = self._target_scaler.inverse_transform(
                predictions.reshape(-1, 1)
            ).ravel()

        return predictions

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Evaluate prediction performance.

        Args:
            y_true: Actual volatility values.
            y_pred: Predicted volatility values.

        Returns:
            Dict with RMSE, MAE, MAPE, and R2 metrics.
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        r2 = r2_score(y_true, y_pred)

        return {
            "rmse": rmse,
            "mse": mse,
            "mae": mae,
            "mape": mape,
            "r2": r2,
        }

    def save(self, path: str):
        """Save model weights and scalers."""
        import torch

        if self._model is None:
            raise RuntimeError("No model to save.")
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "feature_scaler": self._feature_scaler,
                "target_scaler": self._target_scaler,
                "sequence_length": self.sequence_length,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "input_size": self._model.lstm.input_size,
            },
            path,
        )

    def load(self, path: str):
        """Load model weights and scalers."""
        import torch

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.sequence_length = checkpoint["sequence_length"]
        self.hidden_size = checkpoint["hidden_size"]
        self.num_layers = checkpoint["num_layers"]
        self._feature_scaler = checkpoint["feature_scaler"]
        self._target_scaler = checkpoint["target_scaler"]
        self._build_model(checkpoint["input_size"])
        self._model.load_state_dict(checkpoint["model_state_dict"])
