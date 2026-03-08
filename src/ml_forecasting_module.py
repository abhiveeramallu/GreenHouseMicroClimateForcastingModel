"""
Project: Microclimate Forecasting System using Machine Learning
Purpose: Train coordinated forecasting models for greenhouse temperature prediction.
Module Group: Machine Learning Forecasting Module
DFD Connection: Consumes LSTM-ready sequences from Data Management Module and
outputs coordinated predictions to Decision & Control Simulation Module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class TrainedForecaster:
    """Container for trained model artifacts and metadata."""

    backend: str
    model: Any
    history: Dict[str, list]
    fallback_reason: Optional[str] = None


class LinearSequenceRegressor:
    """
    Lightweight autoregressive baseline using least squares on flattened sequences.

    This serves as:
    1) a strong interpretable baseline against LSTM
    2) a guaranteed fallback when TensorFlow is unavailable
    """

    def __init__(self, l2_penalty: float = 1e-3) -> None:
        self.l2_penalty = l2_penalty
        self.weights: Optional[np.ndarray] = None
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_scale: Optional[np.ndarray] = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        x_matrix = x_train.reshape(x_train.shape[0], -1).astype(np.float64)
        x_matrix = np.nan_to_num(x_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        y_vector = np.nan_to_num(y_train.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

        self.feature_mean = x_matrix.mean(axis=0, keepdims=True)
        self.feature_scale = x_matrix.std(axis=0, keepdims=True)
        self.feature_scale[self.feature_scale < 1e-8] = 1.0

        normalized = (x_matrix - self.feature_mean) / self.feature_scale
        ones = np.ones((x_matrix.shape[0], 1), dtype=np.float32)
        design = np.concatenate([normalized, ones], axis=1)

        gram = design.T @ design
        identity = np.eye(gram.shape[0], dtype=np.float64)
        regularized = gram + self.l2_penalty * identity
        try:
            self.weights = np.linalg.pinv(regularized) @ design.T @ y_vector
        except np.linalg.LinAlgError:
            self.weights = np.linalg.lstsq(regularized, design.T @ y_vector, rcond=1e-6)[0]

    def predict(self, x_input: np.ndarray) -> np.ndarray:
        if self.weights is None or self.feature_mean is None or self.feature_scale is None:
            raise ValueError("LinearSequenceRegressor is not fitted.")
        x_matrix = x_input.reshape(x_input.shape[0], -1).astype(np.float64)
        x_matrix = np.nan_to_num(x_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        normalized = (x_matrix - self.feature_mean) / self.feature_scale
        ones = np.ones((x_matrix.shape[0], 1), dtype=np.float32)
        design = np.concatenate([normalized, ones], axis=1)
        return design @ self.weights

    def save(self, output_path: str | Path) -> None:
        if self.weights is None:
            raise ValueError("Cannot save unfitted LinearSequenceRegressor.")
        path_obj = Path(output_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        np.save(path_obj, self.weights)


def build_lstm_model(input_shape: tuple, learning_rate: float = 1e-3):
    """
    Build and compile an LSTM model using PyTorch for greenhouse temperature forecasting.

    Planned input:
    - input_shape: (sequence_length, feature_count)

    Output:
    - PyTorch LSTM model class
    """
    import torch
    import torch.nn as nn

    class GreenhouseLSTM(nn.Module):
        def __init__(self, input_size, hidden_size=32, num_layers=1):
            super(GreenhouseLSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.0)
            self.fc = nn.Linear(hidden_size, 1)
            
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out
    
    return GreenhouseLSTM(input_shape[1], hidden_size=32)


def train_lstm_model(
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 30,
    batch_size: int = 16,
):
    """
    Train the PyTorch LSTM model using preprocessed sequences.

    Inputs:
    - model: PyTorch model
    - x_train, y_train: training sequences and targets
    - x_val, y_val: validation sequences and targets
    - epochs: max training epochs
    - batch_size: mini-batch size

    Output:
    - Training history dictionary
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Convert to tensors
    x_train_t = torch.FloatTensor(x_train).to(device)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    x_val_t = torch.FloatTensor(x_val).to(device)
    y_val_t = torch.FloatTensor(y_val).reshape(-1, 1).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(x_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    history = {'loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
        
        train_loss /= len(train_loader)
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break
    
    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return history


def train_temperature_forecaster(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    model_dir: str | Path,
    model_tag: str,
    epochs: int = 30,
    batch_size: int = 16,
    force_backend: str = "auto",
) -> TrainedForecaster:
    """
    Train primary LSTM model, and fallback to linear baseline if unavailable.

    Inputs:
    - x_train, y_train, x_val, y_val: split sequence arrays
    - model_dir: directory for saved model artifacts
    - model_tag: prefix for saved files
    - epochs, batch_size: training hyperparameters
    - force_backend: "auto", "lstm", or "linear"

    Output:
    - TrainedForecaster with backend metadata and model object
    """

    if force_backend not in {"auto", "lstm", "linear"}:
        raise ValueError("force_backend must be one of: auto, lstm, linear")

    save_dir = Path(model_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if force_backend in {"auto", "lstm"}:
        try:
            import torch
            lstm_model = build_lstm_model(input_shape=(x_train.shape[1], x_train.shape[2]))
            history = train_lstm_model(
                model=lstm_model,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                epochs=epochs,
                batch_size=batch_size,
            )
            # Save PyTorch model
            torch.save(lstm_model.state_dict(), save_dir / f"{model_tag}_lstm.pt")
            return TrainedForecaster(
                backend="lstm",
                model=lstm_model,
                history={key: list(value) for key, value in history.items()},
            )
        except Exception as exc:
            if force_backend == "lstm":
                raise
            fallback_reason = f"LSTM unavailable, switched to linear baseline: {exc}"
    else:
        fallback_reason = "Forced linear backend."

    linear_model = LinearSequenceRegressor(l2_penalty=1e-3)
    linear_model.fit(x_train=x_train, y_train=y_train)
    linear_model.save(save_dir / f"{model_tag}_linear_weights.npy")

    validation_predictions = linear_model.predict(x_val)
    val_mae = float(np.mean(np.abs(validation_predictions - y_val)))

    return TrainedForecaster(
        backend="linear",
        model=linear_model,
        history={"val_mae": [val_mae]},
        fallback_reason=fallback_reason,
    )


def predict_temperature(forecaster: TrainedForecaster, x_input: np.ndarray) -> np.ndarray:
    """
    Run inference for the trained model backend.

    Inputs:
    - forecaster: TrainedForecaster object
    - x_input: sequence input array

    Output:
    - 1D prediction array
    """

    if forecaster.backend == "lstm":
        import torch
        forecaster.model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_input)
            predictions = forecaster.model(x_tensor).numpy().reshape(-1)
        return predictions.astype(np.float32)
    if forecaster.backend == "linear":
        predictions = forecaster.model.predict(x_input)
        return predictions.astype(np.float32)
    raise ValueError(f"Unsupported backend '{forecaster.backend}'.")


def blend_predictions(
    lstm_predictions: Optional[np.ndarray],
    linear_predictions: np.ndarray,
    lstm_weight: float = 0.7,
) -> np.ndarray:
    """
    Blend LSTM and linear predictions to stabilize output.

    Inputs:
    - lstm_predictions: predictions from LSTM or None
    - linear_predictions: predictions from linear baseline
    - lstm_weight: ensemble weight for LSTM when available

    Output:
    - blended prediction array
    """

    if lstm_predictions is None:
        return linear_predictions.astype(np.float32)

    alpha = float(np.clip(lstm_weight, 0.0, 1.0))
    beta = 1.0 - alpha
    return (alpha * lstm_predictions + beta * linear_predictions).astype(np.float32)
