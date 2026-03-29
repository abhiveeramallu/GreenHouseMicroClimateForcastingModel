"""
Project: Microclimate Forecasting System using Machine Learning
Purpose: Train coordinated forecasting models for greenhouse temperature prediction.
Module Group: Machine Learning Forecasting Module
DFD Connection: Consumes sequence-ready tensors from Data Management Module and
outputs primary deep-learning predictions to Hybrid Coordination.
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


def build_gru_model(input_shape: tuple):
    """
    Build a GRU model for time-series temperature forecasting.

    Input:
    - input_shape: (sequence_length, feature_count)
    """

    import torch.nn as nn

    class GreenhouseGRU(nn.Module):
        def __init__(self, input_size: int, hidden_size: int = 48, num_layers: int = 1):
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            out, _ = self.gru(x)
            return self.head(out[:, -1, :])

    return GreenhouseGRU(input_size=input_shape[1])


def build_bilstm_model(input_shape: tuple):
    """
    Build a BiLSTM model for time-series temperature forecasting.

    Input:
    - input_shape: (sequence_length, feature_count)
    """

    import torch.nn as nn

    class GreenhouseBiLSTM(nn.Module):
        def __init__(self, input_size: int, hidden_size: int = 40, num_layers: int = 1):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.0,
                bidirectional=True,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_size * 2, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    return GreenhouseBiLSTM(input_size=input_shape[1])


def train_sequence_model(
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 30,
    batch_size: int = 16,
):
    """
    Train a sequence model (GRU/BiLSTM) using preprocessed windows.

    Output:
    - history dict with loss and val_loss
    """

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    x_train_t = torch.FloatTensor(x_train).to(device)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    x_val_t = torch.FloatTensor(x_val).to(device)
    y_val_t = torch.FloatTensor(y_val).reshape(-1, 1).to(device)

    train_dataset = TensorDataset(x_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history = {"loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for _epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += float(loss.item())

        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val_t)
            val_loss = float(criterion(val_outputs, y_val_t).item())

        train_loss /= max(1, len(train_loader))
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def _train_backend(
    backend: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    model_dir: Path,
    model_tag: str,
    epochs: int,
    batch_size: int,
) -> TrainedForecaster:
    import torch

    input_shape = (x_train.shape[1], x_train.shape[2])
    if backend == "gru":
        model = build_gru_model(input_shape=input_shape)
    elif backend == "bilstm":
        model = build_bilstm_model(input_shape=input_shape)
    else:
        raise ValueError(f"Unsupported backend '{backend}'.")

    history = train_sequence_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / f"{model_tag}_{backend}.pt")
    return TrainedForecaster(backend=backend, model=model, history={k: list(v) for k, v in history.items()})


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
    Train primary deep sequence model.

    Supported backends:
    - auto (tries GRU then BiLSTM)
    - gru
    - bilstm
    """

    if force_backend not in {"auto", "gru", "bilstm"}:
        raise ValueError("force_backend must be one of: auto, gru, bilstm")

    save_dir = Path(model_dir)
    backend_order = ["gru", "bilstm"] if force_backend == "auto" else [force_backend]
    failures: list[str] = []

    for idx, backend in enumerate(backend_order):
        try:
            forecaster = _train_backend(
                backend=backend,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                model_dir=save_dir,
                model_tag=model_tag,
                epochs=epochs,
                batch_size=batch_size,
            )
            if idx > 0 and failures:
                forecaster.fallback_reason = f"Primary deep model fallback. Earlier failures: {' | '.join(failures)}"
            return forecaster
        except Exception as exc:
            failures.append(f"{backend}:{exc}")

    raise RuntimeError(f"Unable to train deep forecasting backend. Details: {' | '.join(failures)}")


def predict_temperature(forecaster: TrainedForecaster, x_input: np.ndarray) -> np.ndarray:
    """
    Run inference for trained GRU/BiLSTM forecaster.
    """

    if forecaster.backend not in {"gru", "bilstm"}:
        raise ValueError(f"Unsupported backend '{forecaster.backend}'.")

    import torch

    forecaster.model.eval()
    with torch.no_grad():
        x_tensor = torch.FloatTensor(x_input)
        predictions = forecaster.model(x_tensor).numpy().reshape(-1)
    return predictions.astype(np.float32)
