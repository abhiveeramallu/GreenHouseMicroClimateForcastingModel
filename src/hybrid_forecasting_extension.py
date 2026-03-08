"""
Project: Microclimate Forecasting System using Machine Learning
Purpose: Extend forecasting with parallel secondary ML models and a coordination layer.
Module Group: Machine Learning Forecasting Module (Extension)
DFD Connection: Consumes LSTM-ready sequences from Data Management, coordinates
multiple model outputs, and returns optimized prediction to Decision & Control Simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
import json

import numpy as np
import pandas as pd

from src.ml_forecasting_module import LinearSequenceRegressor
from src.reporting_module import calculate_regression_metrics


@dataclass
class SecondaryModelRegistry:
    """Container for trained secondary models and backend notes."""

    models: Dict[str, Any]
    notes: Dict[str, str]


@dataclass
class CoordinationResult:
    """Container for coordinated hybrid prediction outputs."""

    final_prediction: np.ndarray
    weights: Dict[str, float]
    validation_rmse: Dict[str, float]
    method: str


def _flatten_sequences(x_sequences: np.ndarray) -> np.ndarray:
    return x_sequences.reshape(x_sequences.shape[0], -1).astype(np.float64)


def _safe_array(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def train_secondary_forecasters(
    x_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> SecondaryModelRegistry:
    """
    Train independent secondary models for hybrid forecasting.

    Inputs:
    - x_train: LSTM sequence array (samples, sequence_length, feature_count)
    - y_train: target vector
    - random_state: reproducibility seed for tree models

    Output:
    - SecondaryModelRegistry with trained models and notes
    """

    models: Dict[str, Any] = {}
    notes: Dict[str, str] = {}

    x_flat = _safe_array(_flatten_sequences(x_train))
    y_safe = _safe_array(y_train)

    sklearn_available = True
    try:
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import LinearRegression
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        sklearn_available = False
        notes["sklearn"] = f"Unavailable: {exc}"

    if sklearn_available:
        lr_model = LinearRegression()
        lr_model.fit(x_flat, y_safe)
        models["linear_regression"] = lr_model

        rf_model = RandomForestRegressor(
            n_estimators=240,
            max_depth=14,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
        rf_model.fit(x_flat, y_safe)
        models["random_forest"] = rf_model

        gb_model = GradientBoostingRegressor(
            random_state=random_state,
            n_estimators=260,
            learning_rate=0.04,
            max_depth=3,
            subsample=0.9,
        )
        gb_model.fit(x_flat, y_safe)
        models["gradient_boosting"] = gb_model

        try:  # Optional dependency
            from xgboost import XGBRegressor

            xgb_model = XGBRegressor(
                n_estimators=320,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=1,
            )
            xgb_model.fit(x_flat, y_safe)
            models["xgboost"] = xgb_model
            notes["xgboost"] = "trained"
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            notes["xgboost"] = f"Unavailable: {exc}"
    else:
        # Fallback path keeps pipeline fully executable if sklearn is not present.
        fallback_model = LinearSequenceRegressor(l2_penalty=1e-3)
        fallback_model.fit(x_train=x_train, y_train=y_safe)
        models["linear_sequence_fallback"] = fallback_model

    return SecondaryModelRegistry(models=models, notes=notes)


def predict_secondary_forecasters(
    registry: SecondaryModelRegistry,
    x_input: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Generate predictions for all secondary models.

    Inputs:
    - registry: trained model registry
    - x_input: sequence array

    Output:
    - dictionary {model_name: prediction_array}
    """

    predictions: Dict[str, np.ndarray] = {}
    x_flat = _safe_array(_flatten_sequences(x_input))

    for model_name, model in registry.models.items():
        if isinstance(model, LinearSequenceRegressor):
            pred = model.predict(x_input)
        else:
            pred = model.predict(x_flat)
        predictions[model_name] = np.asarray(pred, dtype=np.float32).reshape(-1)

    return predictions


def coordinate_hybrid_prediction(
    y_validation: np.ndarray,
    validation_predictions: Dict[str, np.ndarray],
    test_predictions: Dict[str, np.ndarray],
    preferred_model: str = "lstm",
) -> CoordinationResult:
    """
    Coordinate multiple models using inverse-validation-error weighted averaging.

    Inputs:
    - y_validation: validation ground truth in scaled domain
    - validation_predictions: per-model validation predictions
    - test_predictions: per-model test predictions
    - preferred_model: model to prioritize slightly when validation errors are close

    Output:
    - CoordinationResult containing final hybrid predictions and weight metadata
    """

    valid_models: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, val_pred in validation_predictions.items():
        if name not in test_predictions:
            continue
        val_arr = np.asarray(val_pred, dtype=np.float64).reshape(-1)
        test_arr = np.asarray(test_predictions[name], dtype=np.float64).reshape(-1)
        # Skip models with NaN predictions
        if np.isnan(val_arr).any() or np.isnan(test_arr).any():
            continue
        if len(val_arr) != len(y_validation) or len(test_arr) == 0:
            continue
        valid_models[name] = (val_arr, test_arr)

    if not valid_models:
        raise ValueError("No valid model predictions available for coordination.")

    y_val = np.asarray(y_validation, dtype=np.float64).reshape(-1)
    validation_rmse: Dict[str, float] = {}
    raw_weights: Dict[str, float] = {}

    for name, (val_pred, _) in valid_models.items():
        rmse = float(np.sqrt(np.mean((y_val - val_pred) ** 2)))
        validation_rmse[name] = rmse
        inv = 1.0 / max(rmse, 1e-6)
        if name == preferred_model:
            inv *= 1.15
        raw_weights[name] = inv

    total_weight = float(sum(raw_weights.values()))
    if total_weight <= 0:
        weights = {name: 1.0 / len(raw_weights) for name in raw_weights}
    else:
        weights = {name: value / total_weight for name, value in raw_weights.items()}

    final_prediction = np.zeros_like(next(iter(valid_models.values()))[1], dtype=np.float64)
    for name, (_, test_pred) in valid_models.items():
        final_prediction += weights[name] * test_pred

    return CoordinationResult(
        final_prediction=final_prediction.astype(np.float32),
        weights=weights,
        validation_rmse=validation_rmse,
        method="dynamic_inverse_rmse_weighted_ensemble",
    )


def build_model_comparison_table(
    y_true: np.ndarray,
    prediction_map: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Build model-comparison table for regression metrics.

    Inputs:
    - y_true: true values in original scale
    - prediction_map: dictionary {model_name: predictions in original scale}

    Output:
    - DataFrame with MAE, RMSE, R2 sorted by RMSE asc
    """

    rows = []
    y_true_arr = np.asarray(y_true, dtype=np.float32).reshape(-1)

    for model_name, predictions in prediction_map.items():
        pred_arr = np.asarray(predictions, dtype=np.float32).reshape(-1)
        if len(pred_arr) != len(y_true_arr):
            continue
        metrics = calculate_regression_metrics(y_true=y_true_arr, y_pred=pred_arr)
        rows.append(
            {
                "model": model_name,
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "r2": float(metrics["r2"]),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["model", "mae", "rmse", "r2"])

    comparison_df = pd.DataFrame(rows).sort_values(by="rmse", ascending=True).reset_index(drop=True)
    comparison_df["rank"] = np.arange(1, len(comparison_df) + 1)
    return comparison_df[["rank", "model", "mae", "rmse", "r2"]]


def save_model_comparison_artifacts(
    comparison_df: pd.DataFrame,
    output_csv_path: str | Path,
    output_json_path: str | Path,
    output_txt_path: str | Path,
) -> Dict[str, Any]:
    """
    Persist model comparison table in CSV/JSON/TXT formats.

    Inputs:
    - comparison_df: model metrics table
    - output_csv_path: CSV destination
    - output_json_path: JSON destination
    - output_txt_path: text summary destination

    Output:
    - summary dictionary with best model and file paths
    """

    csv_path = Path(output_csv_path)
    json_path = Path(output_json_path)
    txt_path = Path(output_txt_path)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(csv_path, index=False)

    records = comparison_df.to_dict(orient="records")
    summary: Dict[str, Any] = {
        "records": records,
        "best_model": None,
        "best_rmse": None,
        "paths": {
            "csv": str(csv_path),
            "json": str(json_path),
            "txt": str(txt_path),
        },
    }

    if not comparison_df.empty:
        best_row = comparison_df.iloc[0]
        summary["best_model"] = str(best_row["model"])
        summary["best_rmse"] = float(best_row["rmse"])

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["=== Model Comparison Report ==="]
    if comparison_df.empty:
        lines.append("No model metrics available.")
    else:
        lines.append(f"Best model: {summary['best_model']}")
        lines.append(f"Best RMSE: {summary['best_rmse']:.4f}")
        lines.append("")
        for row in records:
            lines.append(
                f"#{row['rank']} {row['model']}: "
                f"MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f}, R2={row['r2']:.4f}"
            )

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return summary
