"""
Project: Microclimate Forecasting System using Machine Learning
Purpose: Generate forecasting and control-simulation reports.
Module Group: Visualization & Reporting Module
DFD Connection: Combines outputs from ML Forecasting and Decision & Control
Simulation Modules into academic evaluation reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json

import numpy as np
import pandas as pd


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    non_zero_mask = np.abs(y_true) > 1e-8
    if not np.any(non_zero_mask):
        return 0.0
    return float(np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100.0)


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard regression metrics for forecasting evaluation.

    Inputs:
    - y_true: ground-truth values
    - y_pred: predicted values

    Output:
    - dictionary of MAE, RMSE, MAPE, R2
    """

    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = _safe_mape(y_true=y_true, y_pred=y_pred)

    total_variance = np.sum((y_true - np.mean(y_true)) ** 2)
    residual_variance = np.sum((y_true - y_pred) ** 2)
    r2 = 0.0 if total_variance == 0 else float(1 - (residual_variance / total_variance))

    return {"mae": mae, "rmse": rmse, "mape_pct": mape, "r2": r2}


def generate_model_performance_report(
    predictions_df: pd.DataFrame,
    crop_name: str,
    backend: str,
    output_txt_path: str | Path,
    output_json_path: str | Path,
) -> Dict[str, float]:
    """
    Generate and persist forecasting performance report.

    Inputs:
    - predictions_df: DataFrame with actual and predicted temperature columns
    - crop_name: crop label for report context
    - backend: active model backend
    - output_txt_path: report text destination
    - output_json_path: report JSON destination

    Output:
    - metric dictionary
    """

    y_true = predictions_df["actual_temperature_c"].to_numpy(dtype=np.float32)
    y_pred = predictions_df["predicted_temperature_c"].to_numpy(dtype=np.float32)
    metrics = calculate_regression_metrics(y_true=y_true, y_pred=y_pred)
    metrics["sample_count"] = int(len(predictions_df))
    metrics["backend"] = backend

    lines = [
        f"=== Forecast Performance Report: {crop_name} ===",
        f"Model backend: {backend}",
        f"Samples: {metrics['sample_count']}",
        f"MAE (C): {metrics['mae']:.4f}",
        f"RMSE (C): {metrics['rmse']:.4f}",
        f"MAPE (%): {metrics['mape_pct']:.2f}",
        f"R2: {metrics['r2']:.4f}",
    ]

    txt_path = Path(output_txt_path)
    json_path = Path(output_json_path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def generate_control_simulation_report(
    control_df: pd.DataFrame,
    crop_name: str,
    output_txt_path: str | Path,
    output_json_path: str | Path,
) -> Dict[str, int]:
    """
    Generate and persist control simulation summary report.

    Inputs:
    - control_df: action timeline DataFrame
    - crop_name: crop label
    - output_txt_path: report text destination
    - output_json_path: report JSON destination

    Output:
    - control action summary dictionary
    """

    summary = {
        "total_records": int(len(control_df)),
        "fan_on_events": int(control_df["fan_on"].sum()) if not control_df.empty else 0,
        "spray_on_events": int(control_df["spray_on"].sum()) if not control_df.empty else 0,
        "fan_and_spray_events": int((control_df["action"] == "fan_and_spray_on").sum())
        if not control_df.empty
        else 0,
        "idle_events": int((control_df["action"] == "idle").sum()) if not control_df.empty else 0,
    }

    lines = [
        f"=== Control Simulation Report: {crop_name} ===",
        f"Total simulated records: {summary['total_records']}",
        f"Fan ON events: {summary['fan_on_events']}",
        f"Spray ON events: {summary['spray_on_events']}",
        f"Fan + Spray events: {summary['fan_and_spray_events']}",
        f"Idle events: {summary['idle_events']}",
    ]

    txt_path = Path(output_txt_path)
    json_path = Path(output_json_path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def generate_contribution_note(output_path: str | Path) -> None:
    """
    Save the project contribution statement relative to greenhouse growth literature.

    Inputs:
    - output_path: destination markdown/text file
    """

    note = [
        "# Contribution Over Baseline Greenhouse Growth Studies",
        "",
        "Reference considered: Scientific Reports paper DOI 10.1038/s41598-025-15615-3",
        "(focus on non-invasive greenhouse monitoring and deep-learning driven prediction).",
        "",
        "Software-only scope maintained:",
        "- No IoT hardware integration in this implementation.",
        "- CSV datasets are the only data input channel.",
        "",
        "Project contributions:",
        "1. Implemented time-series forecasting pipeline (sliding-window) for greenhouse temperature prediction.",
        "2. Extended forecasting with parallel ML models (RandomForest, GradientBoosting, LinearRegression, optional XGBoost).",
        "3. Added model-evaluation layer (MAE, RMSE, R2) with per-crop comparison and ranking reports.",
        "4. Implemented dynamic inverse-RMSE weighted coordination for final hybrid prediction.",
        "5. Integrated threshold-based decision simulation (fan/spray) connected to hybrid prediction.",
        "6. Added UI-ready payload export (actual vs predicted + action timeline) for frontend styling.",
    ]
    path_obj = Path(output_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text("\n".join(note), encoding="utf-8")


def save_overall_results_table(results: List[Dict[str, object]], output_path: str | Path) -> None:
    """
    Persist overall per-crop result table.

    Inputs:
    - results: list of result rows
    - output_path: destination CSV path
    """

    table = pd.DataFrame(results)
    path_obj = Path(output_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path_obj, index=False)


def generate_model_comparison_report(
    comparison_df: pd.DataFrame,
    crop_name: str,
    output_csv_path: str | Path,
    output_txt_path: str | Path,
    output_json_path: str | Path,
) -> Dict[str, object]:
    """
    Save model-comparison metrics and ranking summary for one crop.

    Inputs:
    - comparison_df: model comparison table with rank/model/mae/rmse/r2 columns
    - crop_name: crop/class name
    - output_csv_path: destination CSV path
    - output_txt_path: destination TXT path
    - output_json_path: destination JSON path

    Output:
    - summary dictionary including best model and ranking rows
    """

    csv_path = Path(output_csv_path)
    txt_path = Path(output_txt_path)
    json_path = Path(output_json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_df = comparison_df.copy()
    if not normalized_df.empty and "rank" not in normalized_df.columns:
        normalized_df = normalized_df.sort_values(by="rmse", ascending=True).reset_index(drop=True)
        normalized_df["rank"] = np.arange(1, len(normalized_df) + 1)

    normalized_df.to_csv(csv_path, index=False)

    rows = normalized_df.to_dict(orient="records")
    best_model = rows[0]["model"] if rows else None
    best_rmse = float(rows[0]["rmse"]) if rows else None
    summary: Dict[str, object] = {
        "crop_name": crop_name,
        "best_model": best_model,
        "best_rmse": best_rmse,
        "models_evaluated": int(len(rows)),
        "ranking": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [f"=== Model Comparison Report: {crop_name} ==="]
    if not rows:
        lines.append("No model rows available.")
    else:
        lines.append(f"Best model: {best_model}")
        lines.append(f"Best RMSE: {best_rmse:.4f}")
        lines.append("")
        for row in rows:
            lines.append(
                f"#{int(row['rank'])} {row['model']}: "
                f"MAE={float(row['mae']):.4f}, RMSE={float(row['rmse']):.4f}, R2={float(row['r2']):.4f}"
            )
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return summary
