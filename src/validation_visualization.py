"""
Project: Microclimate Forecasting System using Machine Learning
Purpose: Implement dataset validation, forecasting visualizations, and report figures.
Module Group: Visualization & Reporting Module
DFD Connection: Consumes outputs from Data Management Module to produce quality checks
and plots used before ML Forecasting and Decision & Control stages.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

# Local cache paths avoid matplotlib/fontconfig permission issues in restricted environments.
PROJECT_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
PROJECT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CACHE_DIR / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_dataset_summary(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    scaled_df: pd.DataFrame,
) -> str:
    """
    Build a text summary of dataset health and preprocessing effects.

    Inputs:
    - raw_df: original dataset
    - cleaned_df: cleaned dataset
    - selected_df: temperature feature subset (before scaling)
    - scaled_df: selected features after normalization

    Output:
    - Multi-line summary text
    """

    summary_lines = [
        "=== Data Validation Summary ===",
        f"Raw shape: {raw_df.shape}",
        f"Cleaned shape: {cleaned_df.shape}",
        f"Selected (temperature-only) shape: {selected_df.shape}",
        f"Scaled shape: {scaled_df.shape}",
        "",
        "Missing values in raw dataset:",
        raw_df.isna().sum().to_string(),
        "",
        "Missing values after cleaning:",
        cleaned_df.isna().sum().to_string(),
        "",
        "Selected feature columns:",
        ", ".join(str(col) for col in selected_df.columns),
        "",
        "Scaled feature min values:",
        scaled_df.select_dtypes(include=["number"]).min().round(4).to_string(),
        "",
        "Scaled feature max values:",
        scaled_df.select_dtypes(include=["number"]).max().round(4).to_string(),
    ]
    return "\n".join(summary_lines)


def save_text_report(text: str, output_path: str | Path) -> None:
    """
    Save a text report to disk.

    Inputs:
    - text: report body
    - output_path: destination path
    """

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(text, encoding="utf-8")


def plot_temperature_vs_time(
    dataset: pd.DataFrame,
    output_path: str | Path,
    time_column: str = "timestamp",
) -> None:
    """
    Plot temperature trends over time for all temperature-related columns.

    Inputs:
    - dataset: cleaned dataset with temperature columns
    - output_path: figure destination path
    - time_column: timestamp column
    """

    temp_columns = [col for col in dataset.columns if "temp" in col.lower()]
    if not temp_columns:
        raise ValueError("No temperature columns available for trend visualization.")

    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(12, 5))

    for col in temp_columns:
        axis.plot(dataset[time_column], dataset[col], label=col, linewidth=1.7)

    axis.set_title("Temperature vs Time (Before Scaling)")
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("Temperature (C)")
    axis.legend(loc="best")
    figure.autofmt_xdate()
    figure.tight_layout()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=200)
    plt.close(figure)


def plot_before_after_preprocessing(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    target_column: str,
    output_path: str | Path,
    time_column: str = "timestamp",
) -> None:
    """
    Compare target temperature before and after normalization.

    Inputs:
    - before_df: temperature feature DataFrame (original scale)
    - after_df: normalized temperature feature DataFrame
    - target_column: selected target temperature column
    - output_path: figure destination path
    - time_column: timestamp column
    """

    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(before_df[time_column], before_df[target_column], color="#0C7BDC", linewidth=1.8)
    axes[0].set_title(f"Before Preprocessing: {target_column}")
    axes[0].set_ylabel("Temperature (C)")

    axes[1].plot(after_df[time_column], after_df[target_column], color="#00A86B", linewidth=1.8)
    axes[1].set_title(f"After Preprocessing (Min-Max Scaled): {target_column}")
    axes[1].set_ylabel("Scaled Value (0-1)")
    axes[1].set_xlabel("Timestamp")

    figure.autofmt_xdate()
    figure.tight_layout()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=200)
    plt.close(figure)


def plot_actual_vs_predicted(
    prediction_df: pd.DataFrame,
    output_path: str | Path,
    crop_name: str,
    timestamp_column: str = "timestamp",
) -> None:
    """
    Plot actual versus predicted temperature for a crop.

    Inputs:
    - prediction_df: dataframe containing actual and predicted temperature columns
    - output_path: figure destination
    - crop_name: crop label
    - timestamp_column: timestamp field name
    """

    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(
        prediction_df[timestamp_column],
        prediction_df["actual_temperature_c"],
        label="Actual",
        linewidth=1.8,
        color="#0C7BDC",
    )
    axis.plot(
        prediction_df[timestamp_column],
        prediction_df["predicted_temperature_c"],
        label="Predicted",
        linewidth=1.8,
        color="#E66100",
    )
    axis.set_title(f"Actual vs Predicted Temperature - {crop_name}")
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("Temperature (C)")
    axis.legend(loc="best")
    figure.autofmt_xdate()
    figure.tight_layout()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=200)
    plt.close(figure)


def plot_control_actions(
    control_df: pd.DataFrame,
    output_path: str | Path,
    crop_name: str,
    timestamp_column: str = "timestamp",
) -> None:
    """
    Visualize fan and spray actions over forecast timeline.

    Inputs:
    - control_df: control simulation dataframe
    - output_path: figure destination
    - crop_name: crop label
    - timestamp_column: timestamp field
    """

    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.step(
        control_df[timestamp_column],
        control_df["fan_on"],
        where="post",
        label="Fan",
        linewidth=1.8,
        color="#DC3220",
    )
    axis.step(
        control_df[timestamp_column],
        control_df["spray_on"],
        where="post",
        label="Spray",
        linewidth=1.8,
        color="#2AA198",
    )
    axis.set_title(f"Control Actions Timeline - {crop_name}")
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("Action State (0/1)")
    axis.set_ylim(-0.1, 1.1)
    axis.legend(loc="best")
    figure.autofmt_xdate()
    figure.tight_layout()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=200)
    plt.close(figure)


def plot_crop_metric_comparison(
    metrics_df: pd.DataFrame,
    output_path: str | Path,
    metric_column: str = "rmse",
) -> None:
    """
    Plot per-crop model performance comparison.

    Inputs:
    - metrics_df: table containing crop name and metric values
    - output_path: figure destination
    - metric_column: metric field to plot
    """

    if metrics_df.empty or metric_column not in metrics_df.columns:
        return

    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(10, 4))
    sns.barplot(data=metrics_df, x="crop_type", y=metric_column, ax=axis, color="#1B9E77")
    axis.set_title(f"Per-Crop {metric_column.upper()} Comparison")
    axis.set_xlabel("Crop Type")
    axis.set_ylabel(metric_column.upper())
    figure.tight_layout()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=200)
    plt.close(figure)


def plot_multi_model_temperature_predictions(
    prediction_df: pd.DataFrame,
    output_path: str | Path,
    crop_name: str,
    timestamp_column: str = "timestamp",
) -> None:
    """
    Plot actual temperature against multiple model outputs and hybrid output.

    Inputs:
    - prediction_df: prediction table with model-specific columns
    - output_path: figure destination
    - crop_name: crop label
    - timestamp_column: timestamp field name
    """

    required_columns = {timestamp_column, "actual_temperature_c"}
    if not required_columns.issubset(set(prediction_df.columns)):
        return

    candidate_columns = {
        "gru": "pred_gru_c",
        "bilstm": "pred_bilstm_c",
        "random_forest": "pred_random_forest_c",
        "gradient_boosting": "pred_gradient_boosting_c",
        "svr_rbf": "pred_svr_rbf_c",
        "knn_regressor": "pred_knn_regressor_c",
        "xgboost": "pred_xgboost_c",
        "linear_regression": "pred_linear_regression_c",
        "hybrid": "predicted_temperature_c",
    }

    available = {
        label: column
        for label, column in candidate_columns.items()
        if column in prediction_df.columns and prediction_df[column].notna().any()
    }
    if not available:
        return

    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(
        prediction_df[timestamp_column],
        prediction_df["actual_temperature_c"],
        label="actual",
        linewidth=1.9,
        color="#0C7BDC",
        alpha=0.9,
    )

    color_map = {
        "gru": "#4B4DED",
        "bilstm": "#228BE6",
        "random_forest": "#2D9F5D",
        "gradient_boosting": "#CA6702",
        "svr_rbf": "#1F7A8C",
        "knn_regressor": "#8F5F00",
        "xgboost": "#B93B8F",
        "linear_regression": "#6A4C93",
        "hybrid": "#D94801",
    }

    for label, column in available.items():
        axis.plot(
            prediction_df[timestamp_column],
            prediction_df[column],
            label=label,
            linewidth=1.3 if label != "hybrid" else 1.8,
            color=color_map.get(label, "#555555"),
            alpha=0.85,
        )

    axis.set_title(f"Actual vs Multi-Model Temperature Predictions - {crop_name}")
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("Temperature (C)")
    axis.legend(loc="best", ncol=2)
    figure.autofmt_xdate()
    figure.tight_layout()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=200)
    plt.close(figure)


def plot_model_comparison_bars(
    comparison_df: pd.DataFrame,
    output_path: str | Path,
    crop_name: str,
    metric_column: str = "rmse",
) -> None:
    """
    Plot model comparison bars for one evaluation metric.

    Inputs:
    - comparison_df: model comparison table
    - output_path: figure destination
    - crop_name: crop label
    - metric_column: metric field to visualize
    """

    if comparison_df.empty or metric_column not in comparison_df.columns:
        return

    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(9, 4))
    ordered = comparison_df.sort_values(by=metric_column, ascending=True)
    sns.barplot(
        data=ordered,
        x="model",
        y=metric_column,
        hue="model",
        palette="crest",
        legend=False,
        ax=axis,
    )
    axis.set_title(f"Model {metric_column.upper()} Comparison - {crop_name}")
    axis.set_xlabel("Model")
    axis.set_ylabel(metric_column.upper())
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=200)
    plt.close(figure)


def summarize_sequence_shapes(
    x_train_shape: tuple,
    y_train_shape: tuple,
    x_test_shape: tuple,
    y_test_shape: tuple,
) -> Dict[str, tuple]:
    """
    Create a compact shape dictionary for generated sequences.

    Inputs:
    - x_train_shape, y_train_shape, x_test_shape, y_test_shape: tuple shapes

    Output:
    - dictionary containing all shape metadata
    """

    return {
        "X_train_shape": x_train_shape,
        "y_train_shape": y_train_shape,
        "X_test_shape": x_test_shape,
        "y_test_shape": y_test_shape,
    }
