"""
Project: Microclimate Forecasting System using Machine Learning
Purpose: Add plant growth intelligence analytics from physiological greenhouse metrics.
Module Group: Visualization & Reporting Module (Extension)
DFD Connection: Uses CSV physiological features to provide plant-health and growth
insights that complement the temperature forecasting/control pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json
import os

PROJECT_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
PROJECT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CACHE_DIR / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.reporting_module import calculate_regression_metrics


PLANT_SCHEMA_ALIASES = {
    "ACHP": ["achp", "chlorophyll"],
    "PHR": ["phr", "plant_growth_rate", "growth_rate"],
    "ALAP": ["alap", "leaf_area"],
    "ANPL": ["anpl", "number_of_leaves", "leaf_count"],
    "ARD": ["ard", "root_diameter"],
    "ARL": ["arl", "root_length"],
    "ADWR": ["adwr", "root_dry_weight"],
    "AWWGV": ["awwgv", "vegetative_wet_weight", "biomass"],
    "PDMVG": ["pdmvg", "vegetative_dry_matter_pct"],
    "PDMRG": ["pdmrg", "root_dry_matter_pct"],
    "Class": ["class", "treatment", "crop_type"],
}

CORE_RELATION_COLUMNS = ["ACHP", "ALAP", "ARL", "ADWR", "AWWGV", "PHR", "PDMVG", "PDMRG", "ARD", "ANPL"]


def _normalize_col(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def _derive_greenhouse_system(class_name: str) -> str:
    text = str(class_name).strip().upper()
    if text.startswith("S"):
        return "iot_assisted"
    if text.startswith("T"):
        return "traditional"
    return "unknown"


def normalize_plant_growth_schema(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize physiological dataset into canonical plant-growth schema.

    Inputs:
    - dataset: raw physiological CSV DataFrame

    Output:
    - DataFrame with canonical columns used by plant intelligence analysis
    """

    normalized_map = {_normalize_col(col): col for col in dataset.columns}
    rename_map: Dict[str, str] = {}

    for target_col, aliases in PLANT_SCHEMA_ALIASES.items():
        keys = [_normalize_col(target_col), *[_normalize_col(alias) for alias in aliases]]
        for key in keys:
            if key in normalized_map:
                rename_map[normalized_map[key]] = target_col
                break

    standardized = dataset.rename(columns=rename_map).copy()

    for col in CORE_RELATION_COLUMNS:
        if col in standardized.columns:
            standardized[col] = pd.to_numeric(standardized[col], errors="coerce")

    if "Class" in standardized.columns:
        standardized["Class"] = standardized["Class"].astype(str).str.strip().str.upper()
        standardized["greenhouse_system"] = standardized["Class"].apply(_derive_greenhouse_system)
    else:
        standardized["Class"] = "UNKNOWN"
        standardized["greenhouse_system"] = "unknown"

    return standardized


def _fit_growth_predictor(
    dataset: pd.DataFrame,
    target_column: str = "PHR",
) -> Dict[str, Any]:
    feature_columns = [col for col in CORE_RELATION_COLUMNS if col in dataset.columns and col != target_column]
    if "Class" in dataset.columns:
        feature_columns.append("Class")

    fit_df = dataset[feature_columns + [target_column]].dropna().copy()
    if fit_df.empty:
        return {
            "metrics": {"mae": None, "rmse": None, "r2": None},
            "model_name": "unavailable",
            "feature_importance": pd.DataFrame(columns=["feature", "importance"]),
            "predictions": pd.DataFrame(columns=["actual", "predicted"]),
            "sample_count": 0,
            "notes": "Insufficient rows after cleaning for growth prediction.",
        }

    x = pd.get_dummies(fit_df.drop(columns=[target_column]), columns=["Class"], dummy_na=False)
    y = fit_df[target_column].astype(np.float32)

    split_idx = int(len(x) * 0.8)
    split_idx = max(1, min(split_idx, len(x) - 1))

    x_train = x.iloc[:split_idx]
    x_test = x.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    try:
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=320,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        importance_df = pd.DataFrame(
            {"feature": x.columns.tolist(), "importance": model.feature_importances_.tolist()}
        ).sort_values(by="importance", ascending=False)

        model_name = "RandomForestRegressor"
        notes = "Random Forest growth predictor trained successfully."
    except Exception as exc:  # pragma: no cover - optional runtime fallback
        x_train_np = x_train.to_numpy(dtype=np.float64)
        x_test_np = x_test.to_numpy(dtype=np.float64)
        ones_train = np.ones((x_train_np.shape[0], 1), dtype=np.float64)
        ones_test = np.ones((x_test_np.shape[0], 1), dtype=np.float64)
        design_train = np.concatenate([x_train_np, ones_train], axis=1)
        design_test = np.concatenate([x_test_np, ones_test], axis=1)
        coeffs = np.linalg.pinv(design_train.T @ design_train) @ design_train.T @ y_train.to_numpy(dtype=np.float64)
        y_pred = design_test @ coeffs
        coef_abs = np.abs(coeffs[:-1])
        denom = float(coef_abs.sum()) if float(coef_abs.sum()) > 0 else 1.0
        importance_df = pd.DataFrame(
            {"feature": x.columns.tolist(), "importance": (coef_abs / denom).tolist()}
        ).sort_values(by="importance", ascending=False)

        model_name = "LinearLeastSquaresFallback"
        notes = f"Fallback predictor used because sklearn RandomForest was unavailable: {exc}"

    metrics = calculate_regression_metrics(
        y_true=y_test.to_numpy(dtype=np.float32),
        y_pred=np.asarray(y_pred, dtype=np.float32),
    )

    prediction_df = pd.DataFrame(
        {
            "actual": y_test.to_numpy(dtype=np.float32),
            "predicted": np.asarray(y_pred, dtype=np.float32),
        }
    )

    return {
        "metrics": {"mae": metrics["mae"], "rmse": metrics["rmse"], "r2": metrics["r2"]},
        "model_name": model_name,
        "feature_importance": importance_df,
        "predictions": prediction_df,
        "sample_count": int(len(fit_df)),
        "notes": notes,
    }


def _detect_stress_patterns(dataset: pd.DataFrame) -> pd.DataFrame:
    stress_df = dataset.copy()

    for col in ["ACHP", "PHR", "ARL", "AWWGV"]:
        if col not in stress_df.columns:
            stress_df[col] = np.nan

    achp_thr = float(stress_df["ACHP"].quantile(0.2)) if stress_df["ACHP"].notna().any() else 0.0
    phr_thr = float(stress_df["PHR"].quantile(0.2)) if stress_df["PHR"].notna().any() else 0.0
    arl_thr = float(stress_df["ARL"].quantile(0.2)) if stress_df["ARL"].notna().any() else 0.0
    biomass_thr = float(stress_df["AWWGV"].quantile(0.2)) if stress_df["AWWGV"].notna().any() else 0.0

    stress_score = (
        (stress_df["ACHP"] <= achp_thr).astype(int)
        + (stress_df["PHR"] <= phr_thr).astype(int)
        + (stress_df["ARL"] <= arl_thr).astype(int)
        + (stress_df["AWWGV"] <= biomass_thr).astype(int)
    )
    stress_df["stress_score"] = stress_score

    stress_df["stress_level"] = "low"
    stress_df.loc[stress_df["stress_score"] >= 2, "stress_level"] = "moderate"
    stress_df.loc[stress_df["stress_score"] >= 3, "stress_level"] = "high"

    return stress_df


def _plot_growth_analytics(dataset: pd.DataFrame, figure_dir: Path) -> Dict[str, str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    outputs: Dict[str, str] = {}

    if {"ACHP", "ALAP"}.issubset(dataset.columns):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=dataset, x="ACHP", y="ALAP", hue="Class", ax=ax, s=35, alpha=0.75)
        ax.set_title("Chlorophyll (ACHP) vs Leaf Area (ALAP)")
        fig.tight_layout()
        path = figure_dir / "chlorophyll_vs_leaf_area.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        outputs["chlorophyll_vs_leaf_area"] = str(path)

    if {"ARL", "AWWGV"}.issubset(dataset.columns):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=dataset, x="ARL", y="AWWGV", hue="Class", ax=ax, s=35, alpha=0.75)
        ax.set_title("Root Length (ARL) vs Biomass (AWWGV)")
        fig.tight_layout()
        path = figure_dir / "root_length_vs_biomass.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        outputs["root_length_vs_biomass"] = str(path)

    corr_cols = [col for col in CORE_RELATION_COLUMNS if col in dataset.columns]
    if len(corr_cols) >= 3:
        corr = dataset[corr_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr, cmap="YlGnBu", annot=False, ax=ax)
        ax.set_title("Plant Physiological Correlation Heatmap")
        fig.tight_layout()
        path = figure_dir / "plant_feature_correlation_heatmap.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        outputs["feature_correlation_heatmap"] = str(path)

    if {"greenhouse_system", "PHR"}.issubset(dataset.columns):
        comp_df = dataset[dataset["greenhouse_system"].isin(["iot_assisted", "traditional"])].copy()
        if not comp_df.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(
                data=comp_df,
                x="greenhouse_system",
                y="PHR",
                hue="greenhouse_system",
                ax=ax,
                palette=["#2b8a3e", "#f08c00"],
                legend=False,
            )
            ax.set_title("IoT-Assisted vs Traditional Greenhouse (PHR)")
            ax.set_xlabel("Greenhouse Type")
            ax.set_ylabel("Plant Growth Rate (PHR)")
            fig.tight_layout()
            path = figure_dir / "iot_vs_traditional_growth_comparison.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            outputs["iot_vs_traditional"] = str(path)

    return outputs


def run_plant_growth_intelligence(
    dataset_path: str | Path,
    report_dir: str | Path,
    figure_dir: str | Path,
) -> Dict[str, Any]:
    """
    Execute plant-growth intelligence extension workflow.

    Inputs:
    - dataset_path: plant physiological CSV input
    - report_dir: destination directory for CSV/JSON/TXT insights
    - figure_dir: destination directory for growth analytics figures

    Output:
    - summary dictionary with analysis paths and key metrics
    """

    input_path = Path(dataset_path)
    output_dir = Path(report_dir)
    fig_dir = Path(figure_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        return {
            "status": "skipped",
            "reason": f"Dataset not found: {input_path}",
            "report_dir": str(output_dir),
        }

    raw_df = pd.read_csv(input_path)
    growth_df = normalize_plant_growth_schema(raw_df)

    stress_df = _detect_stress_patterns(growth_df)
    stress_summary = stress_df["stress_level"].value_counts(dropna=False).to_dict()

    predictor_result = _fit_growth_predictor(growth_df, target_column="PHR" if "PHR" in growth_df.columns else "AWWGV")
    importance_df = predictor_result["feature_importance"]

    feature_importance_path = output_dir / "plant_feature_importance.csv"
    importance_df.to_csv(feature_importance_path, index=False)

    stress_samples_path = output_dir / "stress_detection_samples.csv"
    stress_df.to_csv(stress_samples_path, index=False)

    figures = _plot_growth_analytics(growth_df, fig_dir)

    summary = {
        "status": "completed",
        "dataset": str(input_path),
        "sample_count": int(len(growth_df)),
        "growth_prediction_model": predictor_result["model_name"],
        "growth_prediction_metrics": predictor_result["metrics"],
        "stress_summary": stress_summary,
        "top_features": importance_df.head(8).to_dict(orient="records"),
        "notes": predictor_result["notes"],
        "figures": figures,
        "reports": {
            "feature_importance": str(feature_importance_path),
            "stress_samples": str(stress_samples_path),
        },
    }

    json_path = output_dir / "plant_growth_intelligence_summary.json"
    txt_path = output_dir / "plant_growth_intelligence_summary.txt"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: List[str] = [
        "=== Plant Growth Intelligence Summary ===",
        f"Dataset: {input_path}",
        f"Rows analyzed: {len(growth_df)}",
        f"Growth predictor: {predictor_result['model_name']}",
        f"MAE: {summary['growth_prediction_metrics']['mae']:.4f}"
        if summary["growth_prediction_metrics"]["mae"] is not None
        else "MAE: N/A",
        f"RMSE: {summary['growth_prediction_metrics']['rmse']:.4f}"
        if summary["growth_prediction_metrics"]["rmse"] is not None
        else "RMSE: N/A",
        f"R2: {summary['growth_prediction_metrics']['r2']:.4f}"
        if summary["growth_prediction_metrics"]["r2"] is not None
        else "R2: N/A",
        "",
        "Stress level distribution:",
    ]
    for level, count in sorted(stress_summary.items()):
        lines.append(f"- {level}: {count}")

    lines.append("")
    lines.append("Top contributing features:")
    for row in summary["top_features"]:
        lines.append(f"- {row['feature']}: {float(row['importance']):.4f}")

    txt_path.write_text("\n".join(lines), encoding="utf-8")

    summary["reports"]["summary_json"] = str(json_path)
    summary["reports"]["summary_txt"] = str(txt_path)

    return summary

