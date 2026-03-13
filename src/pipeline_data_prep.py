"""
Project: Microclimate Forecasting System using Machine Learning
Purpose: Run complete end-to-end workflow (data, ML, control simulation, reporting).
Module Group: Cross-module integration pipeline
DFD Connection: Orchestrates Data Management -> ML Forecasting -> Decision Simulation ->
Visualization & Reporting modules using CSV-only greenhouse datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence
import re
import json

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_CROP_COLUMN,
    DEFAULT_HIGH_THRESHOLD_C,
    DEFAULT_LOW_THRESHOLD_C,
    DEFAULT_LSTM_BATCH_SIZE,
    DEFAULT_LSTM_EPOCHS,
    DEFAULT_MIN_SAMPLES_PER_CROP,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_SPRAY_THRESHOLD_C,
    DEFAULT_TARGET_COLUMN,
    DEFAULT_TEST_RATIO,
    DEFAULT_TIME_COLUMN,
    DOCS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_CROPWISE_DIR,
    PROCESSED_DIR,
    RAW_CROP_DIR,
    RAW_DATA_PATH,
    RAW_EXTERNAL_KAGGLE_PATH,
    RAW_MULTI_CROP_PATH,
    REPORTS_DIR,
)
from src.data_management import (
    clean_dataset,
    export_cropwise_datasets,
    generate_lstm_sequences,
    inverse_min_max_scale,
    load_and_merge_datasets,
    min_max_scale_features,
    save_scaling_parameters,
    select_forecasting_features,
    split_by_crop,
    train_test_split_time_series,
)
from src.decision_control_module import simulate_fan_spray_actions
from src.hybrid_forecasting_extension import (
    build_model_comparison_table,
    coordinate_hybrid_prediction,
    predict_secondary_forecasters,
    save_model_comparison_artifacts,
    train_secondary_forecasters,
)
from src.ml_forecasting_module import (
    LinearSequenceRegressor,
    predict_temperature,
    train_temperature_forecaster,
)
from src.reporting_module import (
    generate_contribution_note,
    generate_control_simulation_report,
    generate_model_comparison_report,
    generate_model_performance_report,
    save_overall_results_table,
)
from src.validation_visualization import (
    generate_dataset_summary,
    plot_actual_vs_predicted,
    plot_before_after_preprocessing,
    plot_control_actions,
    plot_crop_metric_comparison,
    plot_model_comparison_bars,
    plot_multi_model_temperature_predictions,
    plot_temperature_vs_time,
    save_text_report,
)


def _slugify(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", text.lower().strip())
    return normalized.strip("_") or "unknown"


def _ensure_output_directories() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_CROPWISE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_dataset_inputs(custom_inputs: Sequence[str | Path] | None = None) -> List[Path]:
    if custom_inputs:
        candidates = [Path(item) for item in custom_inputs]
    else:
        if RAW_EXTERNAL_KAGGLE_PATH.exists():
            return [RAW_EXTERNAL_KAGGLE_PATH]
        candidates = [RAW_EXTERNAL_KAGGLE_PATH, RAW_MULTI_CROP_PATH, RAW_CROP_DIR, RAW_DATA_PATH]

    existing = [path for path in candidates if path.exists()]
    if not existing:
        raise ValueError("No dataset source found. Add CSV files to data/raw and rerun.")
    return existing


def _split_train_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    validation_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(x_train) < 5:
        return x_train, y_train, x_train, y_train

    val_size = max(1, int(len(x_train) * validation_ratio))
    train_size = len(x_train) - val_size
    if train_size < 2:
        train_size = len(x_train) - 1
        val_size = 1

    return (
        x_train[:train_size],
        y_train[:train_size],
        x_train[train_size:],
        y_train[train_size:],
    )


def _save_numpy_arrays(output_dir: Path, arrays: Dict[str, np.ndarray]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in arrays.items():
        np.save(output_dir / f"{name}.npy", values)


def _write_json(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_hybrid_architecture_docs() -> Dict[str, str]:
    architecture_path = DOCS_DIR / "hybrid_system_architecture.md"
    pipeline_path = DOCS_DIR / "updated_ml_pipeline.md"

    architecture_text = """# Hybrid AI Microclimate Forecasting Architecture

```mermaid
flowchart LR
    A[Data Management Module\nCSV ingestion, cleaning, scaling, sequences]
    B[Machine Learning Forecasting Module\nLSTM + parallel ML models]
    C[Model Evaluation Module\nMAE, RMSE, R2 comparison table]
    D[Model Coordination Layer\nDynamic weighted hybrid prediction]
    E[Decision and Control Simulation Module\nThreshold rule engine]
    F[Visualization and Reporting Module\nDashboard payload + plots + reports]

    A --> B --> C --> D --> E --> F
```

This architecture implements a complete microclimate forecasting pipeline.
"""

    pipeline_text = """# Updated Methodology Pipeline

Data Collection (CSV-only)
-> Data Preprocessing
-> Feature Engineering
-> Parallel Model Training:
   - LSTM Forecasting
   - Random Forest
   - Gradient Boosting
   - Linear Regression baseline
   - Optional XGBoost (when installed)
-> Model Evaluation (MAE, RMSE, R2)
-> Model Coordination Layer (dynamic inverse-RMSE weighting)
-> Final Optimized Temperature Prediction
-> Decision and Control Simulation (fan/spray thresholds unchanged)
-> Visualization and Reporting
"""

    architecture_path.write_text(architecture_text, encoding="utf-8")
    pipeline_path.write_text(pipeline_text, encoding="utf-8")
    return {"architecture_doc": str(architecture_path), "pipeline_doc": str(pipeline_path)}


def run_full_project_workflow(
    dataset_inputs: Sequence[str | Path] | None = None,
    time_column: str = DEFAULT_TIME_COLUMN,
    crop_column: str = DEFAULT_CROP_COLUMN,
    target_column: str = DEFAULT_TARGET_COLUMN,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    test_ratio: float = DEFAULT_TEST_RATIO,
    min_samples_per_crop: int = DEFAULT_MIN_SAMPLES_PER_CROP,
    epochs: int = DEFAULT_LSTM_EPOCHS,
    batch_size: int = DEFAULT_LSTM_BATCH_SIZE,
    low_threshold_c: float = DEFAULT_LOW_THRESHOLD_C,
    high_threshold_c: float = DEFAULT_HIGH_THRESHOLD_C,
    spray_threshold_c: float = DEFAULT_SPRAY_THRESHOLD_C,
) -> Dict[str, Any]:
    """
    Execute the complete forecasting system pipeline.

    Inputs:
    - dataset_inputs: optional files/directories of raw CSVs
    - time_column, crop_column, target_column: canonical schema fields
    - sequence_length: history window length for LSTM sequence generation
    - test_ratio: chronological split ratio for test partition
    - min_samples_per_crop: minimum samples required to run a crop pipeline
    - epochs, batch_size: model training parameters
    - low_threshold_c, high_threshold_c, spray_threshold_c: control thresholds

    Output:
    - dictionary containing run summary and major artifact paths
    """

    _ensure_output_directories()
    input_sources = _resolve_dataset_inputs(custom_inputs=dataset_inputs)

    raw_df = load_and_merge_datasets(dataset_inputs=input_sources, time_column=time_column, crop_column=crop_column)
    cleaned_df = clean_dataset(dataset=raw_df, time_column=time_column)
    cropwise_raw_exports = export_cropwise_datasets(
        dataset=cleaned_df,
        output_dir=PROCESSED_CROPWISE_DIR / "classified_inputs",
        crop_column=crop_column,
    )

    selected_all_df, resolved_target = select_forecasting_features(
        dataset=cleaned_df,
        preferred_target=target_column,
        time_column=time_column,
        crop_column=crop_column,
    )
    scaled_all_df, _ = min_max_scale_features(
        dataset=selected_all_df,
        time_column=time_column,
        protected_columns=(crop_column,),
    )

    summary_text = generate_dataset_summary(
        raw_df=raw_df,
        cleaned_df=cleaned_df,
        selected_df=selected_all_df,
        scaled_df=scaled_all_df,
    )
    summary_text += (
        "\n\n=== Full Pipeline Configuration ===\n"
        f"Input sources: {[str(path) for path in input_sources]}\n"
        f"Resolved target column: {resolved_target}\n"
        f"Sequence length: {sequence_length}\n"
        f"Test ratio: {test_ratio}\n"
        f"Control thresholds (low/high/spray): {low_threshold_c}/{high_threshold_c}/{spray_threshold_c}\n"
    )
    save_text_report(summary_text, REPORTS_DIR / "data_summary.txt")

    plot_temperature_vs_time(cleaned_df, FIGURES_DIR / "temperature_trend.png", time_column=time_column)
    plot_before_after_preprocessing(
        before_df=selected_all_df,
        after_df=scaled_all_df,
        target_column=resolved_target,
        output_path=FIGURES_DIR / "preprocessing_before_after.png",
        time_column=time_column,
    )
    docs_artifacts = _write_hybrid_architecture_docs()

    crop_groups = split_by_crop(
        dataset=cleaned_df,
        crop_column=crop_column,
        min_samples_per_crop=max(min_samples_per_crop, sequence_length + 6),
    )
    if not crop_groups:
        raise ValueError("No crop groups met minimum sample criteria for model training.")

    overall_rows: List[Dict[str, Any]] = []
    model_ranking_rows: List[Dict[str, Any]] = []
    ui_payload_rows: List[Dict[str, Any]] = []
    processed_crop_count = 0

    for crop_name, crop_df in crop_groups.items():
        crop_slug = _slugify(crop_name)
        crop_dir = PROCESSED_CROPWISE_DIR / crop_slug
        crop_models_dir = MODELS_DIR / crop_slug
        crop_reports_dir = REPORTS_DIR / "crop_reports" / crop_slug
        crop_figures_dir = FIGURES_DIR / "cropwise" / crop_slug

        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_models_dir.mkdir(parents=True, exist_ok=True)
        crop_reports_dir.mkdir(parents=True, exist_ok=True)
        crop_figures_dir.mkdir(parents=True, exist_ok=True)

        selected_df, target_col = select_forecasting_features(
            dataset=crop_df,
            preferred_target=resolved_target,
            time_column=time_column,
            crop_column=crop_column,
        )
        scaled_df, scaling_params = min_max_scale_features(
            dataset=selected_df,
            time_column=time_column,
            protected_columns=(crop_column,),
        )

        selected_df.to_csv(crop_dir / "features_cleaned.csv", index=False)
        scaled_df.to_csv(crop_dir / "features_scaled.csv", index=False)
        save_scaling_parameters(parameters=scaling_params, output_path=crop_dir / "scaling_parameters.json")

        train_df, test_df = train_test_split_time_series(dataset=scaled_df, test_ratio=test_ratio)
        x_train, y_train, y_train_timestamps = generate_lstm_sequences(
            dataset=train_df,
            target_column=target_col,
            sequence_length=sequence_length,
            time_column=time_column,
            protected_columns=(crop_column,),
        )
        x_test, y_test, y_test_timestamps = generate_lstm_sequences(
            dataset=test_df,
            target_column=target_col,
            sequence_length=sequence_length,
            time_column=time_column,
            protected_columns=(crop_column,),
        )

        if len(x_train) < 3 or len(x_test) == 0:
            continue

        _save_numpy_arrays(
            output_dir=crop_dir,
            arrays={
                "X_train": x_train,
                "y_train": y_train,
                "X_test": x_test,
                "y_test": y_test,
            },
        )

        x_fit, y_fit, x_val, y_val = _split_train_validation(x_train=x_train, y_train=y_train, validation_ratio=0.2)

        primary_model = train_temperature_forecaster(
            x_train=x_fit,
            y_train=y_fit,
            x_val=x_val,
            y_val=y_val,
            model_dir=crop_models_dir,
            model_tag=f"{crop_slug}_primary",
            epochs=epochs,
            batch_size=batch_size,
            force_backend="auto",
        )

        # Explicit LSTM candidate: treated as an independent model for hybrid coordination.
        # If primary already uses LSTM, reuse it instead of training a duplicate network.
        lstm_candidate = None
        lstm_candidate_note = "not_attempted"
        if primary_model.backend == "lstm":
            lstm_candidate = primary_model
            lstm_candidate_note = "trained_via_primary"
        else:
            try:
                lstm_candidate = train_temperature_forecaster(
                    x_train=x_fit,
                    y_train=y_fit,
                    x_val=x_val,
                    y_val=y_val,
                    model_dir=crop_models_dir,
                    model_tag=f"{crop_slug}_lstm_explicit",
                    epochs=epochs,
                    batch_size=batch_size,
                    force_backend="lstm",
                )
                lstm_candidate_note = "trained"
            except Exception as exc:
                lstm_candidate_note = f"Unavailable: {exc}"

        # Baseline model from existing pipeline remains intact.
        linear_baseline = LinearSequenceRegressor(l2_penalty=1e-3)
        linear_baseline.fit(x_fit, y_fit)
        linear_baseline.save(crop_models_dir / f"{crop_slug}_linear_baseline.npy")

        feature_columns = [col for col in scaled_df.columns if col not in {time_column, crop_column}]
        target_index = feature_columns.index(target_col)

        model_predictions_test_scaled: Dict[str, np.ndarray] = {
            "linear_baseline": linear_baseline.predict(x_test).astype(np.float32),
            "naive_persistence": x_test[:, -1, target_index].astype(np.float32),
        }
        model_predictions_val_scaled: Dict[str, np.ndarray] = {
            "linear_baseline": linear_baseline.predict(x_val).astype(np.float32),
            "naive_persistence": x_val[:, -1, target_index].astype(np.float32),
        }

        if primary_model.backend == "lstm":
            model_predictions_test_scaled["lstm"] = predict_temperature(primary_model, x_test)
            model_predictions_val_scaled["lstm"] = predict_temperature(primary_model, x_val)
            preferred_model = "lstm"
        else:
            model_predictions_test_scaled["primary_linear"] = predict_temperature(primary_model, x_test)
            model_predictions_val_scaled["primary_linear"] = predict_temperature(primary_model, x_val)
            preferred_model = "primary_linear"

        # Add explicit LSTM model if available and not already present.
        if lstm_candidate is not None and lstm_candidate.backend == "lstm" and "lstm" not in model_predictions_test_scaled:
            model_predictions_test_scaled["lstm"] = predict_temperature(lstm_candidate, x_test)
            model_predictions_val_scaled["lstm"] = predict_temperature(lstm_candidate, x_val)
            preferred_model = "lstm"

        # Extension: train independent ML models (RF/GB/LinearRegression/optional XGBoost).
        secondary_registry = train_secondary_forecasters(x_train=x_fit, y_train=y_fit)
        secondary_test_predictions = predict_secondary_forecasters(secondary_registry, x_input=x_test)
        secondary_val_predictions = predict_secondary_forecasters(secondary_registry, x_input=x_val)
        for model_name, test_pred in secondary_test_predictions.items():
            if model_name in secondary_val_predictions:
                model_predictions_test_scaled[model_name] = test_pred
                model_predictions_val_scaled[model_name] = secondary_val_predictions[model_name]

        registry_notes = dict(secondary_registry.notes)
        registry_notes["primary_backend"] = primary_model.backend
        registry_notes["lstm_candidate"] = lstm_candidate_note
        if primary_model.fallback_reason:
            registry_notes["primary_fallback_reason"] = primary_model.fallback_reason

        _write_json(
            payload={"notes": registry_notes, "models": list(model_predictions_test_scaled.keys())},
            output_path=crop_reports_dir / "secondary_model_registry.json",
        )

        # Coordination layer: dynamic weighted ensemble based on validation RMSE.
        coordination = coordinate_hybrid_prediction(
            y_validation=y_val,
            validation_predictions=model_predictions_val_scaled,
            test_predictions=model_predictions_test_scaled,
            preferred_model=preferred_model,
        )
        ensemble_pred_scaled = coordination.final_prediction
        sorted_weights = sorted(coordination.weights.items(), key=lambda item: item[1], reverse=True)
        weight_text = ", ".join([f"{name}:{weight:.2f}" for name, weight in sorted_weights])
        model_coordination = f"{coordination.method}[{weight_text}]"

        actual_temperature = inverse_min_max_scale(y_test, target_col, scaling_params)
        predicted_temperature = inverse_min_max_scale(ensemble_pred_scaled, target_col, scaling_params)

        model_predictions_original: Dict[str, np.ndarray] = {}
        for model_name, scaled_prediction in model_predictions_test_scaled.items():
            model_predictions_original[model_name] = inverse_min_max_scale(
                np.asarray(scaled_prediction, dtype=np.float32),
                target_col,
                scaling_params,
            )
        model_predictions_original["hybrid_coordinated"] = predicted_temperature

        comparison_df = build_model_comparison_table(
            y_true=actual_temperature,
            prediction_map=model_predictions_original,
        )
        save_model_comparison_artifacts(
            comparison_df=comparison_df,
            output_csv_path=crop_reports_dir / "model_comparison_artifacts.csv",
            output_json_path=crop_reports_dir / "model_comparison_artifacts.json",
            output_txt_path=crop_reports_dir / "model_comparison_artifacts.txt",
        )
        model_ranking = generate_model_comparison_report(
            comparison_df=comparison_df,
            crop_name=crop_name,
            output_csv_path=crop_reports_dir / "model_comparison.csv",
            output_txt_path=crop_reports_dir / "model_comparison.txt",
            output_json_path=crop_reports_dir / "model_comparison.json",
        )

        prediction_payload: Dict[str, Any] = {
            time_column: [ts.isoformat() for ts in y_test_timestamps],
            crop_column: crop_name,
            "actual_temperature_c": actual_temperature,
            "predicted_temperature_c": predicted_temperature,
            "absolute_error_c": np.abs(actual_temperature - predicted_temperature),
        }
        for model_name, values in model_predictions_original.items():
            if model_name == "hybrid_coordinated":
                continue
            prediction_payload[f"pred_{_slugify(model_name)}_c"] = values

        prediction_df = pd.DataFrame(prediction_payload)
        prediction_df.to_csv(crop_dir / "predictions.csv", index=False)

        control_df = simulate_fan_spray_actions(
            forecast_df=prediction_df,
            high_threshold=high_threshold_c,
            low_threshold=low_threshold_c,
            spray_threshold=spray_threshold_c,
            timestamp_column=time_column,
            crop_column=crop_column,
            prediction_column="predicted_temperature_c",
        )
        control_df.to_csv(crop_dir / "control_actions.csv", index=False)

        ui_payload = {
            "crop_type": crop_name,
            "timestamps": prediction_df[time_column].tolist(),
            "actual_temperature_c": prediction_df["actual_temperature_c"].round(4).tolist(),
            "predicted_temperature_c": prediction_df["predicted_temperature_c"].round(4).tolist(),
            "absolute_error_c": prediction_df["absolute_error_c"].round(4).tolist(),
            "actions": control_df["action"].tolist(),
            "fan_on": control_df["fan_on"].tolist(),
            "spray_on": control_df["spray_on"].tolist(),
            "model_coordination": model_coordination,
            "model_weights": coordination.weights,
            "validation_rmse": coordination.validation_rmse,
            "model_comparison": comparison_df.to_dict(orient="records"),
            "best_model": model_ranking.get("best_model"),
            "model_prediction_columns": [col for col in prediction_df.columns if col.startswith("pred_")],
            "model_prediction_series": {
                "hybrid_coordinated": prediction_df["predicted_temperature_c"].round(4).tolist(),
                **{
                    col.replace("pred_", "").replace("_c", ""): prediction_df[col].round(4).tolist()
                    for col in prediction_df.columns
                    if col.startswith("pred_")
                },
            },
        }
        _write_json(ui_payload, crop_reports_dir / "ui_payload.json")

        model_metrics = generate_model_performance_report(
            predictions_df=prediction_df,
            crop_name=crop_name,
            backend=model_coordination,
            output_txt_path=crop_reports_dir / "model_performance.txt",
            output_json_path=crop_reports_dir / "model_performance.json",
        )
        control_summary = generate_control_simulation_report(
            control_df=control_df,
            crop_name=crop_name,
            output_txt_path=crop_reports_dir / "control_simulation.txt",
            output_json_path=crop_reports_dir / "control_simulation.json",
        )

        plot_actual_vs_predicted(
            prediction_df=prediction_df,
            output_path=crop_figures_dir / "actual_vs_predicted.png",
            crop_name=crop_name,
            timestamp_column=time_column,
        )
        plot_multi_model_temperature_predictions(
            prediction_df=prediction_df,
            output_path=crop_figures_dir / "actual_vs_multi_model_vs_hybrid.png",
            crop_name=crop_name,
            timestamp_column=time_column,
        )
        plot_model_comparison_bars(
            comparison_df=comparison_df,
            output_path=crop_figures_dir / "model_rmse_comparison.png",
            crop_name=crop_name,
            metric_column="rmse",
        )
        plot_control_actions(
            control_df=control_df,
            output_path=crop_figures_dir / "control_actions.png",
            crop_name=crop_name,
            timestamp_column=time_column,
        )

        for row in comparison_df.to_dict(orient="records"):
            model_ranking_rows.append({"crop_type": crop_name, **row})

        overall_row: Dict[str, Any] = {
            "crop_type": crop_name,
            "samples_total": int(len(crop_df)),
            "train_sequences": int(len(x_train)),
            "test_sequences": int(len(x_test)),
            "target_column": target_col,
            "model_coordination": model_coordination,
            "best_model": model_ranking.get("best_model"),
            "best_model_rmse": model_ranking.get("best_rmse"),
            "models_evaluated": model_ranking.get("models_evaluated", 0),
            **model_metrics,
            **control_summary,
        }
        if primary_model.fallback_reason:
            overall_row["fallback_reason"] = primary_model.fallback_reason
        overall_rows.append(overall_row)
        ui_payload_rows.append(
            {
                "crop_type": crop_name,
                "model_coordination": model_coordination,
                "payload_file": str(crop_reports_dir / "ui_payload.json"),
            }
        )
        processed_crop_count += 1

    overall_df = pd.DataFrame(overall_rows)
    model_ranking_df = pd.DataFrame(model_ranking_rows)

    save_overall_results_table(overall_rows, REPORTS_DIR / "overall_crop_results.csv")
    plot_crop_metric_comparison(overall_df, FIGURES_DIR / "rmse_by_crop.png", metric_column="rmse")
    overall_model_ranking_path = REPORTS_DIR / "overall_model_ranking.csv"
    overall_model_comparison_path = REPORTS_DIR / "overall_model_comparison.csv"
    if not model_ranking_df.empty:
        model_ranking_df.to_csv(overall_model_comparison_path, index=False)
        ranking_summary_df = (
            model_ranking_df.groupby("model", as_index=False)[["mae", "rmse", "r2"]]
            .mean()
            .sort_values(by="rmse", ascending=True)
            .reset_index(drop=True)
        )
        ranking_summary_df["rank"] = np.arange(1, len(ranking_summary_df) + 1)
        ranking_summary_df = ranking_summary_df[["rank", "model", "mae", "rmse", "r2"]]
        ranking_summary_df.to_csv(overall_model_ranking_path, index=False)
    else:
        pd.DataFrame(columns=["crop_type", "rank", "model", "mae", "rmse", "r2"]).to_csv(
            overall_model_comparison_path, index=False
        )
        pd.DataFrame(columns=["rank", "model", "mae", "rmse", "r2"]).to_csv(overall_model_ranking_path, index=False)

    generate_contribution_note(DOCS_DIR / "paper_contribution_analysis.md")
    _write_json(
        {
            "project": "Microclimate Forecasting System",
            "crops_processed": processed_crop_count,
            "payloads": ui_payload_rows,
            "overall_results": str(REPORTS_DIR / "overall_crop_results.csv"),
            "overall_model_comparison": str(overall_model_comparison_path),
            "overall_model_ranking": str(overall_model_ranking_path),
            "architecture_doc": docs_artifacts["architecture_doc"],
            "pipeline_doc": docs_artifacts["pipeline_doc"],
        },
        REPORTS_DIR / "ui_payload" / "dashboard_payload.json",
    )

    status_lines = [
        "=== Full Pipeline Run Summary ===",
        f"Crops discovered: {len(crop_groups)}",
        f"Crops processed: {processed_crop_count}",
        f"Raw inputs classified into files: {len(cropwise_raw_exports)}",
        f"Overall results table: {REPORTS_DIR / 'overall_crop_results.csv'}",
        f"Overall model comparison: {overall_model_comparison_path}",
        f"Overall model ranking: {overall_model_ranking_path}",
        f"Architecture doc: {docs_artifacts['architecture_doc']}",
        f"Methodology doc: {docs_artifacts['pipeline_doc']}",
        f"Contribution note: {DOCS_DIR / 'paper_contribution_analysis.md'}",
    ]
    save_text_report("\n".join(status_lines), REPORTS_DIR / "full_pipeline_status.txt")

    return {
        "crops_discovered": len(crop_groups),
        "crops_processed": processed_crop_count,
        "raw_classified_files": cropwise_raw_exports,
        "results_table": str(REPORTS_DIR / "overall_crop_results.csv"),
        "overall_model_comparison": str(overall_model_comparison_path),
        "overall_model_ranking": str(overall_model_ranking_path),
        "architecture_doc": docs_artifacts["architecture_doc"],
        "pipeline_doc": docs_artifacts["pipeline_doc"],
        "status_report": str(REPORTS_DIR / "full_pipeline_status.txt"),
        "summary_report": str(REPORTS_DIR / "data_summary.txt"),
    }


def main() -> None:
    results = run_full_project_workflow()
    print("Full greenhouse forecasting pipeline completed.")
    print(f"Crops discovered: {results['crops_discovered']}")
    print(f"Crops processed: {results['crops_processed']}")
    print(f"Results table: {results['results_table']}")
    print(f"Status report: {results['status_report']}")


if __name__ == "__main__":
    main()
