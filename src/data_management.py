"""
Project: Microclimate Forecasting System using Machine Learning
Purpose: Implement the complete Data Management Module for CSV-based greenhouse datasets.
Module Group: Data Management Module
DFD Connection: Transforms raw historical CSV data into cleaned, normalized, and
LSTM-ready sequences consumed by the ML Forecasting Module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import json
import re

import numpy as np
import pandas as pd


@dataclass
class ScalingParameters:
    """Stores min-max values for each feature to support inverse scaling later."""

    minimums: Dict[str, float]
    maximums: Dict[str, float]


COLUMN_ALIASES = {
    "timestamp": [
        "timestamp",
        "time",
        "date",
        "datetime",
        "recorded_at",
        "observation_time",
    ],
    "crop_type": [
        "crop",
        "crop_type",
        "crop_name",
        "plant",
        "plant_type",
        "vegetable",
        "species",
        "class",
    ],
    "indoor_temperature_c": [
        "indoor_temperature_c",
        "indoor_temperature",
        "air_temperature_c",
        "air_temperature",
        "temperature_c",
        "temperature",
        "temp_c",
        "temp",
        "achp",
    ],
    "canopy_temperature_c": ["canopy_temperature_c", "canopy_temperature", "leaf_temperature", "adwr"],
    "ambient_temperature_c": ["ambient_temperature_c", "ambient_temperature", "outside_temperature", "ard"],
    "humidity_pct": ["humidity_pct", "humidity", "relative_humidity", "rh", "phr"],
    "soil_moisture_pct": ["soil_moisture_pct", "soil_moisture", "soil_water_content", "awwgv"],
    "co2_ppm": ["co2_ppm", "co2", "carbon_dioxide", "anpl"],
    "light_lux": ["light_lux", "light_intensity", "lux", "sunlight_lux", "alap"],
    "days_after_planting": ["days_after_planting", "days", "dap", "day"],
}


def _normalize_column_name(column_name: str) -> str:
    text = column_name.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _slugify(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", text.lower().strip())
    return normalized.strip("_") or "unknown_crop"


def discover_csv_files(input_path: str | Path) -> List[Path]:
    """
    Discover CSV files from a file path or directory path.

    Inputs:
    - input_path: file or directory path

    Output:
    - List of discovered CSV file paths
    """

    path_obj = Path(input_path)
    if not path_obj.exists():
        return []
    if path_obj.is_file() and path_obj.suffix.lower() == ".csv":
        return [path_obj]
    if path_obj.is_dir():
        return sorted(path_obj.glob("*.csv"))
    return []


def infer_crop_from_filename(file_path: str | Path) -> str:
    """
    Infer crop name from filename when crop column is absent.

    Inputs:
    - file_path: source CSV path

    Output:
    - inferred crop label
    """

    stem = Path(file_path).stem.lower()
    if "tomato" in stem:
        return "tomato"
    if "chilli" in stem or "chili" in stem:
        return "green_chilli"
    if "brinjal" in stem or "eggplant" in stem:
        return "brinjal"
    if "lady" in stem or "okra" in stem:
        return "ladysfinger"
    return "mixed_crop"


def _apply_alias_renaming(dataset: pd.DataFrame) -> pd.DataFrame:
    renamed = dataset.copy()
    normalized_map = {col: _normalize_column_name(col) for col in renamed.columns}
    reverse_map = {value: key for key, value in normalized_map.items()}

    rename_dict: Dict[str, str] = {}
    for standard_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias_key = _normalize_column_name(alias)
            if alias_key in reverse_map:
                original_column = reverse_map[alias_key]
                rename_dict[original_column] = standard_name
                break

    return renamed.rename(columns=rename_dict)


def standardize_greenhouse_schema(
    dataset: pd.DataFrame,
    source_file: str | Path | None = None,
    time_column: str = "timestamp",
    crop_column: str = "crop_type",
) -> pd.DataFrame:
    """
    Map different greenhouse CSV schemas into a unified project schema.

    Inputs:
    - dataset: raw CSV DataFrame
    - source_file: optional source file path for crop inference
    - time_column: canonical time column name
    - crop_column: canonical crop label column name

    Output:
    - standardized DataFrame
    """

    standardized = _apply_alias_renaming(dataset)

    if crop_column not in standardized.columns:
        inferred_crop = infer_crop_from_filename(source_file) if source_file else "mixed_crop"
        standardized[crop_column] = inferred_crop

    if time_column not in standardized.columns:
        if "days_after_planting" in standardized.columns:
            days = pd.to_numeric(standardized["days_after_planting"], errors="coerce").fillna(0)
            standardized[time_column] = pd.Timestamp("2025-01-01") + pd.to_timedelta(days, unit="D")
        else:
            standardized[time_column] = pd.date_range(
                start="2025-01-01",
                periods=len(standardized),
                freq="h",
            )

    standardized[time_column] = pd.to_datetime(standardized[time_column], errors="coerce")
    standardized[crop_column] = standardized[crop_column].astype(str).str.strip().str.lower()
    standardized[crop_column] = standardized[crop_column].replace({"": "mixed_crop"})

    for col in standardized.columns:
        if col not in {time_column, crop_column}:
            standardized[col] = pd.to_numeric(standardized[col], errors="coerce")

    return standardized


def load_and_merge_datasets(
    dataset_inputs: Sequence[str | Path],
    time_column: str = "timestamp",
    crop_column: str = "crop_type",
) -> pd.DataFrame:
    """
    Load one or many CSV sources and merge into one standardized DataFrame.

    Inputs:
    - dataset_inputs: list of files or directories
    - time_column: canonical time column
    - crop_column: canonical crop label column

    Output:
    - merged standardized DataFrame
    """

    csv_files: List[Path] = []
    for item in dataset_inputs:
        csv_files.extend(discover_csv_files(item))

    if not csv_files:
        raise ValueError("No CSV files found in provided dataset inputs.")

    merged_frames: List[pd.DataFrame] = []
    for csv_file in csv_files:
        frame = pd.read_csv(csv_file)
        merged_frames.append(
            standardize_greenhouse_schema(
                dataset=frame,
                source_file=csv_file,
                time_column=time_column,
                crop_column=crop_column,
            )
        )

    merged = pd.concat(merged_frames, ignore_index=True)
    required_temperature_columns = [col for col in merged.columns if "temperature" in col]
    if not required_temperature_columns:
        raise ValueError("Merged dataset does not contain any temperature columns.")

    return merged


def missing_value_report(dataset: pd.DataFrame) -> pd.Series:
    """
    Calculate missing values per column.

    Input:
    - dataset: any DataFrame

    Output:
    - Series of missing counts by column
    """

    return dataset.isna().sum()


def clean_dataset(dataset: pd.DataFrame, time_column: str = "timestamp") -> pd.DataFrame:
    """
    Clean the dataset by sorting time, removing duplicates, and imputing missing values.

    Inputs:
    - dataset: raw DataFrame
    - time_column: timestamp column name

    Output:
    - Cleaned DataFrame in chronological order with no missing values
    """

    cleaned = dataset.copy()
    crop_column = "crop_type" if "crop_type" in cleaned.columns else None

    cleaned = cleaned.dropna(subset=[time_column])

    if crop_column:
        cleaned = cleaned.sort_values(by=[crop_column, time_column]).drop_duplicates(
            subset=[crop_column, time_column], keep="first"
        )
    else:
        cleaned = cleaned.sort_values(by=time_column).drop_duplicates(subset=[time_column], keep="first")
    cleaned = cleaned.reset_index(drop=True)

    numeric_columns = [col for col in cleaned.columns if col not in {time_column, crop_column}]
    for col in numeric_columns:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    if crop_column:
        cleaned[numeric_columns] = cleaned.groupby(crop_column)[numeric_columns].transform(
            lambda part: part.interpolate(method="linear", limit_direction="both")
        )
        cleaned[numeric_columns] = cleaned.groupby(crop_column)[numeric_columns].transform(
            lambda part: part.fillna(part.median())
        )
    else:
        cleaned[numeric_columns] = cleaned[numeric_columns].interpolate(
            method="linear",
            limit_direction="both",
        )
        cleaned[numeric_columns] = cleaned[numeric_columns].fillna(cleaned[numeric_columns].median())

    cleaned[numeric_columns] = cleaned[numeric_columns].fillna(cleaned[numeric_columns].median())

    # Soft outlier clipping improves robustness for later LSTM training.
    for col in numeric_columns:
        lower = cleaned[col].quantile(0.01)
        upper = cleaned[col].quantile(0.99)
        cleaned[col] = cleaned[col].clip(lower=lower, upper=upper)

    return cleaned


def split_by_crop(
    dataset: pd.DataFrame,
    crop_column: str = "crop_type",
    min_samples_per_crop: int = 12,
) -> Dict[str, pd.DataFrame]:
    """
    Split merged dataset into crop-wise DataFrames.

    Inputs:
    - dataset: cleaned merged DataFrame
    - crop_column: crop identifier column
    - min_samples_per_crop: minimum samples required to keep a crop subset

    Output:
    - dictionary {crop_name: crop_dataframe}
    """

    if crop_column not in dataset.columns:
        return {"mixed_crop": dataset.copy().reset_index(drop=True)}

    crop_groups: Dict[str, pd.DataFrame] = {}
    for crop_name, crop_df in dataset.groupby(crop_column):
        if len(crop_df) >= min_samples_per_crop:
            crop_groups[str(crop_name)] = crop_df.sort_values("timestamp").reset_index(drop=True)
    return crop_groups


def export_cropwise_datasets(
    dataset: pd.DataFrame,
    output_dir: str | Path,
    crop_column: str = "crop_type",
) -> Dict[str, str]:
    """
    Export merged dataset into separate crop-specific CSV files.

    Inputs:
    - dataset: cleaned merged DataFrame
    - output_dir: destination directory
    - crop_column: crop identifier column

    Output:
    - mapping {crop_name: output_csv_path}
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if crop_column not in dataset.columns:
        single_output = output_path / "mixed_crop_dataset.csv"
        dataset.to_csv(single_output, index=False)
        return {"mixed_crop": str(single_output)}

    exports: Dict[str, str] = {}
    for crop_name, crop_df in dataset.groupby(crop_column):
        crop_slug = _slugify(str(crop_name))
        file_path = output_path / f"{crop_slug}_dataset.csv"
        crop_df.to_csv(file_path, index=False)
        exports[str(crop_name)] = str(file_path)
    return exports


def select_forecasting_features(
    dataset: pd.DataFrame,
    preferred_target: str = "indoor_temperature_c",
    time_column: str = "timestamp",
    crop_column: str = "crop_type",
) -> Tuple[pd.DataFrame, str]:
    """
    Select model features for greenhouse microclimate forecasting.

    Inputs:
    - dataset: cleaned DataFrame
    - preferred_target: desired prediction target column
    - time_column: timestamp column name
    - crop_column: crop identifier column

    Outputs:
    - DataFrame containing metadata columns + selected feature columns
    - Resolved target column name
    """

    if preferred_target in dataset.columns:
        target_column = preferred_target
    else:
        temperature_columns = [col for col in dataset.columns if "temperature" in col.lower() or "temp" in col.lower()]
        if not temperature_columns:
            raise ValueError("No temperature-related columns found for forecasting.")
        target_column = temperature_columns[0]

    candidate_features = [
        target_column,
        "canopy_temperature_c",
        "ambient_temperature_c",
        "humidity_pct",
        "soil_moisture_pct",
        "co2_ppm",
        "light_lux",
    ]
    available_features = [col for col in candidate_features if col in dataset.columns]
    available_features = list(dict.fromkeys(available_features))

    if not available_features:
        raise ValueError("No temperature-related columns found for forecasting.")

    # If canonical aliases are limited, add additional numeric predictors to improve model context.
    numeric_candidates = [
        col
        for col in dataset.columns
        if col not in {time_column, crop_column, target_column} and pd.api.types.is_numeric_dtype(dataset[col])
    ]
    for col in numeric_candidates:
        if col not in available_features:
            available_features.append(col)

    selected_columns = [time_column]
    if crop_column in dataset.columns:
        selected_columns.append(crop_column)
    selected_columns.extend(available_features)

    selected = dataset[selected_columns].copy()
    return selected, target_column


def min_max_scale_features(
    dataset: pd.DataFrame,
    time_column: str = "timestamp",
    protected_columns: Iterable[str] = ("crop_type",),
) -> Tuple[pd.DataFrame, ScalingParameters]:
    """
    Apply min-max normalization to feature columns in [0, 1].

    Inputs:
    - dataset: feature DataFrame (time + numerical predictors)
    - time_column: timestamp column name

    Outputs:
    - Scaled DataFrame
    - ScalingParameters (minimum and maximum per feature)
    """

    scaled = dataset.copy()
    protected = set(protected_columns)
    feature_columns = [col for col in scaled.columns if col not in {time_column} | protected]

    minimums: Dict[str, float] = {}
    maximums: Dict[str, float] = {}

    for col in feature_columns:
        min_val = float(scaled[col].min())
        max_val = float(scaled[col].max())
        minimums[col] = min_val
        maximums[col] = max_val

        feature_range = max_val - min_val
        if feature_range == 0:
            scaled[col] = 0.0
        else:
            scaled[col] = (scaled[col] - min_val) / feature_range

    return scaled, ScalingParameters(minimums=minimums, maximums=maximums)


def inverse_min_max_scale(
    values: np.ndarray,
    feature_name: str,
    scaling_parameters: ScalingParameters,
) -> np.ndarray:
    """
    Convert scaled values back to original feature scale.

    Inputs:
    - values: scaled values in [0, 1]
    - feature_name: feature key used during scaling
    - scaling_parameters: ScalingParameters with min/max metadata

    Output:
    - values in original scale
    """

    min_val = scaling_parameters.minimums[feature_name]
    max_val = scaling_parameters.maximums[feature_name]
    return values * (max_val - min_val) + min_val


def train_test_split_time_series(
    dataset: pd.DataFrame,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/test split for time-series forecasting.

    Inputs:
    - dataset: sequential DataFrame
    - test_ratio: fraction reserved for test set

    Outputs:
    - train DataFrame
    - test DataFrame
    """

    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1.")

    split_index = int(len(dataset) * (1 - test_ratio))
    split_index = max(1, min(split_index, len(dataset) - 1))

    train_set = dataset.iloc[:split_index].reset_index(drop=True)
    test_set = dataset.iloc[split_index:].reset_index(drop=True)
    return train_set, test_set


def generate_lstm_sequences(
    dataset: pd.DataFrame,
    target_column: str,
    sequence_length: int,
    time_column: str = "timestamp",
    protected_columns: Iterable[str] = ("crop_type",),
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """
    Convert tabular data into sliding-window LSTM sequences.

    Methodology note:
    This follows the common research-paper approach in microclimate forecasting:
    past N timesteps are used to predict the next target timestep.

    Inputs:
    - dataset: scaled DataFrame in chronological order
    - target_column: column to predict
    - sequence_length: number of historical timesteps per sample
    - time_column: timestamp column name

    Outputs:
    - X: shape (samples, sequence_length, num_features)
    - y: shape (samples,)
    - y_timestamps: list of timestamps for each y sample
    """

    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1.")

    protected = set(protected_columns)
    feature_columns = [col for col in dataset.columns if col not in {time_column} | protected]
    if target_column not in feature_columns:
        raise ValueError(f"Target column '{target_column}' not found in features.")

    target_index = feature_columns.index(target_column)
    values = dataset[feature_columns].to_numpy(dtype=np.float32)
    # Fill NaN values with 0 to prevent LSTM from producing NaN predictions
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    timestamps = dataset[time_column].tolist()

    if len(values) <= sequence_length:
        empty_x = np.empty((0, sequence_length, len(feature_columns)), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.float32)
        return empty_x, empty_y, []

    x_samples: List[np.ndarray] = []
    y_samples: List[float] = []
    y_timestamps: List[pd.Timestamp] = []

    for start_idx in range(len(values) - sequence_length):
        end_idx = start_idx + sequence_length
        x_samples.append(values[start_idx:end_idx, :])
        y_samples.append(float(values[end_idx, target_index]))
        y_timestamps.append(timestamps[end_idx])

    x_array = np.array(x_samples, dtype=np.float32)
    y_array = np.array(y_samples, dtype=np.float32)
    return x_array, y_array, y_timestamps


def save_scaling_parameters(parameters: ScalingParameters, output_path: str | Path) -> None:
    """
    Persist min-max scaling metadata for reproducibility.

    Inputs:
    - parameters: ScalingParameters dataclass
    - output_path: JSON file path

    Output:
    - Writes JSON file to disk
    """

    payload = {
        "minimums": parameters.minimums,
        "maximums": parameters.maximums,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
