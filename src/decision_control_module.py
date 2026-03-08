"""
Project: Microclimate Forecasting System using Machine Learning
Purpose: Simulate threshold-based greenhouse cooling actions from forecasted temperature.
Module Group: Decision & Control Simulation Module
DFD Connection: Consumes predicted temperatures from ML Forecasting Module and
emit simulated control actions (fan/spray) for analysis.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd


def evaluate_temperature_thresholds(
    predicted_temperature: float,
    high_threshold: float,
    low_threshold: float,
    spray_threshold: float,
) -> Dict[str, object]:
    """
    Determine greenhouse cooling action state from forecasted temperature.

    Inputs:
    - predicted_temperature: forecasted air temperature (C)
    - high_threshold: fan activation threshold
    - low_threshold: lower comfort threshold
    - spray_threshold: spray + fan activation threshold

    Output:
    - dictionary with action flags and descriptive state
    """

    if predicted_temperature >= spray_threshold:
        return {
            "action": "fan_and_spray_on",
            "fan_on": 1,
            "spray_on": 1,
            "status_note": "critical_high_temperature",
        }
    if predicted_temperature >= high_threshold:
        return {
            "action": "fan_on",
            "fan_on": 1,
            "spray_on": 0,
            "status_note": "high_temperature",
        }
    if predicted_temperature <= low_threshold:
        return {
            "action": "cooling_off",
            "fan_on": 0,
            "spray_on": 0,
            "status_note": "below_cooling_range",
        }
    return {
        "action": "idle",
        "fan_on": 0,
        "spray_on": 0,
        "status_note": "optimal_range",
    }


def simulate_fan_spray_actions(
    forecast_df: pd.DataFrame,
    high_threshold: float,
    low_threshold: float,
    spray_threshold: float,
    timestamp_column: str = "timestamp",
    crop_column: str = "crop_type",
    prediction_column: str = "predicted_temperature_c",
) -> pd.DataFrame:
    """
    Simulate fan/spray control actions over a forecast horizon.

    Inputs:
    - forecast_df: DataFrame containing timestamp and predicted temperatures
    - high_threshold: fan activation threshold
    - low_threshold: lower comfort threshold
    - spray_threshold: spray activation threshold
    - timestamp_column: timestamp field
    - crop_column: crop label field
    - prediction_column: forecasted temperature field

    Output:
    - DataFrame with action simulation timeline
    """

    required_columns = {timestamp_column, prediction_column}
    missing = [col for col in required_columns if col not in forecast_df.columns]
    if missing:
        raise ValueError(f"Missing required forecast columns: {missing}")

    events = []
    for row in forecast_df.itertuples(index=False):
        row_map = row._asdict()
        decision = evaluate_temperature_thresholds(
            predicted_temperature=float(row_map[prediction_column]),
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            spray_threshold=spray_threshold,
        )
        events.append(
            {
                timestamp_column: row_map[timestamp_column],
                crop_column: row_map.get(crop_column, "mixed_crop"),
                "predicted_temperature_c": float(row_map[prediction_column]),
                "action": decision["action"],
                "fan_on": decision["fan_on"],
                "spray_on": decision["spray_on"],
                "status_note": decision["status_note"],
            }
        )

    return pd.DataFrame(events)


def summarize_control_actions(control_df: pd.DataFrame) -> Dict[str, int]:
    """
    Summarize counts of each action category.

    Inputs:
    - control_df: action simulation dataframe

    Output:
    - dictionary with aggregate action counts
    """

    if control_df.empty:
        return {"total_records": 0, "fan_on_events": 0, "spray_on_events": 0}

    return {
        "total_records": int(len(control_df)),
        "fan_on_events": int(control_df["fan_on"].sum()),
        "spray_on_events": int(control_df["spray_on"].sum()),
        "fan_and_spray_events": int((control_df["action"] == "fan_and_spray_on").sum()),
        "idle_events": int((control_df["action"] == "idle").sum()),
    }
