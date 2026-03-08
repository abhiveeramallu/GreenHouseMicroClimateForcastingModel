"""
Project: Microclimate Forecasting System using Machine Learning
Purpose: Centralized configuration constants and file paths.
Module Group: Shared configuration
DFD Connection: Provides consistent IO paths between Data Management, ML Forecasting,
Decision & Control Simulation, and Visualization & Reporting modules.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "microclimate_temperature_history.csv"
RAW_MULTI_CROP_PATH = PROJECT_ROOT / "data" / "raw" / "greenhouse_crop_growth_timeseries.csv"
RAW_CROP_DIR = PROJECT_ROOT / "data" / "raw" / "crop_samples"
RAW_EXTERNAL_KAGGLE_PATH = PROJECT_ROOT / "data" / "raw" / "external_kaggle" / "greenhouse_plant_growth_metrics.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_CROPWISE_DIR = PROCESSED_DIR / "cropwise"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

DEFAULT_TIME_COLUMN = "timestamp"
DEFAULT_CROP_COLUMN = "crop_type"
DEFAULT_TARGET_COLUMN = "indoor_temperature_c"
DEFAULT_SEQUENCE_LENGTH = 6
DEFAULT_TEST_RATIO = 0.2
DEFAULT_MIN_SAMPLES_PER_CROP = 12
DEFAULT_LSTM_EPOCHS = 30
DEFAULT_LSTM_BATCH_SIZE = 16

DEFAULT_LOW_THRESHOLD_C = 22.0
DEFAULT_HIGH_THRESHOLD_C = 29.0
DEFAULT_SPRAY_THRESHOLD_C = 31.0
