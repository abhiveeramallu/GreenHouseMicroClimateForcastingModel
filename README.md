# Microclimate Forecasting System using Machine Learning

## Project Overview
This project forecasts greenhouse microclimate temperature from historical CSV data and simulates control actions (`fan` / `spray`) from threshold logic.

The implementation is software-only and CSV-based:
- Python
- Pandas, NumPy
- PyTorch LSTM (primary model)
- Parallel secondary models: Random Forest, Gradient Boosting, Linear Regression baseline
- Hybrid coordination layer: dynamic weighted ensemble
- Matplotlib/Seaborn

## Proposed System

The proposed system implements a hybrid machine learning-based microclimate forecasting framework designed to predict greenhouse temperature and simulate environmental control decisions.

### System Architecture

```mermaid
flowchart LR
    A[Data Management Module<br/>CSV ingestion, cleaning, scaling, sequences]
    B[Machine Learning Forecasting Module<br/>LSTM + parallel ML models]
    C[Hybrid Ensemble Coordination Module<br/>Dynamic weighted prediction]
    D[Decision and Control Simulation Module<br/>Threshold rule engine]
    E[Visualization and Reporting Module<br/>Dashboard + plots + reports]

    A --> B --> C --> D --> E
```

### Primary Modules

The system architecture is composed of five primary modules:

1. **Data Management Module**
   - Loads historical greenhouse datasets from CSV files
   - Performs preprocessing: cleaning, normalization, feature selection
   - Generates LSTM-ready sequences using sliding-window approach
   - Splits data by crop types (SA, SB, SC, TA, TB, TC)

2. **Machine Learning Forecasting Module**
   - Trains multiple prediction models in parallel:
     - Linear Regression
     - Random Forest Regressor
     - Gradient Boosting Regressor
     - LSTM Neural Network (PyTorch)
   - Each model produces independent temperature predictions
   - Evaluates models using MAE, RMSE, and R² metrics

3. **Hybrid Ensemble Coordination Module**
   - Combines predictions using dynamic inverse-RMSE weighted ensemble
   - Weight calculation: `weight_m = 1 / RMSE_m`
   - Normalized weight: `weight_m = weight_m / Σ(weight_all)`
   - Final prediction: `Final_Prediction = Σ(weight_m × prediction_m)`
   - Assigns greater importance to models with lower prediction error

4. **Decision and Control Simulation Module**
   - Applies threshold-based control logic to predicted temperatures
   - Low threshold (18°C): Turn ON fan
   - High threshold (28°C): Turn ON spray
   - Normal range (18-28°C): All OFF
   - Generates fan/spray action timeline for each crop

5. **Visualization and Reporting Module**
   - Creates interactive dashboard with temperature predictions
   - Displays model comparison charts and performance metrics
   - Shows control action timelines
   - Exports results to CSV, JSON, and figure formats

### System Capabilities

The system processes historical greenhouse datasets, performs preprocessing and sequence generation, trains multiple machine learning models, combines their predictions using a hybrid ensemble strategy, and generates actuator control decisions.

The architecture enables accurate prediction of greenhouse microclimate conditions while maintaining modular extensibility for future enhancements such as plant growth intelligence and IoT sensor integration.

## Module-wise Inputs and Outputs

### 1. Data Management Module

**Inputs:**
- `dataset_inputs`: List of CSV file paths or directories
- `time_column`: Timestamp column name (default: "timestamp")
- `crop_column`: Crop identifier column (default: "crop_type")
- `sequence_length`: Number of historical timesteps for LSTM (default: 6)
- `test_ratio`: Fraction for test set (default: 0.2)

**Outputs:**
- `X_train`: NumPy array (samples, sequence_length, num_features) - LSTM training sequences
- `y_train`: NumPy array (samples,) - Training targets
- `X_test`: NumPy array (samples, sequence_length, num_features) - LSTM test sequences
- `y_test`: NumPy array (samples,) - Test targets
- `crop_groups`: Dictionary {crop_name: crop_dataframe} - Data split by crop type
- `scaling_parameters`: JSON file with min/max values for each feature
- `classified_inputs/*.csv`: Per-crop CSV files

### 2. Machine Learning Forecasting Module

**Inputs:**
- `x_train`: Training sequences (samples, sequence_length, num_features)
- `y_train`: Training targets (samples,)
- `x_val`: Validation sequences
- `y_val`: Validation targets
- `model_dir`: Directory for saving model artifacts
- `model_tag`: Prefix for saved files (e.g., "sa", "sb")
- `epochs`: Training epochs (default: 30)
- `batch_size`: Batch size (default: 16)

**Outputs:**
- `TrainedForecaster`: Dataclass containing:
  - `backend`: Model type ("lstm" or "linear")
  - `model`: Trained model object (PyTorch LSTM or Linear regressor)
  - `history`: Training history dict with loss values
  - `fallback_reason`: Optional explanation if LSTM unavailable
- `models/<crop>/*_lstm.pt`: PyTorch LSTM model weights
- `models/<crop>/*_linear_weights.npy`: Linear model weights
- `predictions`: NumPy array of temperature forecasts

### 3. Hybrid Ensemble Coordination Module

**Inputs:**
- `y_validation`: Validation ground truth array
- `validation_predictions`: Dictionary {model_name: prediction_array}
- `test_predictions`: Dictionary {model_name: prediction_array}
- `preferred_model`: Model to prioritize (default: "lstm")

**Outputs:**
- `CoordinationResult`: Dataclass containing:
  - `final_prediction`: Weighted ensemble prediction array
  - `weights`: Dictionary {model_name: normalized_weight}
  - `validation_rmse`: Dictionary {model_name: rmse_value}
  - `method`: Ensemble method name ("dynamic_inverse_rmse_weighted_ensemble")
- `comparison_df`: DataFrame with model rankings (MAE, RMSE, R²)
- `model_comparison_artifacts.csv`: Model metrics in CSV format

**Formulas:**
```
Weight calculation: weight_m = 1 / RMSE_m
Normalized weight: weight_m = weight_m / Σ(weight_all)
Final prediction: Final_Prediction = Σ(weight_m × prediction_m)
```

### 4. Decision and Control Simulation Module

**Inputs:**
- `forecast_df`: DataFrame with predicted temperatures
- `high_threshold`: Fan activation threshold (default: 29°C)
- `low_threshold`: Lower comfort threshold (default: 22°C)
- `spray_threshold`: Spray activation threshold (default: 31°C)
- `timestamp_column`: Time field name
- `crop_column`: Crop label field
- `prediction_column`: Forecasted temperature field

**Outputs:**
- `control_df`: DataFrame with columns:
  - `timestamp`: Time of prediction
  - `crop_type`: Crop identifier
  - `predicted_temperature_c`: Forecasted temperature
  - `action`: Control action ("fan_on", "fan_and_spray_on", "idle", "cooling_off")
  - `fan_on`: Binary flag (0 or 1)
  - `spray_on`: Binary flag (0 or 1)
  - `status_note`: Descriptive state
- `control_summary`: Dictionary with aggregate counts:
  - `total_records`: Total predictions
  - `fan_on_events`: Count of fan activations
  - `spray_on_events`: Count of spray activations
  - `fan_and_spray_events`: Count of both activated
  - `idle_events`: Count of no action needed

**Threshold Logic:**
- Temperature ≥ 31°C: fan_and_spray_on (critical high)
- Temperature ≥ 29°C: fan_on (high temperature)
- Temperature ≤ 22°C: cooling_off (below cooling range)
- 22°C < Temperature < 29°C: idle (optimal range)

### 5. Visualization and Reporting Module

**Inputs:**
- `predictions_df`: DataFrame with actual and predicted temperatures
- `control_df`: Action timeline DataFrame
- `comparison_df`: Model comparison metrics
- `crop_name`: Crop identifier for report context
- `output_paths`: File paths for TXT, JSON, CSV outputs

**Outputs:**
- `model_performance_report`: Dictionary with:
  - `mae`: Mean Absolute Error
  - `rmse`: Root Mean Square Error
  - `mape_pct`: Mean Absolute Percentage Error
  - `r2`: R-squared value
  - `sample_count`: Number of predictions
  - `backend`: Model backend used
- `control_simulation_report`: Dictionary with action counts
- `overall_model_ranking.csv`: Aggregated model rankings across all crops
- `overall_crop_results.csv`: Per-crop summary statistics
- `ui_payload.json`: Dashboard-ready JSON with predictions and actions
- `*.png` figures: Actual vs predicted plots, model comparison bars, control timelines

## Research Reference Alignment
- Paper reference integrated: DOI `10.1038/s41598-025-15615-3`
- Alignment note: `/Users/vabhiram/Desktop/softwareeng_project/docs/paper_alignment_s41598-025-15615-3.md`
- Scope reminder: this implementation is strictly software-side; UI consumes generated outputs.

## Current Status
Full implementation completed for:
1. Data Management Module
2. Machine Learning Forecasting Module (LSTM + parallel ML extension)
3. Decision & Control Simulation Module
4. Visualization & Reporting Module (including plant-growth intelligence extension)

## Dataset Inputs
Supported input sources (auto-detected if present):
- `/Users/vabhiram/Desktop/softwareeng_project/data/raw/external_kaggle/greenhouse_plant_growth_metrics.csv`
- `/Users/vabhiram/Desktop/softwareeng_project/data/raw/greenhouse_crop_growth_timeseries.csv`
- `/Users/vabhiram/Desktop/softwareeng_project/data/raw/crop_samples/*.csv`
- `/Users/vabhiram/Desktop/softwareeng_project/data/raw/microclimate_temperature_history.csv`
- `/Users/vabhiram/Desktop/softwareeng_project/data/raw/demo_all_cases_control_logic.csv` (demo for all fan/spray states)

### Kaggle Input Added
Your provided zip is integrated at:
- `/Users/vabhiram/Desktop/softwareeng_project/data/raw/external_kaggle/greenhouse_plant_growth_metrics.csv`

## Pipeline Flow (DFD-Aligned)
1. **Data Management**
   - Load one/multiple CSV files
   - Standardize schema (supports heterogeneous greenhouse column names)
   - Clean missing values and duplicates
   - Split into crop/class subsets
   - Scale features and generate LSTM sequences
2. **ML Forecasting**
   - Train LSTM model (primary backend)
   - Train independent secondary models (RandomForest, GradientBoosting, LinearRegression, optional XGBoost)
   - Evaluate model-wise MAE/RMSE/R2
   - Coordinate predictions using dynamic inverse-RMSE weighted ensemble
3. **Decision & Control Simulation**
   - Use predicted temperature
   - Apply threshold logic (`low`, `high`, `spray`)
   - Generate `fan/spray` action timeline
4. **Visualization & Reporting**
   - Data summary and preprocessing plots
   - Actual vs LSTM/ML/Hybrid plots + model comparison bars
   - Control action timeline
   - Per-crop metrics, aggregate model ranking, and plant-growth intelligence reports

## How to Train and Run

### Quick Start (Recommended)

```bash
# 1. Go to project folder
cd /Users/vabhiram/Desktop/softwareeng_project

# 2. Activate virtual environment
source .venv_new/bin/activate

# 3. Train models and run full pipeline
PYTHONPATH=/Users/vabhiram/Desktop/softwareeng_project python -m src.pipeline_data_prep

# 4. Run hybrid model evaluation
PYTHONPATH=/Users/vabhiram/Desktop/softwareeng_project python -m src.run_hybrid_evaluation

# 5. Start dashboard UI
PYTHONPATH=/Users/vabhiram/Desktop/softwareeng_project python -m src.dashboard_server --port 8080

# 6. Open browser: http://127.0.0.1:8080
```

### Training Process Details

#### Step 1: Data Preparation
- **Input**: Greenhouse CSV files with temperature and plant metrics
- **Processing**: 
  - Load and merge multiple CSV sources
  - Clean missing values and remove duplicates
  - Split data by crop types (SA, SB, SC, TA, TB, TC)
  - Scale features using Min-Max normalization
  - Generate LSTM sequences (6 timesteps, 3 features)

#### Step 2: Model Training
- **Primary Model**: LSTM (Long Short-Term Memory) neural network
  - Architecture: 50 LSTM units → Dense layer
  - Training: 50 epochs, batch size 32
  - Optimizer: Adam, Loss: MSE
- **Secondary Models**:
  - Random Forest (100 trees)
  - Gradient Boosting (100 estimators)
  - Linear Regression (baseline)
  - Linear Autoregressive (baseline)
- **Hybrid Coordination**: Dynamic weighted ensemble based on inverse RMSE

#### Step 3: Evaluation Metrics
- **MAE** (Mean Absolute Error): Lower is better
- **RMSE** (Root Mean Square Error): Lower is better  
- **R²** (R-squared): Higher is better (closer to 1.0)

#### Step 4: Control Simulation
- **Threshold Logic**:
  - Low threshold: 18°C → Turn ON fan
  - High threshold: 28°C → Turn ON spray
  - Normal range (18-28°C): All OFF
- **Outputs**: Fan/spray action timeline for each crop

### Data Requirements

#### Required Columns in CSV files:
- `timestamp` or `time`: Time series data
- `temperature` or `indoor_temperature_c`: Target variable
- `humidity` or similar: Environmental feature
- `light_intensity` or similar: Environmental feature
- `crop` or `class`: Crop type identifier (optional)

#### Supported Data Sources:
1. **Kaggle Dataset**: `data/raw/external_kaggle/greenhouse_plant_growth_metrics.csv`
2. **Microclimate Data**: `data/raw/microclimate_temperature_history.csv`
3. **Crop Samples**: `data/raw/crop_samples/*.csv`
4. **Demo Data**: `data/raw/demo_all_cases_control_logic.csv`

### Model Performance (Latest Results)

| Rank | Model | MAE | RMSE | R² |
|------|-------|-----|-------|-----|
| 1 | LSTM | 1.98 | 2.31 | -0.008 |
| 2 | Random Forest | 2.02 | 2.33 | -0.019 |
| 3 | Linear Baseline | 2.03 | 2.33 | -0.025 |
| 4 | Linear Regression | 2.04 | 2.34 | -0.026 |
| 5 | Gradient Boosting | 2.03 | 2.34 | -0.027 |

### Troubleshooting

#### Common Issues:
1. **Port already in use**: Try different ports (8081, 8082, 9999)
2. **Module not found**: Ensure virtual environment is activated
3. **Memory issues**: Reduce batch size or sequence length in config
4. **TensorFlow errors**: Pipeline continues with secondary models

#### Commands to Fix Issues:
```bash
# Kill processes on ports (if needed)
pkill -f dashboard_server

# Use different port
python -m src.dashboard_server --port 8090

# Reinstall dependencies
pip install numpy pandas scikit-learn matplotlib seaborn torch

# Check data files
ls -la data/raw/
```

## Frontend UI (Recommended Presentation Mode)
Run one command to compute latest outputs and launch a browser dashboard:

```bash
cd /Users/vabhiram/Desktop/softwareeng_project
source .venv_new/bin/activate
python -m src.dashboard_server --port 8080
```

Open:
- `http://127.0.0.1:8080`

## Hybrid Evaluation Command
To run full pipeline plus hybrid ranking printout:

```bash
cd /Users/vabhiram/Desktop/softwareeng_project
source .venv_new/bin/activate
python -m src.run_hybrid_evaluation
```

UI includes:
- Actual vs predicted temperature chart
- Fixed 25-point chart window with horizontal timeline navigation
- Timeline playback for non-technical viewers
- Real-time fan state (`ON` rotating / `OFF` static)
- Real-time spray state (`ON` mist animation / `OFF` static)
- Current temperature, action status, and model metrics
- Built-in demo entries (`DEMO-*`) that guarantee all control logic cases are visible

## Main Outputs
Primary consumption mode is the frontend dashboard.

File outputs are still produced for reproducibility and audit:
- Per-crop classified inputs:
  - `/Users/vabhiram/Desktop/softwareeng_project/data/processed/cropwise/classified_inputs/*.csv`
- Per-crop model artifacts:
  - `/Users/vabhiram/Desktop/softwareeng_project/models/<crop>/`
- Per-crop predictions and actions:
  - `/Users/vabhiram/Desktop/softwareeng_project/data/processed/cropwise/<crop>/predictions.csv`
  - `/Users/vabhiram/Desktop/softwareeng_project/data/processed/cropwise/<crop>/control_actions.csv`
- Reports:
  - `/Users/vabhiram/Desktop/softwareeng_project/reports/data_summary.txt`
  - `/Users/vabhiram/Desktop/softwareeng_project/reports/overall_crop_results.csv`
  - `/Users/vabhiram/Desktop/softwareeng_project/reports/overall_model_comparison.csv`
  - `/Users/vabhiram/Desktop/softwareeng_project/reports/overall_model_ranking.csv`
  - `/Users/vabhiram/Desktop/softwareeng_project/reports/plant_growth_intelligence/*`
  - `/Users/vabhiram/Desktop/softwareeng_project/reports/full_pipeline_status.txt`
  - `/Users/vabhiram/Desktop/softwareeng_project/reports/ui_payload/dashboard_payload.json`
- Figures:
  - `/Users/vabhiram/Desktop/softwareeng_project/reports/figures/temperature_trend.png`
  - `/Users/vabhiram/Desktop/softwareeng_project/reports/figures/preprocessing_before_after.png`
  - `/Users/vabhiram/Desktop/softwareeng_project/reports/figures/rmse_by_crop.png`
  - `/Users/vabhiram/Desktop/softwareeng_project/reports/figures/plant_growth_intelligence/*.png`
  - `/Users/vabhiram/Desktop/softwareeng_project/reports/figures/cropwise/<crop>/*.png`
- Contribution note:
  - `/Users/vabhiram/Desktop/softwareeng_project/docs/paper_contribution_analysis.md`
- Architecture and methodology docs:
  - `/Users/vabhiram/Desktop/softwareeng_project/docs/hybrid_system_architecture.md`
  - `/Users/vabhiram/Desktop/softwareeng_project/docs/updated_ml_pipeline.md`

## Model Coordination Details
- Primary model: LSTM (TensorFlow/Keras)
- Secondary models: Random Forest, Gradient Boosting, Linear Regression, linear autoregressive baseline, optional XGBoost
- Coordinated output: dynamic weighted ensemble based on inverse validation RMSE
- LSTM remains in pipeline; if TensorFlow is unavailable, coordination continues with available secondary models and baseline predictors.
