# Microclimate Forecasting System using Machine Learning

## Project Overview
This project forecasts greenhouse microclimate temperature from historical CSV data and simulates control actions (`fan` / `spray`) from threshold logic.

The implementation is software-only and CSV-based:
- Python
- Pandas, NumPy
- TensorFlow/Keras (LSTM primary model when available)
- Parallel secondary models: Random Forest, Gradient Boosting, Linear Regression baseline, optional XGBoost
- Hybrid coordination layer: dynamic weighted ensemble
- Matplotlib/Seaborn
- Jupyter Notebook support

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

## How to Run
Use your project virtual environment (already containing pandas/numpy/matplotlib/seaborn):

```bash
cd /Users/vabhiram/Desktop/softwareeng_project
source .venv_new/bin/activate
python -m src.pipeline_data_prep
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
