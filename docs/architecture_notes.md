# Architecture Notes

## Project
Microclimate Forecasting System using Machine Learning

## DFD-Oriented Module Mapping
1. Data Management Module
   - Loads and standardizes one/many greenhouse CSV datasets.
   - Cleans missing values/outliers and creates crop-wise subsets.
   - Produces scaled time-series tensors (`X`, `y`) for forecasting models.
2. Machine Learning Forecasting Module
   - Trains LSTM model (primary) when TensorFlow is available.
   - Trains independent secondary models: Random Forest, Gradient Boosting, Linear Regression, optional XGBoost.
   - Retains linear autoregressive baseline for fallback and robustness.
   - Coordinates models via dynamic weighted hybrid ensemble.
   - Produces per-model evaluation tables (MAE, RMSE, R2).
3. Decision & Control Simulation Module
   - Applies threshold logic over forecasted temperature.
   - Simulates `fan` / `spray` actions over timeline.
4. Visualization & Reporting Module
   - Generates data quality, forecast, and action plots.
   - Produces per-crop metrics, aggregate reports, and UI payload JSON.
   - Adds plant-growth intelligence analytics and feature-importance reporting.

## Software-Only Scope
1. No physical IoT deployment in this implementation.
2. Inputs are historical CSV files only.
3. Outputs are files/figures/JSON designed for frontend UI integration.

## Paper Alignment
Reference paper DOI: `10.1038/s41598-025-15615-3`  
Detailed mapping: `/Users/vabhiram/Desktop/softwareeng_project/docs/paper_alignment_s41598-025-15615-3.md`

## Updated Architecture Diagram
`/Users/vabhiram/Desktop/softwareeng_project/docs/hybrid_system_architecture.md`

## Updated ML Methodology
`/Users/vabhiram/Desktop/softwareeng_project/docs/updated_ml_pipeline.md`
