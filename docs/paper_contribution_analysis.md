# Contribution Over Baseline Greenhouse Growth Studies

Reference considered: Scientific Reports paper DOI 10.1038/s41598-025-15615-3
(focus on non-invasive greenhouse monitoring and deep-learning driven prediction).

Software-only scope maintained:
- No IoT hardware integration in this implementation.
- CSV datasets are the only data input channel.

Project contributions:
1. Implemented time-series forecasting pipeline (sliding-window) for greenhouse temperature prediction.
2. Extended forecasting with deep GRU primary model plus parallel ML models (RandomForest, GradientBoosting, SVR, KNN, LinearRegression, optional XGBoost).
3. Added model-evaluation layer (MAE, RMSE, R2) with per-crop comparison and ranking reports.
4. Implemented balanced inverse-RMSE weighted coordination for final hybrid prediction.
5. Integrated threshold-based decision simulation (fan/spray) connected to hybrid prediction.
6. Added UI-ready payload export (actual vs predicted + action timeline) for frontend styling.