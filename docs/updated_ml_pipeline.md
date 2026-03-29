# Updated Methodology Pipeline

Data Collection (CSV-only)
-> Data Preprocessing
-> Feature Engineering
-> Parallel Model Training:
   - GRU Forecasting
   - BiLSTM fallback
   - Random Forest
   - SVR (RBF)
   - KNN Regressor
   - Gradient Boosting
   - Linear Regression
   - Optional XGBoost (when installed)
-> Model Evaluation (MAE, RMSE, R2)
-> Model Coordination Layer (balanced inverse-RMSE weighting)
-> Final Optimized Temperature Prediction
-> Decision and Control Simulation (fan/spray thresholds unchanged)
-> Visualization and Reporting
