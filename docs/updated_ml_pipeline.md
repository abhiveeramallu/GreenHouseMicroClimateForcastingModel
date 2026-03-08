# Updated Methodology Pipeline

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
-> Plant Growth Intelligence Analytics
