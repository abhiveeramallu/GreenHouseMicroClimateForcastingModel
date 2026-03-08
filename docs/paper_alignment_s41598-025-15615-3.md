# Paper Alignment Note

## Reference
- Scientific Reports, DOI: `10.1038/s41598-025-15615-3`
- URL: https://doi.org/10.1038/s41598-025-15615-3

## What we aligned from the paper
1. Non-invasive greenhouse analytics workflow (software-side processing of observed metrics).
2. Deep-learning forecasting mindset for greenhouse decision support.
3. Emphasis on crop-condition intelligence from environmental and growth-related features.

## What this project keeps strictly in scope
1. Software-only pipeline (no live IoT device integration in this implementation).
2. Historical CSV-driven ingestion, cleaning, model training, and simulation.
3. UI-ready outputs that mirror paper-style result communication:
   - actual vs predicted trends
   - per-crop performance metrics
   - action timelines for operational decisions

## Our project contribution over baseline paper flow
1. Added coordinated multi-model forecasting:
   - LSTM (primary)
   - Random Forest / Gradient Boosting / Linear Regression
   - linear autoregressive baseline fallback
   - optional XGBoost when dependency is available
2. Added model-evaluation layer with MAE, RMSE, and R2 comparison per model.
3. Added coordination layer using dynamic inverse-RMSE weighted ensemble.
4. Added threshold-based greenhouse control simulation (fan/spray) tied to final hybrid forecast values.
5. Added plant-growth intelligence extension:
   - growth-rate prediction (`PHR`)
   - stress detection from chlorophyll/root/biomass patterns
   - feature importance and correlation analysis
6. Added structured frontend payload export (`ui_payload.json`) for direct UI styling and dashboards.
