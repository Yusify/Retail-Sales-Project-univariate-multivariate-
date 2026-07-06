# Retail Sales Time Series Forecasting

A time series forecasting project on Walmart retail sales data. The project compares univariate and multivariate modeling approaches, with results visualized in a Power BI report.

## Dataset

[Retail Data Analytics (Kaggle)](https://www.kaggle.com/datasets/manjeetsingh/retaildataset) — weekly sales across 45 Walmart stores, store-level features (Temperature, Fuel Price, CPI, Unemployment, MarkDown, IsHoliday), and store metadata (Type, Size).

## File Structure

```
├── Retail_sales_Prediction.ipynb              # Univariate: SARIMA vs Prophet (Store 1)
├── Retail_sales_Prediction-Multivariate.ipynb # Multivariate: SARIMAX + exogenous variables
├── retail_sales_cleaned.csv                   # Cleaned data for Power BI
├── model_comparison.csv                       # Model results for Power BI
└── Retail_Sales_Report.pbix                   # Power BI report
```

## Methodology

1. **Data Cleaning** — merged sales, features, and stores files; dropped MarkDown columns (>60% missing); fixed Date format
2. **EDA** — visualized trend/seasonality for Store 1
3. **Decomposition** — split series into trend, seasonal, and residual components (additive)
4. **Stationarity** — ADF test
5. **ACF/PACF** — used to select SARIMA parameters (strong yearly seasonality at lag 52)
6. **Modeling:**
   - SARIMA(0,1,1)(1,1,1,52) — univariate baseline
   - Prophet — alternative model for comparison
   - SARIMAX + IsHoliday — exogenous variable test
   - SARIMAX + CPI + Temperature — final model
7. **Evaluation** — MAE, RMSE, MAPE on the test set
8. **Power BI Report** — model comparison and interactive exploration across all 45 stores

## Results

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| **SARIMAX + CPI + Temperature** | **39,797.80** | **52,625.05** | **2.54%** |
| Univariate SARIMA | 42,100.76 | 56,711.66 | 2.68% |
| SARIMAX + IsHoliday | 46,728.30 | 57,307.22 | 2.98% |
| Prophet | 59,309.45 | 75,707.41 | 3.78% |

**Final model: SARIMAX(0,1,1)(1,1,1,52) + CPI + Temperature**

## Key Findings

- The data shows strong **yearly seasonality** (a clear spike at lag 52 in ACF/PACF), though the ADF test found no significant trend.
- **IsHoliday** as an exogenous variable added no value — its information was already captured by the seasonal component `(1,1,1,52)`.
- **CPI** was statistically significant, while **Temperature** was not on its own (p=0.837). However, removing Temperature worsened the AIC and reduced CPI's significance — indicating an interaction effect between the two. Variable selection should therefore rely on overall model performance (AIC, forecast error), not p-values in isolation.
- **auto_arima doesn't always outperform manual selection** — on Store 14, auto_arima's chosen parameters (d=0, D=0) performed worse than the manually selected `(0,1,1)(1,1,1,52)`, since AIC only measures fit to past data, not forecast accuracy.
- Applying fixed parameters across all 45 stores gave an average MAPE of ~4.85%, but with wide variation (2.63%–15.96%) — some stores (e.g. Store 14) show a clear declining trend that fixed parameters fail to capture.

## Limitations & Future Work

- Detailed modeling was performed only on **Store 1**; other stores were evaluated using fixed parameters.
- The limited sample size (121 weeks) relative to the seasonal period (52) caused convergence warnings in some models.
- Future work: per-store optimized parameters, or an ML-based approach (lag features + Random Forest/XGBoost).
