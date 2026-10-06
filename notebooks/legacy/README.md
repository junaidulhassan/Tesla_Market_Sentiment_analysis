# Legacy (v1) material

Kept for reference only; superseded by `notebooks/01_tesla_advanced_forecasting.ipynb` and the `tesla_forecast` package.

| File | What it was | Known issues |
|---|---|---|
| `Tesla_LSTM_baseline.ipynb` | Single LSTM on 7 lagged closes, R² on price level | Trained on the test set (`model.fit(X, y)` with `X_test` as validation), price-level R² (≈0.99 for any naive forecast), calendar-day forecast dates |
| `App_v1.py` | Streamlit app using the `.h5` LSTM + LLM insights | Imports a removed `api_token.py`; weekend forecast dates; recursive 1-step forecasting |
| `RAG_market_insights.ipynb` | LLM + web-search experiment | Needs API keys via environment variables |

These files will not run unmodified (the model file and `api_token.py` were removed).
