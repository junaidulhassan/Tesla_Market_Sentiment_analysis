"""Tesla (TSLA) forecasting dashboard.  Run:  streamlit run app/streamlit_app.py"""

from __future__ import annotations

import copy
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allow running without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from tesla_forecast import load_config  # noqa: E402
from tesla_forecast.data.loader import load_prices  # noqa: E402
from tesla_forecast.forecasting import ForecastPipeline  # noqa: E402
from tesla_forecast.models.registry import FAMILY  # noqa: E402
from tesla_forecast.sentiment import (  # noqa: E402
    fetch_headlines,
    generate_insights,
    sentiment_index,
    technical_snapshot,
)
from tesla_forecast.viz import plots  # noqa: E402

st.set_page_config(page_title="TSLA Forecast Lab", page_icon="📈", layout="wide")

DISCLAIMER = ("Educational research tool, **not financial advice**. Stock prices are close to a random walk: "
              "the intervals and probabilities below are more informative than the point forecast.")


# ----------------------------------------------------------------------------- cached loaders
@st.cache_resource(show_spinner=False)
def get_config():
    return load_config()


@st.cache_resource(show_spinner="Loading trained models…")
def get_pipeline(mtime: float):
    cfg = get_config()
    return ForecastPipeline.load(cfg=cfg)


@st.cache_data(ttl=900, show_spinner="Fetching market data…")
def get_prices(refresh_token: int):
    pd_ = load_prices(get_config(), force_refresh=bool(refresh_token))
    return pd_


@st.cache_data(ttl=900, show_spinner="Fetching headlines…")
def get_news():
    return fetch_headlines("TSLA", limit=25)


def prepared_pipeline(refresh_token: int) -> ForecastPipeline | None:
    path = get_config().pipeline_path
    if not path.exists():
        return None
    base = get_pipeline(path.stat().st_mtime)
    pipe = copy.copy(base)  # per-run shallow copy: the cached models are shared, the data is not
    prices = get_prices(refresh_token)
    # rebuild the feature frame on the freshest data (fitted models are reused as-is)
    return pipe.prepare(prices=prices)


def train_in_app(quick: bool) -> None:
    overrides: dict = {}
    if quick:
        overrides = {"models": {"enabled": ["naive", "drift", "theta", "ridge", "lightgbm", "xgboost"]},
                     "backtest": {"n_test_origins": 378, "refit_every": 126}}
    cfg = load_config(overrides=overrides)
    with st.status("Training… (backtest → ensemble → final fit)", expanded=True) as status:
        pipe = ForecastPipeline(cfg).prepare(force_refresh=True)
        pipe.backtest(progress=lambda m: status.write(m))
        status.write("Fitting final models on all history…")
        pipe.fit()
        pipe.save()
        status.update(label="Training complete", state="complete")
    get_pipeline.clear()
    st.rerun()


# ----------------------------------------------------------------------------- sidebar
cfg = get_config()
with st.sidebar:
    st.title("📈 TSLA Forecast Lab")
    refresh = st.session_state.setdefault("refresh", 0)
    if st.button("🔄 Refresh market data", width="stretch"):
        st.session_state["refresh"] += 1
        get_news.clear()
        st.rerun()
    show_models = st.toggle("Show individual model paths", value=False)
    history = st.slider("History shown (trading days)", 30, 250, 90, step=10)
    st.divider()
    with st.expander("⚙️ Retrain models"):
        st.caption("Quick ≈ 30s (statistical + tree models). Full ≈ 2-4 min (adds the three deep nets, 3-year backtest).")
        c1, c2 = st.columns(2)
        if c1.button("Quick", width="stretch"):
            train_in_app(quick=True)
        if c2.button("Full", width="stretch"):
            train_in_app(quick=False)
    st.caption(DISCLAIMER)

pipe = prepared_pipeline(refresh)
st.title("Tesla (TSLA) · multi-week forecast")

if pipe is None:
    st.info("No trained pipeline found yet. Train one to get started — use the sidebar, or the button below.")
    if st.button("Train quick model now", type="primary"):
        train_in_app(quick=True)
    st.stop()

try:
    result = pipe.forecast()
except Exception as exc:
    st.error(f"Could not produce a forecast: {exc}")
    st.stop()

prices = pipe.prices
df = prices.df
if not prices.is_live:
    st.warning("Live data unavailable — showing the last " f"cached/bundled data ending {prices.last_date.date()}. " + "; ".join(prices.notes))
age_days = (datetime.now() - pipe.trained_at).days if pipe.trained_at else None
if age_days is not None and age_days > 30:
    st.warning(f"Models were trained {age_days} days ago — consider retraining (sidebar).")

# ----------------------------------------------------------------------------- KPIs
last, prev = df["Close"].iloc[-1], df["Close"].iloc[-2]
end_idx = -1
k = st.columns(5)
k[0].metric("Last close", f"${last:,.2f}", f"{(last / prev - 1) * 100:+.2f}%")
k[1].metric(f"Forecast t+{result.horizon}", f"${result.price[end_idx]:,.2f}", f"{(result.price[end_idx] / last - 1) * 100:+.2f}%")
lo80, hi80 = result.bounds[0.8][0][end_idx], result.bounds[0.8][1][end_idx]
k[2].metric("80% interval", f"${lo80:,.0f} – ${hi80:,.0f}")
k[3].metric("P(price up)", f"{result.prob_up[end_idx] * 100:.0f}%")
k[4].metric("Data as of", f"{result.as_of.date()}", prices.source)

tabs = st.tabs(["🔮 Forecast", "📊 Market", "🧪 Model performance", "🎲 Risk simulation", "📰 Insights & news", "ℹ️ About"])

# ----------------------------------------------------------------------------- forecast
with tabs[0]:
    st.plotly_chart(plots.forecast_chart(df["Close"], result, history, show_models), width="stretch", theme="streamlit")
    tbl = result.to_frame()
    st.subheader("Day-by-day forecast")
    st.dataframe(
        tbl.style.format({"forecast": "${:,.2f}", "change_%": "{:+.2f}%", "prob_up_%": "{:.0f}%",
                          **{c: "${:,.2f}" for c in tbl.columns if c.startswith(("lower", "upper"))}}),
        width="stretch")
    st.download_button("⬇️ Download forecast (CSV)", tbl.to_csv().encode(), "tsla_forecast.csv", "text/csv")
    with st.expander("What does each model say at the horizon?"):
        mp = pd.DataFrame({"model": list(result.model_prices), "family": [FAMILY.get(m, "") for m in result.model_prices],
                           f"price t+{result.horizon}": [v[end_idx] for v in result.model_prices.values()],
                           "ensemble weight %": [result.weights.get(m, np.zeros(result.horizon))[end_idx] * 100 for m in result.model_prices]})
        st.dataframe(mp.sort_values("ensemble weight %", ascending=False), hide_index=True, width="stretch")

# ----------------------------------------------------------------------------- market
with tabs[1]:
    st.plotly_chart(plots.price_chart(df, last_n=max(history, 120)), width="stretch", theme="streamlit")
    snap = technical_snapshot(df)
    c = st.columns(6)
    c[0].metric("RSI (14)", f"{snap['rsi_14']:.0f}")
    c[1].metric("vs SMA50", f"{snap['vs_sma50_%']:+.1f}%")
    c[2].metric("vs SMA200", f"{snap['vs_sma200_%']:+.1f}%")
    c[3].metric("21d vol (ann.)", f"{snap['realized_vol_21d_ann_%']:.0f}%")
    c[4].metric("From 52w high", f"{snap['from_52w_high_%']:.1f}%")
    c[5].metric("Volume vs avg", f"{snap['volume_vs_21d_avg']:.1f}×")

# ----------------------------------------------------------------------------- performance
with tabs[2]:
    bt, oof, m = pipe.bt, pipe.oof, pipe.metrics
    st.markdown(
        f"Expanding-window **walk-forward** backtest: **{len(bt.positions)} out-of-sample origins** "
        f"({bt.dates[0].date()} → {bt.dates[-1].date()}), models refit every {pipe.cfg.backtest.refit_every} days. "
        "Skill is RMSE improvement over the **random-walk (naive)** forecast — the honest benchmark for stocks.")
    h = st.select_slider("Horizon (trading days)", options=list(range(1, pipe.horizon + 1)), value=min(5, pipe.horizon))
    c1, c2 = st.columns([1, 1])
    c1.plotly_chart(plots.leaderboard_chart(m, h), width="stretch", theme="streamlit")
    c2.plotly_chart(plots.skill_by_horizon_chart(m), width="stretch", theme="streamlit")
    cols = ["model", "RMSE", "MAE", "MAPE_%", "MASE", "DirAcc_%", "Skill_vs_naive_%", "DM_p"]
    st.dataframe(m[m.horizon == h][cols].sort_values("RMSE").style.format(precision=3), hide_index=True, width="stretch")
    st.caption("DM_p: Diebold–Mariano p-value vs naive (small = significantly different). DirAcc is undefined for the naive model.")
    st.plotly_chart(plots.backtest_chart(bt, oof, h), width="stretch", theme="streamlit")
    c3, c4 = st.columns(2)
    c3.plotly_chart(plots.calibration_chart(pipe.coverage), width="stretch", theme="streamlit")
    c4.plotly_chart(plots.weights_heatmap(bt, oof, h), width="stretch", theme="streamlit")

# ----------------------------------------------------------------------------- risk
with tabs[3]:
    sim = pipe.simulate(result)
    st.plotly_chart(plots.simulation_chart(sim), width="stretch", theme="streamlit")
    risk = pd.Series(sim.risk_summary()).rename("value").to_frame()
    st.dataframe(risk.style.format("{:.2f}"), width="stretch")
    st.caption(f"{len(sim.paths):,} simulated paths: bootstrapped volatility-standardised historical shocks, EWMA volatility "
               "clustering, drift = ensemble forecast. VaR/CVaR are on the terminal return over the full horizon.")

# ----------------------------------------------------------------------------- insights
with tabs[4]:
    news = get_news()
    senti = sentiment_index(news)
    fdict = {"horizon": result.horizon, "price_end": float(result.price[end_idx]), "change_%": float((result.price[end_idx] / last - 1) * 100),
             "lo80": float(lo80), "hi80": float(hi80), "prob_up_%": float(result.prob_up[end_idx] * 100)}
    insights, source = generate_insights(technical_snapshot(df), fdict, senti, list(news["title"]) if len(news) else [])
    st.caption(f"Insights source: **{'LLM-written from computed data' if source == 'llm' else 'rule-based (set LLM_PROVIDER in .env for LLM prose)'}**")
    labels = {"trend_analysis": "📊 Trend", "technical_indicators": "📈 Technicals", "market_sentiment": "💹 Sentiment",
              "economic_factors": "🌍 Macro / industry", "prediction": "🔮 Outlook", "risks": "⚠️ Risks"}
    for key, title in labels.items():
        with st.expander(title, expanded=key in ("prediction", "trend_analysis")):
            st.write(insights.get(key, "N/A"))
    st.subheader("Recent headlines")
    if senti:
        st.metric("Headline tone index", f"{senti['index']:+.2f}", senti["label"])
        view = senti["scored"][["published", "title", "source", "score", "label"]]
        st.dataframe(view, hide_index=True, width="stretch")
    else:
        st.info("No headlines available (offline or feed blocked).")
    st.caption("Tone is a transparent keyword score shown for context only — it is not a model input.")

# ----------------------------------------------------------------------------- about
with tabs[5]:
    st.markdown(f"""
**Pipeline** — data (yfinance → cache → bundled CSV) → {len(pipe.feature_cols_)} causal features (momentum, volatility regimes, trend,
volume, calendar, SPY/QQQ/VIX context) → 10 models predicting *cumulative log-returns* for every day of the horizon →
leak-free **online inverse-error ensemble** → **volatility-normalised conformal intervals** → Monte-Carlo risk view.

**Models** — baselines (random walk, drift) · ARIMA · Theta · Ridge · LightGBM · XGBoost · GRU-attention · TCN · Transformer.

**Trained** {pipe.trained_at:%Y-%m-%d %H:%M} on data through {pipe.trained_through.date() if pipe.trained_through is not None else '?'}.

{DISCLAIMER}
""")
