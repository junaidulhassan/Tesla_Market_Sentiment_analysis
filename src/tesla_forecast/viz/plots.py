"""Plotly figure builders shared by the Streamlit app and the notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..features.indicators import add_display_indicators
from .theme import ACTUAL, FORECAST, GRID, NEUTRAL, PALETTE, SEQ_BLUES

_LAYOUT = dict(hovermode="x unified", margin=dict(l=10, r=10, t=50, b=10),
               legend=dict(orientation="h", y=1.04, x=0), font=dict(size=12))


def _style(fig: go.Figure, title: str, height: int = 460, template: str | None = None) -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0.01), height=height, **_LAYOUT)
    if template:
        fig.update_layout(template=template)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def price_chart(prices: pd.DataFrame, last_n: int = 250, template: str | None = None) -> go.Figure:
    """Candlesticks + SMAs + Bollinger band, with volume and RSI panels."""
    d = add_display_indicators(prices).iloc[-last_n:]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.62, 0.18, 0.20])
    fig.add_trace(go.Scatter(x=d.index, y=d["bb_upper"], line=dict(width=0), hoverinfo="skip", showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["bb_lower"], line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(42,120,214,0.10)", name="Bollinger (20, 2σ)", hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="TSLA",
                                 increasing_line_color=PALETTE["blue"], decreasing_line_color=PALETTE["orange"],
                                 increasing_fillcolor=PALETTE["blue"], decreasing_fillcolor=PALETTE["orange"]), 1, 1)
    for col, color in (("sma_20", PALETTE["aqua"]), ("sma_50", PALETTE["violet"]), ("sma_200", PALETTE["magenta"])):
        fig.add_trace(go.Scatter(x=d.index, y=d[col], name=col.upper().replace("_", " "), line=dict(color=color, width=1.4)), 1, 1)
    up = d["Close"] >= d["Open"]
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], marker_color=np.where(up, PALETTE["blue"], PALETTE["orange"]),
                         opacity=0.7, name="Volume", showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["rsi_14"], name="RSI 14", line=dict(color=PALETTE["violet"], width=1.5)), 3, 1)
    for lvl in (30, 70):
        fig.add_hline(y=lvl, line=dict(color=NEUTRAL, width=1, dash="dot"), row=3, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="USD", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    return _style(fig, "TSLA price, trend and momentum", 640, template)


def forecast_chart(close: pd.Series, result, history: int = 90, show_models: bool = False,
                   template: str | None = None) -> go.Figure:
    """History + ensemble forecast with 80%/95% conformal bands."""
    hist = close.iloc[-history:]
    x_f = [hist.index[-1], *result.dates]
    fig = go.Figure()
    for lv in sorted(result.bounds, reverse=True):
        lo, hi = result.bounds[lv]
        a = 0.12 if lv >= 0.9 else 0.24
        fig.add_trace(go.Scatter(x=x_f + x_f[::-1], y=[result.last_close, *hi, result.last_close, *lo[::-1]],
                                 fill="toself", fillcolor=f"rgba(235,104,52,{a})", line=dict(width=0),
                                 name=f"{int(lv * 100)}% interval", hoverinfo="skip"))
    if show_models:
        for name, p in result.model_prices.items():
            fig.add_trace(go.Scatter(x=x_f, y=[result.last_close, *p], name=name, mode="lines",
                                     line=dict(width=1, dash="dot"), opacity=0.55))
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Close", line=dict(color=ACTUAL, width=2)))
    fig.add_trace(go.Scatter(x=x_f, y=[result.last_close, *result.price], name="Ensemble forecast",
                             mode="lines+markers", line=dict(color=FORECAST, width=2.4), marker=dict(size=7)))
    fig.update_yaxes(title_text="USD")
    return _style(fig, f"TSLA {result.horizon}-trading-day forecast", 480, template)


def backtest_chart(bt, oof, h: int, last_n: int = 250, level: float = 0.80, template: str | None = None) -> go.Figure:
    """Out-of-sample ensemble forecast vs realised price at horizon ``h`` (aligned on the target date)."""
    j = h - 1
    sl = slice(-last_n, None)
    idx = bt.dates[sl]
    base, pred = bt.close[sl], oof.pred[sl, j]
    lo, hi = oof.bounds[level][0][sl, j], oof.bounds[level][1][sl, j]
    # target date = origin + h trading rows -> approximate with position offset on the origin axis
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=base * np.exp(hi), line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=idx, y=base * np.exp(lo), line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(235,104,52,0.20)", name=f"{int(level * 100)}% interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=idx, y=base * np.exp(bt.actual[sl, j]), name=f"Actual price (t+{h})",
                             line=dict(color=ACTUAL, width=2)))
    fig.add_trace(go.Scatter(x=idx, y=base * np.exp(pred), name=f"Ensemble forecast (t+{h})",
                             line=dict(color=FORECAST, width=1.8)))
    fig.update_yaxes(title_text="USD")
    return _style(fig, f"Walk-forward backtest, {h}-day-ahead (plotted at forecast origin)", 440, template)


def leaderboard_chart(metrics: pd.DataFrame, h: int, template: str | None = None) -> go.Figure:
    """RMSE skill vs the random walk at horizon h (positive = better than naive)."""
    d = metrics[(metrics.horizon == h) & (metrics.model != "naive")].sort_values("Skill_vs_naive_%")
    colors = [FORECAST if m == "ensemble" else ACTUAL for m in d.model]
    fig = go.Figure(go.Bar(x=d["Skill_vs_naive_%"], y=d.model, orientation="h", marker_color=colors,
                           text=[f"{v:+.2f}%" for v in d["Skill_vs_naive_%"]], textposition="outside",
                           hovertemplate="%{y}: %{x:.2f}% RMSE skill<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=NEUTRAL, width=1))
    fig.update_xaxes(title_text="RMSE improvement over random walk (%)")
    return _style(fig, f"Model skill vs naive at t+{h}", 120 + 38 * len(d), template)


def skill_by_horizon_chart(metrics: pd.DataFrame, highlight: list[str] | None = None, template: str | None = None) -> go.Figure:
    names = highlight or ["ensemble"]
    fig = go.Figure()
    others = [m for m in metrics.model.unique() if m not in names + ["naive"]]
    for m in others:
        d = metrics[metrics.model == m]
        fig.add_trace(go.Scatter(x=d.horizon, y=d["Skill_vs_naive_%"], name=m, mode="lines",
                                 line=dict(color=NEUTRAL, width=1), opacity=0.5))
    for m in names:
        d = metrics[metrics.model == m]
        fig.add_trace(go.Scatter(x=d.horizon, y=d["Skill_vs_naive_%"], name=m, mode="lines+markers",
                                 line=dict(color=FORECAST, width=2.6)))
    fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1, dash="dot"))
    fig.update_xaxes(title_text="Forecast horizon (trading days)", dtick=1)
    fig.update_yaxes(title_text="RMSE skill vs naive (%)")
    return _style(fig, "Skill vs random walk across horizons (above 0 = better)", 420, template)


def weights_heatmap(bt, oof, h: int, template: str | None = None) -> go.Figure:
    j = h - 1
    fig = go.Figure(go.Heatmap(z=oof.weights[:, j, :].T * 100, x=bt.dates, y=oof.names,
                               colorscale=[[i / (len(SEQ_BLUES) - 1), c] for i, c in enumerate(SEQ_BLUES)],
                               colorbar=dict(title="weight %"), hovertemplate="%{y} · %{x|%Y-%m-%d}: %{z:.1f}%<extra></extra>"))
    return _style(fig, f"Ensemble weights over time (t+{h})", 120 + 32 * len(oof.names), template)


def calibration_chart(coverage: pd.DataFrame, template: str | None = None) -> go.Figure:
    fig = go.Figure()
    for (lv, color) in zip(sorted(coverage.level.unique()), (PALETTE["blue"], PALETTE["orange"], PALETTE["aqua"])):
        d = coverage[coverage.level == lv]
        fig.add_trace(go.Scatter(x=d.horizon, y=d.empirical_coverage * 100, mode="lines+markers",
                                 name=f"{int(lv * 100)}% empirical", line=dict(color=color, width=2.2)))
        fig.add_hline(y=lv * 100, line=dict(color=color, width=1, dash="dot"))
    fig.update_xaxes(title_text="Horizon (trading days)", dtick=1)
    fig.update_yaxes(title_text="Empirical coverage (%)", range=[50, 100])
    return _style(fig, "Interval calibration (dotted = nominal)", 380, template)


def simulation_chart(sim, template: str | None = None) -> go.Figure:
    """Fan of simulated paths + terminal-return histogram."""
    fig = make_subplots(rows=1, cols=2, column_widths=[0.64, 0.36], subplot_titles=("Simulated price paths", "Terminal return"))
    x = [sim.dates[0] - pd.tseries.offsets.BDay(1), *sim.dates]
    for q, a in ((5, 95), (25, 75)):
        lo, hi = np.percentile(sim.paths, q, axis=0), np.percentile(sim.paths, a, axis=0)
        fig.add_trace(go.Scatter(x=x + x[::-1], y=[*hi, *lo[::-1]], fill="toself", line=dict(width=0),
                                 fillcolor=f"rgba(235,104,52,{0.14 if q == 5 else 0.28})", name=f"{q}-{a} pct", hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=np.median(sim.paths, axis=0), name="Median", line=dict(color=FORECAST, width=2.4)), 1, 1)
    ret = (sim.terminal / sim.last_close - 1) * 100
    fig.add_trace(go.Histogram(x=ret, nbinsx=50, marker_color=ACTUAL, opacity=0.85, name="Terminal return %", showlegend=False), 1, 2)
    fig.add_vline(x=0, line=dict(color=NEUTRAL, dash="dot"), row=1, col=2)
    fig.update_xaxes(title_text="Return (%)", row=1, col=2)
    return _style(fig, "Monte-Carlo risk view (filtered historical simulation)", 430, template)
