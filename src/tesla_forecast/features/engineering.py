"""Stationary, scale-free, strictly causal features + multi-horizon targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as ind


@dataclass
class ForecastFrame:
    """Model-ready table.

    ``frame`` holds ``close`` (price level, never a feature) and ``feature_cols``.
    """

    frame: pd.DataFrame
    feature_cols: list[str]
    has_context: bool

    def __len__(self) -> int:
        return len(self.frame)


def forward_log_returns(close: pd.Series | np.ndarray, horizon: int) -> np.ndarray:
    """Matrix (n, horizon): ``log(P[t+h] / P[t])`` for h=1..horizon, NaN where not yet realised."""
    logp = np.log(np.asarray(close, dtype=float))
    n = len(logp)
    out = np.full((n, horizon), np.nan)
    for h in range(1, horizon + 1):
        out[: n - h, h - 1] = logp[h:] - logp[: n - h]
    return out


def _zscore(s: pd.Series, n: int) -> pd.Series:
    return (s - s.rolling(n, min_periods=n).mean()) / s.rolling(n, min_periods=n).std().replace(0, np.nan)


def build_frame(
    prices: pd.DataFrame,
    context: pd.DataFrame | None = None,
    warmup: int = 252,
) -> ForecastFrame:
    """Create the feature table from OHLCV (+ optional SPY/QQQ/VIX closes).

    Every column at row ``t`` depends only on rows ``<= t`` (verified in tests), so it is safe to
    slice the frame at any origin without leakage.
    """
    o, h, low, c, v = (prices[k].astype(float) for k in ["Open", "High", "Low", "Close", "Volume"])
    r = np.log(c).diff()
    f = pd.DataFrame(index=prices.index)

    # --- returns & momentum
    for k in (1, 2, 3, 5, 10, 21, 63):
        f[f"ret_{k}"] = np.log(c).diff(k)
    f["ret_abs_1"] = r.abs()
    f["gap"] = np.log(o / c.shift(1))
    f["intraday"] = np.log(c / o)
    f["range_hl"] = np.log(h / low)
    for k in (5, 10, 21):
        f[f"mean_ret_{k}"] = r.rolling(k, min_periods=k).mean()

    # --- volatility regimes
    for k in (5, 10, 21, 63):
        f[f"vol_{k}"] = r.rolling(k, min_periods=k).std()
    f["vol_ewma"] = ind.ewma_vol(r)
    f["vol_parkinson"] = ind.parkinson_vol(h, low)
    f["vol_gk"] = ind.garman_klass_vol(o, h, low, c)
    f["vol_ratio_5_21"] = f["vol_5"] / f["vol_21"]
    f["vol_ratio_21_63"] = f["vol_21"] / f["vol_63"]
    f["atr_pct"] = ind.atr(h, low, c) / c
    f["ret_skew_63"] = r.rolling(63, min_periods=63).skew()
    f["ret_kurt_63"] = r.rolling(63, min_periods=63).kurt()

    # --- trend / mean-reversion
    for k in (10, 20, 50, 200):
        f[f"px_sma_{k}"] = np.log(c / ind.sma(c, k))
    f["sma_20_50"] = np.log(ind.sma(c, 20) / ind.sma(c, 50))
    f["sma_50_200"] = np.log(ind.sma(c, 50) / ind.sma(c, 200))
    f["rsi_14"] = ind.rsi(c) / 100.0
    f["stoch_14"] = ind.stochastic(h, low, c) / 100.0
    m = ind.macd(c)
    f["macd_hist"] = m["macd_hist"] / c
    f["macd_line"] = m["macd"] / c
    bb = ind.bollinger(c)
    f["bb_pctb"], f["bb_width"] = bb["bb_pctb"], bb["bb_width"]
    hi252, lo252 = c.rolling(252, min_periods=126).max(), c.rolling(252, min_periods=126).min()
    f["dist_52w_high"] = np.log(c / hi252)
    f["dist_52w_low"] = np.log(c / lo252)
    f["drawdown_63"] = np.log(c / c.rolling(63, min_periods=63).max())

    # --- volume / liquidity
    lv = np.log1p(v)
    f["vol_z_21"] = _zscore(lv, 21)
    f["vol_z_63"] = _zscore(lv, 63)
    f["dollar_vol_chg_5"] = np.log1p(v * c).diff(5)
    f["obv_z_63"] = _zscore(ind.obv(c, v), 63)

    # --- calendar (cyclical encodings)
    # (month-of-year is deliberately NOT used: with ~11 years of data it only acts as a year identifier.)
    dow = prices.index.dayofweek.to_numpy()
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * dow / 5), np.cos(2 * np.pi * dow / 5)

    # --- market context (optional)
    has_ctx = False
    if context is not None and len(context):
        ctx = context.reindex(prices.index).ffill()
        if "spy" in ctx:
            sr = np.log(ctx["spy"]).diff()
            f["spy_ret_1"], f["spy_ret_5"] = sr, np.log(ctx["spy"]).diff(5)
            f["spy_vol_21"] = sr.rolling(21, min_periods=21).std()
            f["beta_63"] = r.rolling(63, min_periods=63).cov(sr) / sr.rolling(63, min_periods=63).var()
            f["rel_strength_21"] = f["ret_21"] - np.log(ctx["spy"]).diff(21)
            has_ctx = True
        if "qqq" in ctx:
            f["qqq_ret_1"], f["qqq_ret_5"] = np.log(ctx["qqq"]).diff(), np.log(ctx["qqq"]).diff(5)
            has_ctx = True
        if "vix" in ctx:
            lvix = np.log(ctx["vix"])
            f["vix_level"] = _zscore(lvix, 252)
            f["vix_chg_5"] = lvix.diff(5)
            has_ctx = True

    f = f.replace([np.inf, -np.inf], np.nan)
    feature_cols = list(f.columns)
    f.insert(0, "close", c)
    f = f.iloc[warmup:]
    # Forward-fill the odd isolated gap from context data, then drop rows still incomplete.
    f[feature_cols] = f[feature_cols].ffill(limit=3)
    f = f.dropna(subset=feature_cols + ["close"])
    return ForecastFrame(frame=f, feature_cols=feature_cols, has_context=has_ctx)
