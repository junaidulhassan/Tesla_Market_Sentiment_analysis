"""Classic technical indicators. All functions are causal (value at t uses data <= t only)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's relative strength index in [0, 100]."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper, lower = mid + k * sd, mid - k * sd
    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_pctb": (close - lower) / (upper - lower).replace(0, np.nan),
            "bb_width": (upper - lower) / mid,
        }
    )


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return true_range(high, low, close).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    lo, hi = low.rolling(n, min_periods=n).min(), high.rolling(n, min_periods=n).max()
    return 100 * (close - lo) / (hi - lo).replace(0, np.nan)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (np.sign(close.diff()).fillna(0) * volume).cumsum()


def parkinson_vol(high: pd.Series, low: pd.Series, n: int = 21) -> pd.Series:
    """Range-based daily volatility estimator (more efficient than close-to-close)."""
    hl = np.log(high / low) ** 2 / (4 * np.log(2))
    return np.sqrt(hl.rolling(n, min_periods=n).mean())


def garman_klass_vol(o: pd.Series, h: pd.Series, low: pd.Series, c: pd.Series, n: int = 21) -> pd.Series:
    term = 0.5 * np.log(h / low) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    return np.sqrt(term.clip(lower=0).rolling(n, min_periods=n).mean())


def ewma_vol(log_ret: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics exponentially weighted daily volatility."""
    return np.sqrt((log_ret**2).ewm(alpha=1 - lam, adjust=False, min_periods=20).mean())


def add_display_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Raw-scale indicators for charting (SMAs, Bollinger bands, RSI, MACD)."""
    out = df.copy()
    c = out["Close"]
    for n in (20, 50, 200):
        out[f"sma_{n}"] = sma(c, n)
    out = out.join(bollinger(c)).join(macd(c))
    out["rsi_14"] = rsi(c)
    out["atr_14"] = atr(out["High"], out["Low"], c)
    return out
