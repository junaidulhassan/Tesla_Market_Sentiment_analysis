"""Volatility forecasting (used to scale prediction intervals and simulate risk)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_variance_path(log_returns: pd.Series, horizon: int, lam: float = 0.94) -> np.ndarray:
    """Flat EWMA daily-variance forecast for h=1..H (RiskMetrics)."""
    r = log_returns.dropna().to_numpy()
    var = pd.Series(r**2).ewm(alpha=1 - lam, adjust=False).mean().iloc[-1]
    return np.full(horizon, var)


def garch_variance_path(log_returns: pd.Series, horizon: int) -> tuple[np.ndarray, str]:
    """GJR-GARCH(1,1,1) with Student-t innovations; falls back to EWMA if ``arch`` is unavailable/fails.

    Returns ``(daily variance forecasts for h=1..H, method_name)``.
    """
    try:
        from arch import arch_model

        r = log_returns.dropna().iloc[-1500:] * 100
        res = arch_model(r, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off", show_warning=False)
        var = res.forecast(horizon=horizon, reindex=False).variance.to_numpy()[-1] / 1e4
        if np.all(np.isfinite(var)) and np.all(var > 0):
            return var, "GJR-GARCH(1,1)-t"
    except Exception:
        pass
    return ewma_variance_path(log_returns, horizon), "EWMA"


def garch_conditional_vol(log_returns: pd.Series) -> pd.Series | None:
    """In-sample annualised conditional volatility from a GJR-GARCH-t fit (for charts)."""
    try:
        from arch import arch_model

        r = log_returns.dropna() * 100
        res = arch_model(r, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off", show_warning=False)
        return res.conditional_volatility / 100 * np.sqrt(252)
    except Exception:
        return None
