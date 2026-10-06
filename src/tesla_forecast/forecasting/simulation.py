"""Filtered-historical-simulation of future price paths (risk view complementing the intervals)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SimulationResult:
    paths: np.ndarray  # (n_paths, H+1) prices, column 0 = last close
    dates: pd.DatetimeIndex  # H forecast dates
    last_close: float

    @property
    def terminal(self) -> np.ndarray:
        return self.paths[:, -1]

    def risk_summary(self, tail: float = 0.05) -> dict[str, float]:
        ret = self.terminal / self.last_close - 1
        var = float(np.quantile(ret, tail))
        running_max = np.maximum.accumulate(self.paths, axis=1)
        mdd = (self.paths / running_max - 1).min(axis=1)
        return {
            "prob_up_%": float((ret > 0).mean() * 100),
            "expected_return_%": float(ret.mean() * 100),
            "median_return_%": float(np.median(ret) * 100),
            f"VaR_{int((1 - tail) * 100)}_%": var * 100,
            f"CVaR_{int((1 - tail) * 100)}_%": float(ret[ret <= var].mean() * 100),
            "prob_gain_>10%": float((ret > 0.10).mean() * 100),
            "prob_loss_>10%": float((ret < -0.10).mean() * 100),
            "expected_max_drawdown_%": float(mdd.mean() * 100),
            "worst_5%_max_drawdown_%": float(np.quantile(mdd, 0.05) * 100),
        }


def simulate_paths(
    last_close: float,
    cum_log_return_forecast: np.ndarray,
    log_returns: pd.Series,
    dates: pd.DatetimeIndex,
    n_paths: int = 5000,
    seed: int = 42,
    lam: float = 0.94,
) -> SimulationResult:
    """Bootstrap vol-standardised historical shocks, re-inflated by an EWMA volatility recursion.

    The expected path follows the model forecast (daily drift = increments of the cumulative forecast).
    """
    r = log_returns.dropna().to_numpy()
    burn = 60  # seed the EWMA with the first sample variance and discard its warm-up (it is badly biased)
    seeded = np.concatenate([[np.var(r[:burn])], r[1:] ** 2])
    var = pd.Series(seeded).ewm(alpha=1 - lam, adjust=False).mean().to_numpy()
    z = (r[burn + 1:] / np.sqrt(var[burn:-1]))
    z = (z - z.mean()) / z.std()  # unit-variance, zero-mean empirical innovations (fat tails kept)
    H = len(cum_log_return_forecast)
    drift = np.diff(np.concatenate([[0.0], cum_log_return_forecast]))
    rng = np.random.default_rng(seed)
    sigma2 = np.full(n_paths, lam * var[-1] + (1 - lam) * r[-1] ** 2)
    cum = np.zeros((n_paths, H + 1))
    for d in range(H):
        shock = np.sqrt(sigma2) * rng.choice(z, n_paths)
        cum[:, d + 1] = cum[:, d] + drift[d] + shock
        sigma2 = lam * sigma2 + (1 - lam) * shock**2
    return SimulationResult(paths=last_close * np.exp(cum), dates=dates, last_close=float(last_close))
