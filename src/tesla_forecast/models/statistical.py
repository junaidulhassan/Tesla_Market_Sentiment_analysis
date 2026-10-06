"""Classical time-series models (statsmodels): ARIMA on log-returns and the Theta method."""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .base import Forecaster

_SCALE = 100.0  # fit on percent log-returns: better conditioned optimisation


class ARIMAForecaster(Forecaster):
    """ARMA(p, q) on daily log-returns, order chosen by AIC at fit time.

    At prediction time the fitted parameters are *held fixed* and the Kalman filter is re-run over
    the history up to each origin (``apply``), so one fit serves a whole walk-forward fold.
    """

    name = "arima"

    def __init__(self, horizon: int, feature_cols: list[str] | None = None, max_p: int = 2, max_q: int = 2,
                 max_fit_obs: int = 1500):
        super().__init__(horizon, feature_cols)
        self.max_p, self.max_q, self.max_fit_obs = max_p, max_q, max_fit_obs
        self.order_: tuple[int, int, int] = (0, 0, 0)
        self._res = None

    def fit(self, frame: pd.DataFrame) -> ARIMAForecaster:
        from statsmodels.tsa.arima.model import ARIMA

        r = np.log(frame["close"]).diff().dropna().to_numpy()[-self.max_fit_obs:] * _SCALE
        best = (np.inf, None, (0, 0, 0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for p, q in itertools.product(range(self.max_p + 1), range(self.max_q + 1)):
                try:
                    res = ARIMA(r, order=(p, 0, q), trend="c").fit()
                except Exception:
                    continue
                if np.isfinite(res.aic) and res.aic < best[0]:
                    best = (res.aic, res, (p, 0, q))
        self._res, self.order_ = best[1], best[2]
        return self

    def predict(self, frame: pd.DataFrame, origins: Sequence[int] | None = None) -> np.ndarray:
        idx = self._origins(frame, origins)
        r_all = np.log(frame["close"]).diff().fillna(0.0).to_numpy() * _SCALE
        out = np.zeros((len(idx), self.horizon))
        if self._res is None:
            return out
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for k, t in enumerate(idx):
                try:
                    hist = r_all[max(1, t + 1 - 1500): t + 1]
                    fc = self._res.apply(hist, refit=False).forecast(self.horizon)
                    out[k] = np.cumsum(np.asarray(fc)) / _SCALE
                except Exception:
                    out[k] = 0.0
        return out


class ThetaForecaster(Forecaster):
    """Theta method (M3-competition winner) on log-prices, re-estimated on a trailing window."""

    name = "theta"

    def __init__(self, horizon: int, feature_cols: list[str] | None = None, window: int = 252):
        super().__init__(horizon, feature_cols)
        self.window = window

    def fit(self, frame: pd.DataFrame) -> ThetaForecaster:
        return self

    def predict(self, frame: pd.DataFrame, origins: Sequence[int] | None = None) -> np.ndarray:
        from statsmodels.tsa.forecasting.theta import ThetaModel

        idx = self._origins(frame, origins)
        logp = np.log(frame["close"].to_numpy())
        out = np.zeros((len(idx), self.horizon))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for k, t in enumerate(idx):
                y = pd.Series(logp[max(0, t + 1 - self.window): t + 1])
                try:
                    fc = ThetaModel(y, period=1, deseasonalize=False).fit().forecast(self.horizon)
                    out[k] = np.asarray(fc) - logp[t]
                except Exception:
                    out[k] = 0.0
        return out
