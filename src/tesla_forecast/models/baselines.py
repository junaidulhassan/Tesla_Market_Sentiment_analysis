"""Benchmarks every real model must beat. A random walk is brutally hard to beat for stocks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .base import Forecaster


class NaiveForecaster(Forecaster):
    """Random walk: tomorrow's price = today's price (zero expected return)."""

    name = "naive"

    def fit(self, frame: pd.DataFrame) -> NaiveForecaster:
        return self

    def predict(self, frame: pd.DataFrame, origins: Sequence[int] | None = None) -> np.ndarray:
        return np.zeros((len(self._origins(frame, origins)), self.horizon))


class DriftForecaster(Forecaster):
    """Random walk with drift = trailing mean daily log-return (estimated causally per origin)."""

    name = "drift"

    def __init__(self, horizon: int, feature_cols: list[str] | None = None, window: int = 252):
        super().__init__(horizon, feature_cols)
        self.window = window

    def fit(self, frame: pd.DataFrame) -> DriftForecaster:
        return self

    def predict(self, frame: pd.DataFrame, origins: Sequence[int] | None = None) -> np.ndarray:
        idx = self._origins(frame, origins)
        r = np.log(frame["close"]).diff().rolling(self.window, min_periods=20).mean().fillna(0.0).to_numpy()
        steps = np.arange(1, self.horizon + 1)
        return r[idx][:, None] * steps[None, :]
