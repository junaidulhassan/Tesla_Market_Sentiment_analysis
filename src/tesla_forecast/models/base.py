"""Common forecaster interface.

All models forecast the *cumulative log-return* ``log(P[t+h]/P[t])`` for ``h = 1..H`` from a forecast
origin ``t``. Working in return space (not price level) removes the non-stationarity that makes
price-level R^2 look deceptively good, and lets every model share one target.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..features.engineering import forward_log_returns


class Forecaster(ABC):
    """``fit(frame)`` uses only rows of ``frame``; ``predict(frame, origins)`` only rows ``<= origin``."""

    name: str = "base"

    def __init__(self, horizon: int, feature_cols: list[str] | None = None):
        self.horizon = horizon
        self.feature_cols = feature_cols or []

    @abstractmethod
    def fit(self, frame: pd.DataFrame) -> Forecaster: ...

    @abstractmethod
    def predict(self, frame: pd.DataFrame, origins: Sequence[int] | None = None) -> np.ndarray:
        """Return array ``(len(origins), horizon)`` of cumulative log-returns. Default: last row."""

    # ------------------------------------------------------------------ helpers
    def _origins(self, frame: pd.DataFrame, origins: Sequence[int] | None) -> np.ndarray:
        return np.array([len(frame) - 1] if origins is None else list(origins), dtype=int)

    def _targets(self, frame: pd.DataFrame) -> np.ndarray:
        return forward_log_returns(frame["close"].to_numpy(), self.horizon)

    def _row_scale(self, frame: pd.DataFrame, vol_norm: bool = True) -> np.ndarray:
        """Per-row, per-horizon scale ``vol_ewma * sqrt(h)`` used to volatility-normalise targets.

        Dividing returns by the volatility known at the origin removes the dominant source of
        heteroskedastic noise so the learners can focus on the (tiny) predictable component.
        """
        if not vol_norm:
            return np.ones((len(frame), self.horizon))
        vol = frame["vol_ewma"].to_numpy(dtype=float)[:, None]
        return vol * np.sqrt(np.arange(1, self.horizon + 1))[None, :]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, horizon={self.horizon})"
