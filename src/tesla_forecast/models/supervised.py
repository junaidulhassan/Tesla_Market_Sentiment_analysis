"""Tabular supervised models trained *directly* per horizon: Ridge, LightGBM, XGBoost.

Direct multi-horizon training avoids the error accumulation of recursive forecasting. GBMs are trained
at a handful of anchor horizons and interpolated (cumulative return is smooth in h), which both
regularises and keeps walk-forward refits cheap.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import Forecaster

_ANCHORS = (1, 2, 3, 5, 7, 10, 14, 20, 30, 45, 60)


def _winsorize(y: np.ndarray, k: float = 4.0) -> np.ndarray:
    """Clip targets at +-k robust sigmas per column (TSLA has violent tails)."""
    med = np.nanmedian(y, axis=0)
    mad = np.nanmedian(np.abs(y - med), axis=0) * 1.4826 + 1e-12
    return np.clip(y, med - k * mad, med + k * mad)


def _interp_horizons(anchor_h: Sequence[int], anchor_pred: np.ndarray, horizon: int) -> np.ndarray:
    """Linear interpolation of (n, n_anchors) predictions to every h in 1..horizon (pred(0)=0)."""
    xs = np.concatenate([[0], anchor_h])
    ys = np.concatenate([np.zeros((anchor_pred.shape[0], 1)), anchor_pred], axis=1)
    hs = np.arange(1, horizon + 1)
    return np.stack([np.interp(hs, xs, row) for row in ys])


class _TabularForecaster(Forecaster):
    def _xy(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        y = self._targets(frame)
        x = frame[self.feature_cols].to_numpy(dtype=float)
        return x, y


class RidgeForecaster(_TabularForecaster):
    """Strongly regularised linear model; alpha picked by purged time-series CV."""

    name = "ridge"

    def __init__(self, horizon: int, feature_cols: list[str] | None = None, vol_norm: bool = True):
        super().__init__(horizon, feature_cols)
        self.vol_norm = vol_norm
        self._pipe: Pipeline | None = None
        self.alpha_: float | None = None

    def fit(self, frame: pd.DataFrame) -> RidgeForecaster:
        x, y = self._xy(frame)
        y = y / self._row_scale(frame, self.vol_norm)
        ok = ~np.isnan(y).any(axis=1)
        x, y = x[ok], _winsorize(y[ok])
        pipe = Pipeline([("sc", StandardScaler()), ("ridge", Ridge())])
        grid = GridSearchCV(
            pipe,
            {"ridge__alpha": np.logspace(1, 5, 9)},
            cv=TimeSeriesSplit(n_splits=4, gap=self.horizon),
            scoring="neg_mean_squared_error",
        )
        grid.fit(np.clip(x, -1e6, 1e6), y)
        self._pipe, self.alpha_ = grid.best_estimator_, float(grid.best_params_["ridge__alpha"])
        return self

    def predict(self, frame: pd.DataFrame, origins: Sequence[int] | None = None) -> np.ndarray:
        idx = self._origins(frame, origins)
        raw = self._pipe.predict(frame[self.feature_cols].to_numpy(dtype=float)[idx])  # type: ignore[union-attr]
        return raw * self._row_scale(frame, self.vol_norm)[idx]


class GBMForecaster(_TabularForecaster):
    """Gradient-boosted trees (LightGBM or XGBoost) with shallow, heavily regularised trees."""

    def __init__(self, horizon: int, feature_cols: list[str] | None = None, backend: str = "lightgbm",
                 seed: int = 42, n_estimators: int = 150, vol_norm: bool = True):
        super().__init__(horizon, feature_cols)
        self.vol_norm = vol_norm
        if backend not in {"lightgbm", "xgboost"}:
            raise ValueError(f"unknown backend {backend}")
        self.backend, self.seed, self.n_estimators = backend, seed, n_estimators
        self.name = backend
        self.anchors_ = [h for h in _ANCHORS if h < horizon] + [horizon]
        self.models_: list = []

    def _new_model(self):
        if self.backend == "lightgbm":
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                n_estimators=self.n_estimators, learning_rate=0.03, num_leaves=7, max_depth=4,
                min_child_samples=150, subsample=0.7, subsample_freq=1, colsample_bytree=0.5,
                reg_lambda=30.0, extra_trees=True, random_state=self.seed, verbosity=-1, n_jobs=4,
            )
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=100, learning_rate=0.03, max_depth=2, min_child_weight=150, subsample=0.7,
            colsample_bytree=0.4, reg_lambda=100.0, reg_alpha=1.0, random_state=self.seed,
            tree_method="hist", n_jobs=4, verbosity=0,
        )

    def fit(self, frame: pd.DataFrame) -> GBMForecaster:
        x, y = self._xy(frame)
        y = y / self._row_scale(frame, self.vol_norm)
        self.models_ = []
        for h in self.anchors_:
            ok = ~np.isnan(y[:, h - 1])
            yh = _winsorize(y[ok][:, [h - 1]])[:, 0]
            self.models_.append(self._new_model().fit(x[ok], yh))
        return self

    def predict(self, frame: pd.DataFrame, origins: Sequence[int] | None = None) -> np.ndarray:
        idx = self._origins(frame, origins)
        x = frame[self.feature_cols].to_numpy(dtype=float)[idx]
        anchor_pred = np.column_stack([m.predict(x) for m in self.models_])
        # interpolate in the normalised space, then rescale by the origin's vol * sqrt(h)
        return _interp_horizons(self.anchors_, anchor_pred, self.horizon) * self._row_scale(frame, self.vol_norm)[idx]
