"""Model factory: maps config names to constructed forecasters."""

from __future__ import annotations

from ..config import AppConfig
from .base import Forecaster
from .baselines import DriftForecaster, NaiveForecaster
from .deep import ARCHITECTURES, DeepForecaster
from .statistical import ARIMAForecaster, ThetaForecaster
from .supervised import GBMForecaster, RidgeForecaster

ALL_MODELS = ["naive", "drift", "arima", "theta", "ridge", "lightgbm", "xgboost", *ARCHITECTURES]

FAMILY = {
    "naive": "Baseline", "drift": "Baseline", "arima": "Statistical", "theta": "Statistical",
    "ridge": "Linear ML", "lightgbm": "Boosted trees", "xgboost": "Boosted trees",
    "gru_attention": "Deep learning", "tcn": "Deep learning", "transformer": "Deep learning",
    "ensemble": "Ensemble",
}


def build_model(name: str, cfg: AppConfig, horizon: int, feature_cols: list[str]) -> Forecaster:
    if name == "naive":
        return NaiveForecaster(horizon, feature_cols)
    if name == "drift":
        return DriftForecaster(horizon, feature_cols)
    if name == "arima":
        return ARIMAForecaster(horizon, feature_cols)
    if name == "theta":
        return ThetaForecaster(horizon, feature_cols)
    if name == "ridge":
        return RidgeForecaster(horizon, feature_cols)
    if name in ("lightgbm", "xgboost"):
        return GBMForecaster(horizon, feature_cols, backend=name, seed=cfg.seed)
    if name in ARCHITECTURES:
        return DeepForecaster(name, horizon, feature_cols, cfg=cfg.models.deep, seed=cfg.seed)
    raise ValueError(f"unknown model {name!r}; available: {ALL_MODELS}")


def build_models(cfg: AppConfig, horizon: int, feature_cols: list[str]) -> dict[str, Forecaster]:
    return {n: build_model(n, cfg, horizon, feature_cols) for n in cfg.models.enabled}
