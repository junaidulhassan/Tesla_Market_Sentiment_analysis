"""End-to-end orchestration: data -> features -> walk-forward backtest -> ensemble -> intervals -> forecast."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..config import AppConfig, load_config
from ..data.calendar import next_trading_days
from ..data.loader import PriceData, load_market_context, load_prices
from ..evaluation.backtest import BacktestResult, run_walk_forward
from ..evaluation.conformal import (
    horizon_scale,
    prob_up,
    rolling_conformal_bounds,
    standardized_residuals,
)
from ..evaluation.ensemble import online_inverse_error_weights, weighted_combination
from ..evaluation.metrics import coverage_table, metrics_by_horizon
from ..features.engineering import ForecastFrame, build_frame
from ..models.base import Forecaster
from ..models.registry import build_model
from ..utils import get_logger, set_seed
from .simulation import SimulationResult, simulate_paths

log = get_logger(__name__)


@dataclass
class ForecastResult:
    """A forecast from the last observed close, for the next ``horizon`` trading days."""

    as_of: pd.Timestamp
    last_close: float
    dates: pd.DatetimeIndex
    log_return: np.ndarray  # ensemble cumulative log-return, (H,)
    price: np.ndarray  # ensemble point forecast, (H,)
    bounds: dict[float, tuple[np.ndarray, np.ndarray]]  # level -> (lower, upper) price paths
    prob_up: np.ndarray  # P(price_h > last_close)
    model_prices: dict[str, np.ndarray]
    weights: dict[str, np.ndarray]  # model -> (H,) ensemble weight
    data_source: str = ""
    trained_through: pd.Timestamp | None = None

    @property
    def horizon(self) -> int:
        return len(self.price)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "date": self.dates, "forecast": self.price,
            "change_%": (self.price / self.last_close - 1) * 100,
            "prob_up_%": self.prob_up * 100,
        })
        for lv, (lo, hi) in self.bounds.items():
            df[f"lower_{int(lv * 100)}"], df[f"upper_{int(lv * 100)}"] = lo, hi
        return df.set_index("date")


@dataclass
class EnsembleOOF:
    """Out-of-sample ensemble artefacts derived from a backtest."""

    names: list[str]
    weights: np.ndarray  # (n, H, M)
    pred: np.ndarray  # (n, H)
    scale: np.ndarray  # (n, H)
    scores: np.ndarray  # (n, H) standardised residuals
    bounds: dict[float, tuple[np.ndarray, np.ndarray]]


@dataclass
class ForecastPipeline:
    cfg: AppConfig
    horizon: int | None = None
    prices: PriceData | None = None
    ff: ForecastFrame | None = None
    bt: BacktestResult | None = None
    oof: EnsembleOOF | None = None
    models: dict[str, Forecaster] = field(default_factory=dict)
    metrics: pd.DataFrame | None = None
    coverage: pd.DataFrame | None = None
    trained_at: datetime | None = None
    trained_through: pd.Timestamp | None = None
    feature_cols_: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.horizon = self.horizon or self.cfg.horizon

    # ----------------------------------------------------------------- data
    def prepare(self, force_refresh: bool = False, prices: PriceData | None = None,
                context: pd.DataFrame | None = None, load_context: bool = True) -> ForecastPipeline:
        """Load data and build features. ``prices``/``context`` can be injected (tests, notebooks)."""
        self.prices = prices or load_prices(self.cfg, force_refresh=force_refresh)
        for n in self.prices.notes:
            log.info("data: %s", n)
        if context is None and load_context:
            context = load_market_context(self.cfg, self.prices.df.index)
        self.ff = build_frame(self.prices.df, context, warmup=self.cfg.features.warmup)
        log.info("data source=%s rows=%d features=%d context=%s last=%s", self.prices.source, len(self.ff),
                 len(self.ff.feature_cols), self.ff.has_context, self.ff.frame.index[-1].date())
        return self

    def _factories(self, names: list[str] | None = None) -> dict[str, Callable[[], Forecaster]]:
        assert self.ff is not None
        names = names or self.cfg.models.enabled
        return {n: (lambda n=n: build_model(n, self.cfg, self.horizon, self.ff.feature_cols)) for n in names}

    # ------------------------------------------------------------- backtest
    def backtest(self, model_names: list[str] | None = None,
                 progress: Callable[[str], None] | None = None) -> BacktestResult:
        assert self.ff is not None, "call prepare() first"
        set_seed(self.cfg.seed)
        t0 = time.perf_counter()
        self.bt = run_walk_forward(self.ff, self._factories(model_names), self.cfg, self.horizon, progress)
        self._build_oof()
        log.info("backtest finished in %.1fs (%d origins, %d models)", time.perf_counter() - t0,
                 len(self.bt.positions), len(self.bt.preds))
        return self.bt

    def _build_oof(self) -> None:
        bt, ec, ic = self.bt, self.cfg.ensemble, self.cfg.intervals
        assert bt is not None
        names, W = online_inverse_error_weights(bt.preds, bt.actual, bt.positions, window=ec.window,
                                                min_obs=ec.min_obs, power=ec.power,
                                                shrink_to_equal=ec.shrink_to_equal)
        pred = weighted_combination(bt.preds, names, W)
        scale = horizon_scale(bt.vol, bt.horizon)
        scores = standardized_residuals(bt.actual, pred, scale)
        bounds = rolling_conformal_bounds(scores, bt.positions, pred, scale, ic.levels,
                                          window=ic.window, min_obs=ic.min_obs)
        self.oof = EnsembleOOF(names, W, pred, scale, scores, bounds)
        horizons = list(range(1, bt.horizon + 1))
        allp = {**bt.preds, "ensemble": pred}
        self.metrics = metrics_by_horizon(bt.close, bt.actual, allp, horizons)
        self.coverage = coverage_table(bt.actual, bounds, horizons)

    # ----------------------------------------------------------------- fit
    def fit(self, model_names: list[str] | None = None) -> ForecastPipeline:
        """Fit every model on all available history (the models used for live forecasts)."""
        assert self.ff is not None, "call prepare() first"
        names = model_names or (self.bt.models if self.bt else self.cfg.models.enabled)
        frame = self.ff.frame
        self.models = {}
        for n in names:
            t0 = time.perf_counter()
            try:
                self.models[n] = self._factories([n])[n]().fit(frame)
                log.info("fitted %-14s %.1fs", n, time.perf_counter() - t0)
            except Exception as exc:
                log.warning("final fit of %s failed: %s", n, exc)
        self.trained_at, self.trained_through = datetime.now(), frame.index[-1]
        self.feature_cols_ = list(self.ff.feature_cols)
        return self

    # ------------------------------------------------------------- forecast
    def forecast(self, frame: pd.DataFrame | None = None) -> ForecastResult:
        """Forecast the next ``horizon`` trading days from the last row of ``frame``."""
        assert self.models and self.bt is not None and self.oof is not None, "fit() and backtest() first"
        frame = self.ff.frame if frame is None else frame
        missing = [c for c in self.feature_cols_ if c not in frame.columns]
        if missing:
            raise ValueError(
                f"{len(missing)} trained features are missing from the current data (e.g. {missing[:3]}); "
                "market-context data (SPY/QQQ/VIX) was probably unavailable. Reconnect or retrain offline.")
        H, t = self.horizon, len(frame) - 1
        point = {n: m.predict(frame, [t])[0] for n, m in self.models.items()}

        names, W = online_inverse_error_weights(
            {k: v for k, v in self.bt.preds.items() if k in point}, self.bt.actual, self.bt.positions,
            target_positions=np.array([t]), window=self.cfg.ensemble.window,
            min_obs=self.cfg.ensemble.min_obs, power=self.cfg.ensemble.power,
            shrink_to_equal=self.cfg.ensemble.shrink_to_equal)
        w = W[0]  # (H, M)
        ens = np.sum(np.stack([point[k] for k in names], axis=-1) * w, axis=-1)

        scale = horizon_scale(np.array([frame["vol_ewma"].iloc[-1]]), H)
        bounds = rolling_conformal_bounds(
            self.oof.scores, self.bt.positions, ens[None, :], scale, self.cfg.intervals.levels,
            target_positions=np.array([t]), window=self.cfg.intervals.window,
            min_obs=self.cfg.intervals.min_obs)
        recent = self.oof.scores[-self.cfg.intervals.window:]
        pu = prob_up(recent, ens, scale[0], self.cfg.intervals.min_obs)

        last = float(frame["close"].iloc[-1])
        as_of = frame.index[-1]
        return ForecastResult(
            as_of=as_of, last_close=last, dates=next_trading_days(as_of, H),
            log_return=ens, price=last * np.exp(ens),
            bounds={lv: (last * np.exp(lo[0]), last * np.exp(hi[0])) for lv, (lo, hi) in bounds.items()},
            prob_up=pu, model_prices={n: last * np.exp(p) for n, p in point.items()},
            weights={n: w[:, i] for i, n in enumerate(names)},
            data_source=self.prices.source if self.prices else "", trained_through=self.trained_through,
        )

    def simulate(self, result: ForecastResult) -> SimulationResult:
        assert self.ff is not None
        r = np.log(self.ff.frame["close"]).diff()
        return simulate_paths(result.last_close, result.log_return, r, result.dates,
                              self.cfg.simulation.n_paths, self.cfg.simulation.seed)

    # ----------------------------------------------------------- persistence
    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path or self.cfg.pipeline_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cfg": self.cfg.model_dump(), "horizon": self.horizon, "models": self.models, "bt": self.bt,
            "oof": self.oof, "metrics": self.metrics, "coverage": self.coverage,
            "trained_at": self.trained_at, "trained_through": self.trained_through,
            "feature_cols_": self.feature_cols_,
        }
        joblib.dump(payload, path, compress=3)
        log.info("saved pipeline -> %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path | None = None, cfg: AppConfig | None = None) -> ForecastPipeline:
        """Load a pipeline saved by :meth:`save`. Only load files you created yourself (pickle)."""
        cfg = cfg or load_config()
        path = Path(path or cfg.pipeline_path)
        p = joblib.load(path)
        saved_cfg = AppConfig.model_validate(p["cfg"])
        # keep the *current* data settings (paths, offline flag) but the *trained* model settings
        saved_cfg.data = cfg.data
        pipe = cls(cfg=saved_cfg, horizon=p["horizon"])
        for k in ("models", "bt", "oof", "metrics", "coverage", "trained_at", "trained_through"):
            setattr(pipe, k, p[k])
        pipe.feature_cols_ = p.get("feature_cols_", [])
        return pipe
