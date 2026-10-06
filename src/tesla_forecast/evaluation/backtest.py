"""Expanding-window walk-forward backtest with periodic refitting.

For each fold the models are fitted on rows ``< fold_start`` only and then asked to forecast from every
origin in the fold using data up to (and including) that origin. Nothing after an origin is ever
visible to a model, so out-of-sample metrics are honest estimates of live performance.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..config import AppConfig
from ..features.engineering import ForecastFrame, forward_log_returns
from ..utils import get_logger

log = get_logger(__name__)


@dataclass
class BacktestResult:
    horizon: int
    positions: np.ndarray  # integer row positions of the forecast origins within the frame
    dates: pd.DatetimeIndex
    close: np.ndarray  # price at each origin
    vol: np.ndarray  # EWMA daily vol known at each origin
    actual: np.ndarray  # (n, H) realised cumulative log-returns (NaN if not yet realised)
    preds: dict[str, np.ndarray]  # model -> (n, H) predicted cumulative log-returns
    folds: list[dict] = field(default_factory=list)
    failures: dict[str, int] = field(default_factory=dict)
    fit_seconds: dict[str, float] = field(default_factory=dict)

    @property
    def models(self) -> list[str]:
        return list(self.preds)

    def price_frame(self, model: str, h: int) -> pd.DataFrame:
        """Actual vs predicted price at horizon ``h``, indexed by the *target* date (origin + h rows)."""
        a, p = self.actual[:, h - 1], self.preds[model][:, h - 1]
        return pd.DataFrame(
            {"origin": self.dates, "actual": self.close * np.exp(a), "predicted": self.close * np.exp(p),
             "base": self.close}, index=self.dates)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> BacktestResult:
        return joblib.load(path)


def plan_origins(n_rows: int, cfg: AppConfig) -> tuple[int, np.ndarray]:
    """First test position and the array of origin positions."""
    bt = cfg.backtest
    first = max(bt.min_train, n_rows - 1 - bt.n_test_origins)
    if first >= n_rows - 2:
        raise ValueError(f"not enough history ({n_rows} rows) for min_train={bt.min_train}")
    return first, np.arange(first, n_rows - 1, bt.step)  # last row has no realised target


def run_walk_forward(
    ff: ForecastFrame,
    factories: dict[str, Callable[[], object]],
    cfg: AppConfig,
    horizon: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> BacktestResult:
    """Run every model through the same folds and return aligned out-of-sample forecasts."""
    H = horizon or cfg.horizon
    frame = ff.frame
    first, origins = plan_origins(len(frame), cfg)
    refit = cfg.backtest.refit_every
    bounds = list(range(first, len(frame) - 1, refit))
    preds = {k: np.full((len(origins), H), np.nan) for k in factories}
    failures = dict.fromkeys(factories, 0)
    secs = dict.fromkeys(factories, 0.0)
    folds: list[dict] = []
    say = progress or log.info

    for fi, start in enumerate(bounds):
        end = min(start + refit, len(frame) - 1)
        in_fold = np.where((origins >= start) & (origins < end))[0]
        if len(in_fold) == 0:
            continue
        train = frame.iloc[:start]
        view = frame.iloc[: origins[in_fold].max() + 1]  # nothing beyond the last origin is visible
        folds.append({"fold": fi, "train_end": frame.index[start - 1], "n_train": len(train),
                      "origins": (frame.index[origins[in_fold[0]]], frame.index[origins[in_fold[-1]]])})
        say(f"fold {fi + 1}/{len(bounds)}: train<{frame.index[start].date()} ({len(train)} rows), "
            f"{len(in_fold)} origins")
        for name, make in factories.items():
            t0 = time.perf_counter()
            try:
                model = make()
                model.fit(train)
                preds[name][in_fold] = model.predict(view, origins[in_fold])
            except Exception as exc:  # one failing model/fold must not sink the whole backtest
                failures[name] += 1
                log.warning("model %s failed in fold %d: %s", name, fi, exc)
            secs[name] += time.perf_counter() - t0

    # drop models that never produced a forecast; zero-fill (random walk) isolated failed folds
    for name in list(preds):
        if np.isnan(preds[name]).all():
            log.error("model %s produced no forecasts and is excluded", name)
            preds.pop(name)
        else:
            preds[name] = np.where(np.isnan(preds[name]), 0.0, preds[name])
    actual_all = forward_log_returns(frame["close"].to_numpy(), H)
    return BacktestResult(
        horizon=H, positions=origins, dates=frame.index[origins],
        close=frame["close"].to_numpy()[origins], vol=frame["vol_ewma"].to_numpy()[origins],
        actual=actual_all[origins], preds=preds, folds=folds, failures=failures, fit_seconds=secs,
    )
