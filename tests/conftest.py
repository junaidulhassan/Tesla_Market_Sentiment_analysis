from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tesla_forecast import load_config
from tesla_forecast.features.engineering import build_frame


def make_prices(n: int = 900, seed: int = 0) -> pd.DataFrame:
    """Random walk with GARCH-like volatility clustering and a weak momentum term."""
    rng = np.random.default_rng(seed)
    vol, r, prev = np.empty(n), np.empty(n), 0.0
    v = 0.02
    for i in range(n):
        v = np.sqrt(1e-6 + 0.08 * prev**2 + 0.9 * v**2)
        vol[i] = v
        r[i] = 0.0005 + 0.03 * prev + v * rng.standard_t(6) / np.sqrt(1.5)
        prev = r[i]
    close = 100 * np.exp(np.cumsum(r))
    open_ = close * np.exp(rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.006, n)))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                         "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float)}, index=idx)


@pytest.fixture(scope="session")
def prices() -> pd.DataFrame:
    return make_prices()


@pytest.fixture(scope="session")
def ff(prices):
    return build_frame(prices, None, warmup=252)


@pytest.fixture(scope="session")
def fast_cfg():
    return load_config(overrides={
        "horizon": 5,
        "data": {"offline": True, "use_context": False},
        "models": {"enabled": ["naive", "drift", "theta", "ridge", "lightgbm"],
                   "deep": {"seq_len": 20, "max_epochs": 3, "n_seeds": 1, "hidden": 16, "patience": 2}},
        "backtest": {"n_test_origins": 120, "refit_every": 60, "min_train": 300},
        "ensemble": {"min_obs": 20},
        "intervals": {"min_obs": 20, "window": 200},
        "simulation": {"n_paths": 500},
    })
