from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from tesla_forecast.data.loader import PriceData
from tesla_forecast.forecasting import ForecastPipeline
from tesla_forecast.forecasting.simulation import simulate_paths

from .conftest import make_prices


@pytest.fixture(scope="module")
def pipe(fast_cfg):
    pd_ = PriceData(make_prices(), "synthetic", "TSLA", datetime.now())
    p = ForecastPipeline(fast_cfg).prepare(prices=pd_, load_context=False)
    p.backtest()
    p.fit()
    return p


def test_backtest_alignment_and_no_nan_preds(pipe):
    bt = pipe.bt
    n = len(bt.positions)
    assert n > 100 and all(v.shape == (n, 5) for v in bt.preds.values())
    assert all(np.isfinite(v).all() for v in bt.preds.values())
    assert np.isfinite(bt.actual[:-5]).all() and np.isnan(bt.actual[-1, -1])  # last origins unrealised
    assert set(pipe.metrics.model) == {*bt.preds, "ensemble"}


def test_intervals_nested_and_roughly_calibrated(pipe):
    (lo80, hi80), (lo95, hi95) = pipe.oof.bounds[0.8], pipe.oof.bounds[0.95]
    assert (lo95 <= lo80 + 1e-12).all() and (hi95 >= hi80 - 1e-12).all()
    cov = pipe.coverage.pivot(index="horizon", columns="level", values="empirical_coverage")
    assert 0.68 < cov[0.8].mean() < 0.92 and cov[0.95].mean() > 0.88


def test_forecast_structure(pipe):
    res = pipe.forecast()
    assert res.horizon == 5 and len(res.dates) == 5
    assert all(d.weekday() < 5 for d in res.dates) and res.dates[0] > res.as_of
    lo, hi = res.bounds[0.8]
    assert (lo < res.price).all() and (res.price < hi).all()
    assert ((res.prob_up >= 0) & (res.prob_up <= 1)).all()
    w = np.stack(list(res.weights.values()))
    np.testing.assert_allclose(w.sum(0), 1.0)
    assert res.to_frame().shape[0] == 5


def test_save_load_roundtrip(pipe, tmp_path):
    path = pipe.save(tmp_path / "p.joblib")
    loaded = ForecastPipeline.load(path, cfg=pipe.cfg).prepare(prices=pipe.prices, load_context=False)
    np.testing.assert_allclose(loaded.forecast().price, pipe.forecast().price, rtol=1e-6)


def test_missing_features_raises_clear_error(pipe):
    broken = pipe.ff.frame.drop(columns=[pipe.feature_cols_[0]])
    with pytest.raises(ValueError, match="missing"):
        pipe.forecast(broken)


def test_simulation_matches_drift_and_is_positive(pipe):
    res = pipe.forecast()
    sim = pipe.simulate(res)
    assert (sim.paths > 0).all() and sim.paths.shape == (500, 6)
    r = np.log(pipe.ff.frame["close"]).diff()
    flat = simulate_paths(100.0, np.zeros(5), r, res.dates, 4000, seed=1)
    assert abs(np.log(flat.terminal / 100).mean()) < 0.02
    assert {"prob_up_%", "VaR_95_%", "CVaR_95_%"} <= set(sim.risk_summary())


def test_simulation_dispersion_matches_volatility():
    """Terminal std of simulated log-returns must be close to sigma*sqrt(H) for constant-vol data."""
    rng = np.random.default_rng(0)
    sigma, H = 0.02, 10
    idx = pd.bdate_range("2020-01-01", periods=1500)
    r = pd.Series(rng.normal(0, sigma, 1500), index=idx)
    dates = pd.bdate_range("2026-01-05", periods=H)
    sim = simulate_paths(100.0, np.zeros(H), r, dates, 6000, seed=3)
    term = np.log(sim.terminal / 100)
    assert 0.85 < term.std() / (sigma * np.sqrt(H)) < 1.15
    assert np.percentile(term, 0.1) > -0.35  # no absurd -60% tail from a mis-seeded EWMA
