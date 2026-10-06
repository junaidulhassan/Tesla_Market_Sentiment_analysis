import numpy as np

from tesla_forecast.evaluation.metrics import (
    diebold_mariano,
    directional_accuracy,
    oos_r2,
    price_metrics,
)


def test_price_metrics_known_values():
    close = np.array([100.0, 100.0])
    actual = np.log(np.array([110.0, 90.0]) / 100)
    pred = np.log(np.array([105.0, 95.0]) / 100)
    m = price_metrics(close, actual, pred)
    assert np.isclose(m["MAE"], 5.0) and np.isclose(m["RMSE"], 5.0)
    assert np.isclose(m["MAPE_%"], (5 / 110 + 5 / 90) / 2 * 100)


def test_directional_accuracy_and_zero_forecast():
    a = np.array([1.0, -1.0, 1.0, -1.0])
    assert directional_accuracy(a, np.array([1, -1, -1, -1.0])) == 0.75
    assert np.isnan(directional_accuracy(a, np.zeros(4)))


def test_oos_r2_signs():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 500)
    assert oos_r2(y, y) == 1.0 and oos_r2(y, np.zeros_like(y)) == 0.0 and oos_r2(y, -y) < 0


def test_diebold_mariano_detects_better_model():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 800)
    good, bad = y + rng.normal(0, 0.2, 800), y + rng.normal(0, 1.0, 800)
    stat, p = diebold_mariano(y, good, bad, h=1)
    assert stat < 0 and p < 0.01
    _, p_same = diebold_mariano(y, bad, bad + rng.normal(0, 1e-3, 800), h=1)
    assert p_same > 0.05
