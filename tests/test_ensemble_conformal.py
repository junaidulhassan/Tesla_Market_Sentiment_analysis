import numpy as np

from tesla_forecast.evaluation.conformal import rolling_conformal_bounds, standardized_residuals
from tesla_forecast.evaluation.ensemble import online_inverse_error_weights, weighted_combination


def _toy(n=300, H=3, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.arange(1000, 1000 + n)
    actual = rng.normal(0, 1, (n, H))
    preds = {"good": actual + rng.normal(0, 0.3, (n, H)), "bad": actual + rng.normal(0, 2.0, (n, H))}
    return pos, actual, preds


def test_weights_sum_to_one_and_prefer_better_model():
    pos, actual, preds = _toy()
    names, W = online_inverse_error_weights(preds, actual, pos, window=200, min_obs=30)
    assert np.allclose(W.sum(-1), 1.0)
    assert W[-1, 0, names.index("good")] > 0.8
    assert np.allclose(W[0], 0.5)  # cold start -> equal weights


def test_weights_have_no_lookahead():
    """Weights at origin i must not depend on outcomes that are unrealised at i."""
    pos, actual, preds = _toy()
    names, W1 = online_inverse_error_weights(preds, actual, pos, window=200, min_obs=30)
    i, H = 150, actual.shape[1]
    a2 = actual.copy()
    for h in range(1, H + 1):  # destroy every outcome not yet realised at position pos[i]
        bad = pos + h > pos[i]
        a2[bad, h - 1] = 1e6
    _, W2 = online_inverse_error_weights(preds, a2, pos, window=200, min_obs=30)
    assert np.allclose(W1[i], W2[i])


def test_weighted_combination_shape():
    pos, actual, preds = _toy()
    names, W = online_inverse_error_weights(preds, actual, pos)
    assert weighted_combination(preds, names, W).shape == actual.shape


def test_conformal_coverage_close_to_nominal():
    rng = np.random.default_rng(3)
    n, H = 1500, 2
    pos = np.arange(n)
    vol = np.full(n, 0.02)
    scale = vol[:, None] * np.sqrt(np.arange(1, H + 1))[None, :]
    point = np.zeros((n, H))
    actual = rng.standard_t(5, (n, H)) * scale
    scores = standardized_residuals(actual, point, scale)
    b = rolling_conformal_bounds(scores, pos, point, scale, [0.8, 0.95], window=400, min_obs=60)
    for lv, (lo, hi) in b.items():
        cov = np.mean((actual[600:] >= lo[600:]) & (actual[600:] <= hi[600:]))
        assert abs(cov - lv) < 0.03, (lv, cov)


def test_interval_widths_never_shrink_with_horizon():
    rng = np.random.default_rng(5)
    n, H = 400, 6
    pos = np.arange(n)
    scale = np.full((n, H), 0.02) * np.sqrt(np.arange(1, H + 1))
    point = rng.normal(0, 0.01, (n, H))
    scores = rng.standard_t(4, (n, H))
    b = rolling_conformal_bounds(scores, pos, point, scale, [0.8], window=200, min_obs=30)
    lo, hi = b[0.8]
    assert (np.diff(hi - point, axis=1) >= -1e-12).all() and (np.diff(point - lo, axis=1) >= -1e-12).all()
