import numpy as np
import pandas as pd

from tesla_forecast.features.engineering import build_frame, forward_log_returns


def test_no_nans_and_close_not_a_feature(ff):
    assert not ff.frame[ff.feature_cols].isna().any().any()
    assert "close" not in ff.feature_cols and np.isfinite(ff.frame[ff.feature_cols].to_numpy()).all()


def test_features_are_causal(prices):
    """Row t of the feature table must not change when future data is removed or altered."""
    full = build_frame(prices, None, warmup=252).frame
    cut = 700
    part = build_frame(prices.iloc[:cut], None, warmup=252).frame
    common = part.index
    pd.testing.assert_frame_equal(full.loc[common], part, check_exact=False, rtol=1e-9, atol=1e-12)
    tampered = prices.copy()
    tampered.iloc[cut:] *= 3.0
    t = build_frame(tampered, None, warmup=252).frame.loc[common]
    pd.testing.assert_frame_equal(t, part, check_exact=False, rtol=1e-9, atol=1e-12)


def test_forward_log_returns():
    close = np.array([100.0, 110.0, 121.0, 100.0])
    y = forward_log_returns(close, 2)
    assert np.isclose(y[0, 0], np.log(1.1)) and np.isclose(y[0, 1], np.log(1.21))
    assert np.isnan(y[2, 1]) and np.isnan(y[3, 0]) and y.shape == (4, 2)
