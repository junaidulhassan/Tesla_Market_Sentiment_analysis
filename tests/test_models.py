import numpy as np
import pytest

from tesla_forecast.models.registry import ALL_MODELS, build_model

H = 5


@pytest.mark.parametrize("name", ALL_MODELS)
def test_model_fit_predict_shapes_and_causality(name, ff, fast_cfg):
    frame = ff.frame
    model = build_model(name, fast_cfg, H, ff.feature_cols)
    train = frame.iloc[:450]
    model.fit(train)
    origins = [455, 470, 500]
    p_full = model.predict(frame.iloc[:520], origins)
    assert p_full.shape == (3, H) and np.isfinite(p_full).all()
    # causality: forecasts from an origin must not change if later rows are removed / corrupted
    p_cut = model.predict(frame.iloc[: max(origins) + 1], origins)
    np.testing.assert_allclose(p_full, p_cut, rtol=1e-6, atol=1e-9)
    corrupted = frame.iloc[:520].copy()
    corrupted.iloc[max(origins) + 1:, :] = corrupted.iloc[max(origins) + 1:, :] * 7
    np.testing.assert_allclose(model.predict(corrupted, origins), p_full, rtol=1e-6, atol=1e-9)
    assert model.predict(frame.iloc[:520]).shape == (1, H)  # default = last row


def test_naive_is_zero_and_drift_linear(ff, fast_cfg):
    nv = build_model("naive", fast_cfg, H, ff.feature_cols).predict(ff.frame, [100, 200])
    assert not nv.any()
    dr = build_model("drift", fast_cfg, H, ff.feature_cols).predict(ff.frame, [300])[0]
    np.testing.assert_allclose(np.diff(dr), np.diff(dr)[0], atol=1e-12)


def test_unknown_model_raises(ff, fast_cfg):
    with pytest.raises(ValueError):
        build_model("nope", fast_cfg, H, ff.feature_cols)
