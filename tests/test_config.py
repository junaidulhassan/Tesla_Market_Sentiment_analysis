import pytest
from pydantic import ValidationError

from tesla_forecast import load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg.horizon == 10 and "ridge" in cfg.models.enabled and cfg.intervals.levels == [0.8, 0.95]


def test_overrides_deep_merge():
    cfg = load_config(overrides={"models": {"deep": {"hidden": 8}}, "horizon": 7})
    assert cfg.models.deep.hidden == 8 and cfg.models.deep.n_seeds == 2 and cfg.horizon == 7


def test_validation_rejects_bad_values():
    with pytest.raises(ValidationError):
        load_config(overrides={"horizon": 0})
    with pytest.raises(ValidationError):
        load_config(overrides={"intervals": {"levels": [1.5]}})
