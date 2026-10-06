"""Typed configuration loaded from YAML (see ``configs/default.yaml``)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


def project_root() -> Path:
    """Repository root: ``$TESLA_FORECAST_HOME`` or the nearest parent holding pyproject.toml."""
    env = os.environ.get("TESLA_FORECAST_HOME")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "configs").exists():
            return parent
    return Path.cwd()


class DataConfig(BaseModel):
    ticker: str = "TSLA"
    start: str = "2015-01-01"
    context_tickers: list[str] = Field(default_factory=lambda: ["SPY", "QQQ", "^VIX"])
    use_context: bool = True
    local_csv: str = "data/raw/HistoricalData_tesla.csv"
    cache_dir: str = "data/processed"
    cache_ttl_hours: float = 6
    offline: bool = False


class FeatureConfig(BaseModel):
    warmup: int = 252


class DeepConfig(BaseModel):
    seq_len: int = 60
    max_epochs: int = 60
    patience: int = 8
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 1e-2
    dropout: float = 0.15
    hidden: int = 48
    n_seeds: int = 2
    device: str = "auto"


class ModelsConfig(BaseModel):
    enabled: list[str] = Field(
        default_factory=lambda: [
            "naive", "drift", "arima", "theta", "ridge",
            "lightgbm", "xgboost", "gru_attention", "tcn", "transformer",
        ]
    )
    deep: DeepConfig = Field(default_factory=DeepConfig)


class BacktestConfig(BaseModel):
    n_test_origins: int = 756
    refit_every: int = 63
    step: int = 1
    min_train: int = 504


class EnsembleConfig(BaseModel):
    window: int = 252
    min_obs: int = 40
    power: float = 2.0
    shrink_to_equal: float = 0.15


class IntervalConfig(BaseModel):
    levels: list[float] = Field(default_factory=lambda: [0.80, 0.95])
    window: int = 400
    min_obs: int = 60

    @field_validator("levels")
    @classmethod
    def _check_levels(cls, v: list[float]) -> list[float]:
        if not v or any(not 0 < x < 1 for x in v):
            raise ValueError("interval levels must lie strictly between 0 and 1")
        return sorted(v)


class SimulationConfig(BaseModel):
    n_paths: int = 5000
    seed: int = 42


class PathsConfig(BaseModel):
    artifacts_dir: str = "artifacts"
    reports_dir: str = "reports"


class AppConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    horizon: int = Field(10, ge=1, le=60)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    intervals: IntervalConfig = Field(default_factory=IntervalConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    seed: int = 42

    @property
    def root(self) -> Path:
        return project_root()

    def resolve(self, relative: str) -> Path:
        """Resolve a config path relative to the project root."""
        p = Path(relative).expanduser()
        return p if p.is_absolute() else self.root / p

    @property
    def artifacts_dir(self) -> Path:
        d = self.resolve(self.paths.artifacts_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def pipeline_path(self) -> Path:
        return self.artifacts_dir / "pipeline.joblib"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
    """Load ``configs/default.yaml``, then deep-merge ``path`` and ``overrides`` on top."""
    data: dict[str, Any] = {}
    default = project_root() / "configs" / "default.yaml"
    if default.exists():
        data = yaml.safe_load(default.read_text()) or {}
    if path:
        data = _deep_merge(data, yaml.safe_load(Path(path).read_text()) or {})
    if overrides:
        data = _deep_merge(data, overrides)
    return AppConfig.model_validate(data)
