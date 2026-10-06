"""Price data access: yfinance (live) -> disk cache -> bundled legacy CSV, with validation.

The loader never raises on a flaky network: it degrades through the sources above and records
what it used in :class:`PriceData` so the UI can tell the user.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..utils import get_logger

log = get_logger(__name__)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
NY = ZoneInfo("America/New_York")


@dataclass
class PriceData:
    """OHLCV history plus provenance."""

    df: pd.DataFrame
    source: str  # "yfinance" | "cache" | "stale-cache" | "legacy-csv"
    ticker: str
    fetched_at: datetime
    notes: list[str] = field(default_factory=list)

    @property
    def last_date(self) -> pd.Timestamp:
        return self.df.index[-1]

    @property
    def is_live(self) -> bool:
        return self.source in {"yfinance", "cache"}


# --------------------------------------------------------------------------- parsing & cleaning
def parse_legacy_csv(path: str | Path) -> pd.DataFrame:
    """Parse the Nasdaq-style export used by the original project ("$354.11" strings, newest first)."""
    raw = pd.read_csv(path)
    raw = raw.rename(columns={"Close/Last": "Close"})
    out = pd.DataFrame({"Date": pd.to_datetime(raw["Date"], format="mixed")})
    for col in ["Open", "High", "Low", "Close"]:
        out[col] = pd.to_numeric(raw[col].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    out["Volume"] = pd.to_numeric(raw["Volume"].astype(str).str.replace(",", ""), errors="coerce")
    return out.set_index("Date").sort_index()[OHLCV]


def validate_ohlcv(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Sort, de-duplicate, drop invalid rows. Returns ``(clean_df, notes)``."""
    notes: list[str] = []
    df = df.copy()
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).tz_localize(None).normalize()
    df.index.name = "Date"
    df = df.sort_index()
    if df.index.has_duplicates:
        n = int(df.index.duplicated().sum())
        df = df[~df.index.duplicated(keep="last")]
        notes.append(f"dropped {n} duplicate dates")
    df = df[OHLCV].astype(float)
    bad = df[["Open", "High", "Low", "Close"]].isna().any(axis=1) | (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
    if bad.any():
        notes.append(f"dropped {int(bad.sum())} rows with missing/non-positive prices")
        df = df[~bad]
    df["Volume"] = df["Volume"].fillna(0.0)
    inconsistent = (df["High"] < df[["Open", "Close", "Low"]].max(axis=1) - 1e-9) | (
        df["Low"] > df[["Open", "Close", "High"]].min(axis=1) + 1e-9
    )
    if inconsistent.any():  # repair rather than drop: keep the calendar intact
        notes.append(f"repaired {int(inconsistent.sum())} rows with inconsistent High/Low")
        df["High"] = df[["Open", "High", "Low", "Close"]].max(axis=1)
        df["Low"] = df[["Open", "High", "Low", "Close"]].min(axis=1)
    jumps = np.log(df["Close"]).diff().abs()
    if (jumps > 0.4).any():
        notes.append(f"{int((jumps > 0.4).sum())} daily moves >40% (check for unadjusted splits)")
    return df, notes


def _drop_incomplete_bar(df: pd.DataFrame, now: datetime | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Remove today's bar while the NYSE session is still open (it is not a final close)."""
    now = (now or datetime.now(NY)).astimezone(NY)
    if len(df) and df.index[-1].date() == now.date() and (now.hour, now.minute) < (16, 5):
        return df.iloc[:-1], [f"dropped incomplete intraday bar for {now.date()}"]
    return df, []


# --------------------------------------------------------------------------- network
def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def download_ohlcv(ticker: str, start: str, retries: int = 3, pause: float = 2.0) -> pd.DataFrame:
    """Download split/dividend-adjusted daily bars with retry + back-off."""
    import yfinance as yf

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(ticker, start=start, auto_adjust=True, progress=False, threads=False)
            df = _flatten(df)
            if df is not None and len(df) > 0 and "Close" in df.columns:
                return df
            last_err = RuntimeError(f"empty response for {ticker}")
        except Exception as exc:  # network / rate-limit / parsing
            last_err = exc
        log.warning("download %s failed (attempt %d/%d): %s", ticker, attempt, retries, last_err)
        time.sleep(pause * attempt)
    raise ConnectionError(f"could not download {ticker}: {last_err}")


def _cache_path(cfg: AppConfig, name: str) -> Path:
    d = cfg.resolve(cfg.data.cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    safe = name.replace("^", "").replace("/", "_")
    return d / f"{safe}_ohlcv.csv"


def _read_cache(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return None


def _cache_age_hours(path: Path) -> float:
    return (time.time() - path.stat().st_mtime) / 3600 if path.exists() else float("inf")


# --------------------------------------------------------------------------- public API
def load_prices(cfg: AppConfig, force_refresh: bool = False) -> PriceData:
    """Load TSLA OHLCV using the best available source (see module docstring)."""
    ticker = cfg.data.ticker
    cache = _cache_path(cfg, ticker)
    notes: list[str] = []
    df: pd.DataFrame | None = None
    source = ""

    fresh = _cache_age_hours(cache) < cfg.data.cache_ttl_hours
    if not cfg.data.offline and (force_refresh or not fresh):
        try:
            df = download_ohlcv(ticker, cfg.data.start)
            source = "yfinance"
        except ConnectionError as exc:
            notes.append(f"live download failed: {exc}")
    if df is None and cache.exists():
        cached = _read_cache(cache)
        if cached is not None and len(cached):
            df, source = cached, "cache" if fresh else "stale-cache"
            if source == "stale-cache":
                notes.append(f"using cached data ({_cache_age_hours(cache):.0f}h old)")
    if df is None:
        legacy = cfg.resolve(cfg.data.local_csv)
        if not legacy.exists():
            raise FileNotFoundError(f"no data source available (tried network, cache, {legacy})")
        df, source = parse_legacy_csv(legacy), "legacy-csv"
        notes.append("using bundled legacy CSV (ends Feb-2025): forecasts will be stale")

    df, clean_notes = validate_ohlcv(df)
    notes += clean_notes
    df, bar_notes = _drop_incomplete_bar(df) if source == "yfinance" else (df, [])
    notes += bar_notes
    if source == "yfinance":
        df.to_csv(cache)
    if len(df) < cfg.backtest.min_train + cfg.features.warmup + cfg.horizon + 50:
        notes.append(f"short history ({len(df)} rows); backtest settings may be reduced automatically")
    return PriceData(df=df, source=source, ticker=ticker, fetched_at=datetime.now(), notes=notes)


def load_market_context(cfg: AppConfig, index: pd.DatetimeIndex | None = None) -> pd.DataFrame | None:
    """Close prices of context tickers (SPY, QQQ, VIX) as ``spy``, ``qqq``, ``vix`` columns.

    Returns ``None`` if disabled/offline/unavailable - context features are strictly optional.
    """
    if not cfg.data.use_context or cfg.data.offline or not cfg.data.context_tickers:
        return None
    cols: dict[str, pd.Series] = {}
    for tk in cfg.data.context_tickers:
        path = _cache_path(cfg, tk)
        df = None
        if _cache_age_hours(path) < cfg.data.cache_ttl_hours:
            df = _read_cache(path)
        if df is None:
            try:
                df = _flatten(download_ohlcv(tk, cfg.data.start, retries=2, pause=1.0))
                df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
                df.to_csv(path)
            except ConnectionError:
                df = _read_cache(path)  # stale is better than nothing
        if df is not None and "Close" in df.columns and len(df):
            cols[tk.replace("^", "").lower()] = df["Close"].astype(float)
    if not cols:
        return None
    ctx = pd.DataFrame(cols)
    ctx.index = pd.DatetimeIndex(ctx.index).tz_localize(None).normalize()
    ctx = ctx[~ctx.index.duplicated(keep="last")].sort_index()
    if index is not None:
        ctx = ctx.reindex(index).ffill()
    return ctx
