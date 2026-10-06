import numpy as np
import pandas as pd

from tesla_forecast.config import project_root
from tesla_forecast.data.loader import _drop_incomplete_bar, parse_legacy_csv, validate_ohlcv


def test_parse_legacy_csv():
    df = parse_legacy_csv(project_root() / "data/raw/HistoricalData_tesla.csv")
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.index[0] == pd.Timestamp("2015-02-19") and np.isclose(df["Close"].iloc[0], 14.1137)
    assert (df["High"] >= df["Low"]).all()


def test_validate_drops_bad_rows(prices):
    bad = prices.iloc[:50].copy()
    bad = pd.concat([bad, bad.iloc[[3]]])  # duplicate date
    bad.iloc[5, bad.columns.get_loc("Close")] = -1.0
    clean, notes = validate_ohlcv(bad)
    assert clean.index.is_unique and (clean["Close"] > 0).all()
    assert any("duplicate" in n for n in notes) and any("non-positive" in n for n in notes)


def test_incomplete_bar_dropped_during_session(prices):
    from datetime import datetime

    from tesla_forecast.data.loader import NY

    today = datetime(2026, 10, 7, 11, 0, tzinfo=NY)
    df = prices.iloc[:5].copy()
    df.index = pd.date_range(end="2026-10-07", periods=5, freq="B")
    out, notes = _drop_incomplete_bar(df, now=today)
    assert len(out) == 4 and notes
    out2, _ = _drop_incomplete_bar(df, now=datetime(2026, 10, 7, 17, 0, tzinfo=NY))
    assert len(out2) == 5
