import pandas as pd

from tesla_forecast.data.calendar import is_trading_day, next_trading_days, nyse_holidays


def test_known_holidays_2026():
    hol = set(nyse_holidays(2026, 2026).date)
    from datetime import date

    assert date(2026, 11, 26) in hol  # Thanksgiving
    assert date(2026, 4, 3) in hol  # Good Friday
    assert date(2026, 12, 25) in hol  # Christmas (Friday)
    assert date(2026, 7, 3) in hol  # Independence Day observed (July 4 is a Saturday)
    assert date(2026, 6, 19) in hol  # Juneteenth


def test_observed_rules():
    assert not is_trading_day("2021-12-24") or True  # Christmas 2021 is Sat -> observed Fri Dec 24
    assert not is_trading_day("2021-12-24")
    assert is_trading_day("2021-12-31")  # NYSE open when New Year's Day falls on Saturday
    assert not is_trading_day("2022-01-17")  # MLK


def test_next_trading_days_skips_weekends_and_holidays():
    nxt = next_trading_days(pd.Timestamp("2026-11-24"), 3)  # Tue -> Wed, (Thu closed), Fri, Mon
    assert [d.strftime("%Y-%m-%d") for d in nxt] == ["2026-11-25", "2026-11-27", "2026-11-30"]
    assert all(d.weekday() < 5 for d in next_trading_days(pd.Timestamp("2026-10-06"), 30))
