"""NYSE trading calendar (weekends + exchange holidays), so forecasts never land on a closed day."""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import pandas as pd
from dateutil.easter import easter

# One-off exchange closures that follow no rule.
_SPECIAL_CLOSURES = {
    date(2007, 1, 2),    # National Day of Mourning (President Ford)
    date(2012, 10, 29),  # Hurricane Sandy
    date(2012, 10, 30),
    date(2018, 12, 5),   # National Day of Mourning (President G.H.W. Bush)
    date(2025, 1, 9),    # National Day of Mourning (President Carter)
}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th ``weekday`` (Mon=0) of a month; n=-1 means the last one."""
    if n > 0:
        first = date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + timedelta(days=offset + 7 * (n - 1))
    last = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(d: date) -> date | None:
    """Saturday holidays are observed Friday, Sunday holidays Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=64)
def _holidays_for_year(year: int) -> frozenset[date]:
    days: set[date] = set()
    ny = date(year, 1, 1)
    if ny.weekday() != 5:  # NYSE does not close the prior Friday when Jan 1 is a Saturday
        days.add(ny + timedelta(days=1) if ny.weekday() == 6 else ny)
    days.add(_nth_weekday(year, 1, 0, 3))  # Martin Luther King Jr. Day
    days.add(_nth_weekday(year, 2, 0, 3))  # Presidents' Day
    days.add(easter(year) - timedelta(days=2))  # Good Friday
    days.add(_nth_weekday(year, 5, 0, -1))  # Memorial Day
    if year >= 2022:
        days.add(_observed(date(year, 6, 19)))  # Juneteenth
    days.add(_observed(date(year, 7, 4)))  # Independence Day
    days.add(_nth_weekday(year, 9, 0, 1))  # Labor Day
    days.add(_nth_weekday(year, 11, 3, 4))  # Thanksgiving
    days.add(_observed(date(year, 12, 25)))  # Christmas
    days |= {d for d in _SPECIAL_CLOSURES if d.year == year}
    return frozenset(d for d in days if d is not None)


def nyse_holidays(start_year: int, end_year: int) -> pd.DatetimeIndex:
    """All NYSE holidays falling on weekdays, ``start_year..end_year`` inclusive."""
    days = sorted(d for y in range(start_year, end_year + 1) for d in _holidays_for_year(y))
    return pd.DatetimeIndex(days)


def is_trading_day(d: date | pd.Timestamp) -> bool:
    d = pd.Timestamp(d).date()
    return d.weekday() < 5 and d not in _holidays_for_year(d.year)


def next_trading_days(last: date | pd.Timestamp, n: int) -> pd.DatetimeIndex:
    """The ``n`` NYSE trading days strictly after ``last``."""
    out: list[pd.Timestamp] = []
    cur = pd.Timestamp(last).normalize()
    while len(out) < n:
        cur += pd.Timedelta(days=1)
        if is_trading_day(cur):
            out.append(cur)
    return pd.DatetimeIndex(out)
