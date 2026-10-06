"""Headline retrieval (Google News RSS first, yfinance news as a backup). Fails soft: returns []."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import pandas as pd
import requests

from ..utils import get_logger

log = get_logger(__name__)
_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) tesla-forecast/2.0"}


def _norm(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def _google_news(query: str, limit: int, timeout: float) -> list[dict]:
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    resp = requests.get(url, headers=_UA, timeout=timeout)
    resp.raise_for_status()
    out = []
    for item in ET.fromstring(resp.content).iter("item"):
        title = (item.findtext("title") or "").strip()
        src = item.findtext("source") or ""
        if src and title.endswith(f" - {src}"):
            title = title[: -len(src) - 3]
        try:
            ts = pd.Timestamp(parsedate_to_datetime(item.findtext("pubDate"))).tz_convert(None)
        except Exception:
            ts = pd.NaT
        out.append({"title": title, "source": src, "published": ts, "link": item.findtext("link") or ""})
        if len(out) >= limit:
            break
    return out


def _yahoo_news(ticker: str, limit: int) -> list[dict]:
    import yfinance as yf

    out = []
    for n in (yf.Ticker(ticker).news or [])[:limit]:
        c = n.get("content", n)
        title = c.get("title") or n.get("title")
        if not title:
            continue
        ts = pd.to_datetime(c.get("pubDate") or n.get("providerPublishTime"), errors="coerce", utc=True)
        out.append({"title": title, "source": (c.get("provider") or {}).get("displayName", n.get("publisher", "")),
                    "published": ts.tz_convert(None) if pd.notna(ts) else pd.NaT,
                    "link": (c.get("canonicalUrl") or {}).get("url", n.get("link", ""))})
    return out


def fetch_headlines(ticker: str = "TSLA", query: str | None = None, limit: int = 30,
                    timeout: float = 8.0) -> pd.DataFrame:
    """Recent, de-duplicated headlines as a DataFrame[title, source, published, link] (newest first)."""
    rows: list[dict] = []
    for name, fn in (("google", lambda: _google_news(query or f"{ticker} Tesla stock", limit, timeout)),
                     ("yahoo", lambda: _yahoo_news(ticker, limit))):
        try:
            rows += fn()
        except Exception as exc:
            log.warning("news source %s unavailable: %s", name, exc)
        if len(rows) >= limit:
            break
    seen, uniq = set(), []
    for r in rows:
        key = _norm(r["title"])
        if key and key not in seen:
            seen.add(key)
            uniq.append(r)
    df = pd.DataFrame(uniq, columns=["title", "source", "published", "link"])
    if len(df):
        df = df.sort_values("published", ascending=False, na_position="last").head(limit).reset_index(drop=True)
    return df
