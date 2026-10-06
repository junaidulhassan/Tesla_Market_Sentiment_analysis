"""Market insights: a deterministic rule-based summary, optionally rewritten by an LLM.

The LLM never produces numbers: it receives the computed technicals/forecast/headlines and only
writes prose. With no provider configured (or on any failure) the rule-based text is returned, so the
app always has insights and never needs an API key.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd

from ..features.indicators import add_display_indicators
from ..utils import get_logger

log = get_logger(__name__)

PROMPT = """You are a careful equity-research assistant. Using ONLY the JSON below (computed from market data,
a statistical forecast and news headlines), write brief insights for TSLA. Do not invent numbers or events.
Return ONLY valid JSON with exactly these keys, each a 2-3 sentence string:
trend_analysis, technical_indicators, market_sentiment, economic_factors, prediction, risks.

DATA:
{data}
"""


def technical_snapshot(prices: pd.DataFrame) -> dict:
    """Latest technical readings as plain floats."""
    d = add_display_indicators(prices)
    c, last = d["Close"], d.iloc[-1]
    r = np.log(c).diff()
    hi, lo = c.iloc[-252:].max(), c.iloc[-252:].min()
    return {
        "close": float(last["Close"]), "ret_1d_%": float(c.pct_change().iloc[-1] * 100),
        "ret_5d_%": float(c.pct_change(5).iloc[-1] * 100), "ret_21d_%": float(c.pct_change(21).iloc[-1] * 100),
        "vs_sma20_%": float((last["Close"] / last["sma_20"] - 1) * 100),
        "vs_sma50_%": float((last["Close"] / last["sma_50"] - 1) * 100),
        "vs_sma200_%": float((last["Close"] / last["sma_200"] - 1) * 100) if pd.notna(last["sma_200"]) else np.nan,
        "rsi_14": float(last["rsi_14"]), "macd_hist": float(last["macd_hist"]), "bb_pctb": float(last["bb_pctb"]),
        "atr_%": float(last["atr_14"] / last["Close"] * 100),
        "realized_vol_21d_ann_%": float(r.iloc[-21:].std() * np.sqrt(252) * 100),
        "from_52w_high_%": float((last["Close"] / hi - 1) * 100), "from_52w_low_%": float((last["Close"] / lo - 1) * 100),
        "volume_vs_21d_avg": float(last["Volume"] / d["Volume"].iloc[-21:].mean()),
    }


def rule_based_insights(snap: dict, forecast: dict | None = None, sentiment: dict | None = None) -> dict:
    """Deterministic text from the numbers (also the fallback when no LLM is available)."""
    s = snap
    trend = ("uptrend" if s["vs_sma50_%"] > 0 and s["vs_sma200_%"] > 0 else
             "downtrend" if s["vs_sma50_%"] < 0 and s["vs_sma200_%"] < 0 else "mixed trend")
    mom = "overbought" if s["rsi_14"] > 70 else "oversold" if s["rsi_14"] < 30 else "neutral"
    out = {
        "trend_analysis": (f"TSLA is in a {trend}: {s['vs_sma50_%']:+.1f}% vs the 50-day and {s['vs_sma200_%']:+.1f}% vs the "
                           f"200-day average. Returns: {s['ret_5d_%']:+.1f}% over 5 days and {s['ret_21d_%']:+.1f}% over 21 days; "
                           f"price is {abs(s['from_52w_high_%']):.1f}% below its 52-week high."),
        "technical_indicators": (f"RSI(14) is {s['rsi_14']:.0f} ({mom}); MACD histogram is "
                                 f"{'positive' if s['macd_hist'] > 0 else 'negative'}; Bollinger %B is {s['bb_pctb']:.2f}. "
                                 f"21-day realised volatility is {s['realized_vol_21d_ann_%']:.0f}% annualised "
                                 f"(ATR {s['atr_%']:.1f}% of price); volume is {s['volume_vs_21d_avg']:.1f}x its 21-day average."),
        "market_sentiment": "Headline sentiment unavailable (offline or no news returned).",
        "economic_factors": ("Not derived from data here. Typical drivers for TSLA: rates and growth expectations, EV demand "
                             "and pricing, delivery numbers, regulation/tariffs, and broader tech risk appetite."),
        "prediction": "No forecast supplied.",
        "risks": "Statistical forecasts of stocks are weak; large gap moves on news can fall outside any interval.",
    }
    if sentiment:
        out["market_sentiment"] = (f"Recency-weighted headline tone is {sentiment['label']} (index {sentiment['index']:+.2f}) across "
                                   f"{sentiment['n']} headlines: {sentiment['positive']} positive, {sentiment['negative']} negative, "
                                   f"{sentiment['neutral']} neutral.")
    if forecast:
        f = forecast
        out["prediction"] = (f"The ensemble expects ${f['price_end']:,.2f} in {f['horizon']} trading days "
                             f"({f['change_%']:+.1f}%), 80% interval ${f['lo80']:,.0f}-${f['hi80']:,.0f}, with a model-implied "
                             f"{f['prob_up_%']:.0f}% chance of finishing above today's close. Treat the direction as low-confidence: "
                             f"the central estimate is small relative to the interval width.")
    return out


def build_llm(provider: str | None = None):
    """Construct a LangChain LLM from env config, or ``None`` when disabled/unconfigured."""
    provider = (provider or os.environ.get("LLM_PROVIDER", "none")).lower()
    if provider in ("", "none"):
        return None
    if provider == "huggingface":
        token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            return None
        from langchain_huggingface import HuggingFaceEndpoint

        return HuggingFaceEndpoint(repo_id=os.environ.get("HF_REPO_ID", "mistralai/Mistral-7B-Instruct-v0.3"),
                                   huggingfacehub_api_token=token, task="text-generation",
                                   max_new_tokens=900, temperature=0.1)
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            return None
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.1)
    raise ValueError(f"unknown LLM_PROVIDER {provider!r} (use none | huggingface | openai)")


def _parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def generate_insights(snapshot: dict, forecast: dict | None = None, sentiment: dict | None = None,
                      headlines: list[str] | None = None, llm=None) -> tuple[dict, str]:
    """Return ``(insights, source)`` where source is ``"llm"`` or ``"rules"``."""
    base = rule_based_insights(snapshot, forecast, sentiment)
    try:
        llm = llm if llm is not None else build_llm()
    except Exception as exc:
        log.warning("LLM unavailable: %s", exc)
        llm = None
    if llm is None:
        return base, "rules"
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import PromptTemplate

        payload = json.dumps({"technicals": {k: round(v, 3) for k, v in snapshot.items()}, "forecast": forecast,
                              "sentiment": {k: v for k, v in (sentiment or {}).items() if k != "scored"},
                              "headlines": (headlines or [])[:12]}, default=str, indent=1)
        chain = PromptTemplate.from_template(PROMPT) | llm | StrOutputParser()
        parsed = _parse_json(str(chain.invoke({"data": payload})))
        if parsed and all(k in parsed for k in base):
            return {k: str(parsed[k]) for k in base}, "llm"
        log.warning("LLM returned unusable JSON; using rule-based insights")
    except Exception as exc:
        log.warning("LLM call failed (%s); using rule-based insights", exc)
    return base, "rules"
