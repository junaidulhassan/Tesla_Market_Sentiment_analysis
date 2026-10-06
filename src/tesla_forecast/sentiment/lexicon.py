"""A small finance-oriented lexicon scorer: transparent, offline, and deliberately simple.

It is a *display* signal (headline tone). It is NOT used as a forecasting feature: there is no
point-in-time news history in the training data, so it could not be validated honestly.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

POSITIVE = set("""beat beats surge surges surged soar soars soared rally rallies jump jumps gain gains gained growth
record upgrade upgrades upgraded outperform bullish strong stronger profit profits profitable boost boosts rise rises
rose climb climbs climbed optimism optimistic breakthrough approval approved expands expansion win wins accelerate
accelerates upside recovery rebound rebounds exceeds exceeded raised raises higher buy rating momentum milestone
success successful improves improved improvement demand robust bull highs surpass surpasses top tops""".split())
NEGATIVE = set("""miss misses missed plunge plunges plunged slump slumps fall falls fell drop drops dropped decline declines
declined cut cuts downgrade downgrades downgraded bearish weak weaker loss losses lawsuit sued probe investigation
recall recalls crash crashes slide slides tumble tumbles concern concerns risk risks warning warns warned fear fears
selloff layoffs layoff delay delays delayed fraud halt halts sink sinks lower slash slashes underperform struggle
struggles turmoil fine fined penalty scrutiny slows slowdown decrease decreased pressure threat tariff tariffs
disappoint disappoints disappointing lawsuit bear lows slumped""".split())
NEGATORS = {"not", "no", "never", "without", "fails", "fail", "despite", "unlikely"}
BOOSTERS = {"sharply", "heavily", "massive", "major", "significantly", "record"}


def score_headline(text: str) -> float:
    """Tone in [-1, 1]: (pos - neg) / (pos + neg + 1) with simple negation/intensifier handling."""
    tokens = re.findall(r"[a-z']+", text.lower())
    pos = neg = 0.0
    for i, tok in enumerate(tokens):
        w = 1.5 if i > 0 and tokens[i - 1] in BOOSTERS else 1.0
        flip = any(t in NEGATORS for t in tokens[max(0, i - 3): i])
        if tok in POSITIVE:
            pos, neg = (pos, neg + w) if flip else (pos + w, neg)
        elif tok in NEGATIVE:
            pos, neg = (pos + w, neg) if flip else (pos, neg + w)
    return float((pos - neg) / (pos + neg + 1.0))


def label(score: float, thr: float = 0.15) -> str:
    return "positive" if score > thr else "negative" if score < -thr else "neutral"


def sentiment_index(headlines: pd.DataFrame, half_life_hours: float = 48.0) -> dict:
    """Recency-weighted average tone plus counts. Returns ``{}`` for an empty frame."""
    if headlines is None or headlines.empty:
        return {}
    df = headlines.copy()
    df["score"] = df["title"].map(score_headline)
    df["label"] = df["score"].map(label)
    age_h = (pd.Timestamp.now() - df["published"]).dt.total_seconds().div(3600).fillna(half_life_hours * 3)
    w = 0.5 ** (age_h.clip(lower=0) / half_life_hours)
    index = float(np.average(df["score"], weights=w)) if w.sum() > 0 else float(df["score"].mean())
    return {
        "index": index, "label": label(index, 0.05), "n": len(df),
        "positive": int((df["label"] == "positive").sum()), "negative": int((df["label"] == "negative").sum()),
        "neutral": int((df["label"] == "neutral").sum()), "scored": df,
    }
