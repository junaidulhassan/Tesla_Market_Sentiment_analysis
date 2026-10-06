import pandas as pd

from tesla_forecast.sentiment import (
    generate_insights,
    score_headline,
    sentiment_index,
    technical_snapshot,
)
from tesla_forecast.sentiment.insights import rule_based_insights


def test_lexicon_scoring():
    assert score_headline("Tesla stock surges after record deliveries beat estimates") > 0.4
    assert score_headline("Tesla shares plunge on recall and lawsuit concerns") < -0.4
    assert score_headline("Tesla not expected to beat estimates") < score_headline("Tesla expected to beat estimates")
    assert score_headline("Tesla holds annual meeting") == 0


def test_sentiment_index_recency_weighted():
    now = pd.Timestamp.now()
    df = pd.DataFrame({"title": ["Tesla surges to record high", "Tesla plunges on recall"],
                       "published": [now, now - pd.Timedelta(days=10)], "source": ["a", "b"], "link": ["", ""]})
    s = sentiment_index(df)
    assert s["index"] > 0 and s["n"] == 2 and sentiment_index(pd.DataFrame()) == {}


def test_insights_rules_and_llm_paths(prices, monkeypatch):
    from langchain_core.language_models.fake import FakeListLLM

    snap = technical_snapshot(prices)
    fc = {"horizon": 5, "price_end": 101.0, "change_%": 1.0, "lo80": 95.0, "hi80": 107.0, "prob_up_%": 55.0}
    monkeypatch.setenv("LLM_PROVIDER", "none")
    out, src = generate_insights(snap, fc, None, [])
    assert src == "rules" and set(out) == set(rule_based_insights(snap))
    keys = list(out)
    good = FakeListLLM(responses=["Sure! " + str({k: "ok" for k in keys}).replace("'", '"')])
    out2, src2 = generate_insights(snap, fc, None, ["h1"], llm=good)
    assert src2 == "llm" and out2["prediction"] == "ok"
    bad = FakeListLLM(responses=["not json at all"])
    assert generate_insights(snap, fc, None, [], llm=bad)[1] == "rules"
