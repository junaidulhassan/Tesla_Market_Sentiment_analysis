"""Static (PNG) report figures built with matplotlib - used for the README and ``reports/figures``."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..features.indicators import add_display_indicators  # noqa: E402
from .theme import PALETTE, apply_matplotlib_style  # noqa: E402

BLUE, ORANGE, AQUA, VIOLET, GREY = PALETTE["blue"], PALETTE["orange"], PALETTE["aqua"], PALETTE["violet"], "#8a8a85"


def _save(fig, out: Path, name: str) -> Path:
    path = out / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return path


def technical_chart(prices: pd.DataFrame, out: Path, last_n: int = 250) -> Path:
    d = add_display_indicators(prices).iloc[-last_n:]
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})
    ax[0].fill_between(d.index, d.bb_lower, d.bb_upper, color=BLUE, alpha=0.1, label="Bollinger (20, 2σ)")
    ax[0].plot(d.index, d.Close, color=BLUE, lw=1.8, label="Close")
    for c, col in (("sma_20", AQUA), ("sma_50", VIOLET), ("sma_200", PALETTE["magenta"])):
        ax[0].plot(d.index, d[c], color=col, lw=1.1, label=c.upper().replace("_", " "))
    ax[0].set_title("TSLA price, trend and momentum"); ax[0].set_ylabel("USD"); ax[0].legend(ncol=5, loc="upper left")
    up = d.Close >= d.Open
    ax[1].bar(d.index, d.Volume / 1e6, color=np.where(up, BLUE, ORANGE), width=1.0); ax[1].set_ylabel("Volume (M)")
    ax[2].plot(d.index, d.rsi_14, color=VIOLET)
    for lvl in (30, 70):
        ax[2].axhline(lvl, color=GREY, ls=":")
    ax[2].set_ylim(0, 100); ax[2].set_ylabel("RSI 14")
    return _save(fig, out, "technical_chart")


def model_leaderboard(metrics: pd.DataFrame, out: Path, horizons: tuple[int, ...] = (1, 5, 10)) -> Path:
    horizons = tuple(h for h in horizons if h in set(metrics.horizon))
    fig, ax = plt.subplots(1, len(horizons), figsize=(5.2 * len(horizons), 4.6), sharex=True)
    ax = np.atleast_1d(ax)
    for a, h in zip(ax, horizons, strict=True):
        d = metrics[(metrics.horizon == h) & (metrics.model != "naive")].sort_values("Skill_vs_naive_%")
        a.barh(d.model, d["Skill_vs_naive_%"], color=[ORANGE if m == "ensemble" else BLUE for m in d.model])
        a.axvline(0, color=GREY, lw=1); a.set_title(f"t+{h}"); a.set_xlabel("RMSE gain vs random walk (%)")
        for y, v in enumerate(d["Skill_vs_naive_%"]):
            a.text(v, y, f" {v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    fig.suptitle("Model skill vs the random-walk benchmark (positive = better)", fontweight="semibold")
    return _save(fig, out, "model_leaderboard")


def skill_by_horizon(metrics: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for m in sorted(set(metrics.model) - {"naive", "ensemble"}):
        d = metrics[metrics.model == m]
        ax.plot(d.horizon, d["Skill_vs_naive_%"], color=GREY, alpha=0.5, lw=1)
        ax.text(d.horizon.iloc[-1] + 0.1, d["Skill_vs_naive_%"].iloc[-1], m, fontsize=7, color=GREY, va="center")
    e = metrics[metrics.model == "ensemble"]
    ax.plot(e.horizon, e["Skill_vs_naive_%"], color=ORANGE, lw=2.6, marker="o", label="ensemble")
    ax.axhline(0, color=GREY, ls=":"); ax.set_xticks(sorted(set(metrics.horizon)))
    ax.set_xlabel("Forecast horizon (trading days)"); ax.set_ylabel("RMSE skill vs naive (%)")
    ax.set_title("Skill vs random walk across horizons (above 0 = better)"); ax.legend(loc="lower left")
    return _save(fig, out, "skill_by_horizon")


def calibration(coverage: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for lv, c in zip(sorted(coverage.level.unique()), (BLUE, ORANGE, AQUA), strict=False):
        d = coverage[coverage.level == lv]
        ax.plot(d.horizon, d.empirical_coverage * 100, marker="o", color=c, label=f"{int(lv * 100)}% interval (empirical)")
        ax.axhline(lv * 100, color=c, ls=":", lw=1)
    ax.set_ylim(60, 100); ax.set_xticks(sorted(coverage.horizon.unique()))
    ax.set_xlabel("Horizon (trading days)"); ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("Interval calibration on out-of-sample data (dotted = nominal)"); ax.legend(loc="lower left")
    return _save(fig, out, "interval_calibration")


def backtest_intervals(bt, oof, out: Path, h: int = 5, last_n: int = 250, level: float = 0.8) -> Path:
    j, sl = h - 1, slice(-last_n, None)
    idx = bt.dates[sl] + pd.tseries.offsets.BDay(h)  # plot at the target date
    base = bt.close[sl]
    lo, hi = oof.bounds[level][0][sl, j], oof.bounds[level][1][sl, j]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.fill_between(idx, base * np.exp(lo), base * np.exp(hi), color=ORANGE, alpha=0.2, label=f"{int(level * 100)}% interval")
    ax.plot(idx, base * np.exp(bt.actual[sl, j]), color=BLUE, lw=1.8, label="Actual price")
    ax.plot(idx, base * np.exp(oof.pred[sl, j]), color=ORANGE, lw=1.3, label=f"Ensemble forecast (made {h} days earlier)")
    ax.set_ylabel("USD"); ax.legend(loc="upper left"); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.set_title(f"Walk-forward backtest, t+{h}: realised price vs out-of-sample forecast and interval")
    return _save(fig, out, f"backtest_t{h}_intervals")


def ensemble_weights(bt, oof, out: Path, h: int = 5) -> Path:
    fig, ax = plt.subplots(figsize=(12, 3.8))
    im = ax.imshow(oof.weights[:, h - 1, :].T * 100, aspect="auto", cmap="Blues", vmin=0,
                   extent=[mdates.date2num(bt.dates[0]), mdates.date2num(bt.dates[-1]), len(oof.names), 0])
    ax.set_yticks(np.arange(len(oof.names)) + 0.5, oof.names); ax.xaxis_date(); ax.grid(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    fig.colorbar(im, ax=ax, label="weight (%)", shrink=0.9)
    ax.set_title(f"Online ensemble weights over time (t+{h})")
    return _save(fig, out, f"ensemble_weights_t{h}")


def risk_simulation(sim, out: Path) -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.7, 1]})
    x = [sim.dates[0] - pd.tseries.offsets.BDay(1), *sim.dates]
    for (q, a), alpha in (((5, 95), 0.15), ((25, 75), 0.3)):
        ax[0].fill_between(x, np.percentile(sim.paths, q, axis=0), np.percentile(sim.paths, a, axis=0), color=ORANGE, alpha=alpha, label=f"{q}–{a} percentile")
    ax[0].plot(x, np.median(sim.paths, axis=0), color=ORANGE, lw=2.4, label="median")
    ax[0].set_title("Simulated price paths"); ax[0].set_ylabel("USD"); ax[0].legend(loc="upper left")
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ret = (sim.terminal / sim.last_close - 1) * 100
    ax[1].hist(ret, bins=50, color=BLUE, alpha=0.85); ax[1].axvline(0, color=GREY, ls=":")
    ax[1].axvline(np.percentile(ret, 5), color=ORANGE, lw=2, label=f"VaR 95%: {np.percentile(ret, 5):.1f}%")
    ax[1].set_title("Terminal return distribution"); ax[1].set_xlabel("Return (%)"); ax[1].legend()
    return _save(fig, out, "risk_simulation")


def save_report_figures(pipe, result, sim, out_dir: str | Path) -> list[Path]:
    """Write the curated static figures for a fitted pipeline + forecast + simulation."""
    apply_matplotlib_style()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return [
        technical_chart(pipe.prices.df, out), model_leaderboard(pipe.metrics, out, (1, 5, pipe.horizon)),
        skill_by_horizon(pipe.metrics, out), calibration(pipe.coverage, out),
        backtest_intervals(pipe.bt, pipe.oof, out), ensemble_weights(pipe.bt, pipe.oof, out),
        risk_simulation(sim, out),
    ]
