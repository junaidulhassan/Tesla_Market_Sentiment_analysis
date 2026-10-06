"""Forecast accuracy metrics.

Everything is computed on the *same* out-of-sample origins for every model. Price-level errors are
derived from return forecasts via ``P_hat = P_t * exp(r_hat)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _mask(*arrs: np.ndarray) -> np.ndarray:
    m = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m


def directional_accuracy(actual: np.ndarray, pred: np.ndarray) -> float:
    """Share of origins where the sign of the predicted return matches the realised sign."""
    m = _mask(actual, pred)
    if m.sum() == 0 or not np.any(pred[m] != 0):  # a constant-zero forecast makes no directional call
        return float("nan")
    return float(np.mean(np.sign(actual[m]) == np.sign(pred[m])))


def price_metrics(close_t: np.ndarray, actual_ret: np.ndarray, pred_ret: np.ndarray,
                  naive_mae_scale: float | None = None) -> dict[str, float]:
    """MAE / RMSE / MAPE / sMAPE / MASE in dollars for one horizon."""
    m = _mask(close_t, actual_ret, pred_ret)
    p_true, p_hat = close_t[m] * np.exp(actual_ret[m]), close_t[m] * np.exp(pred_ret[m])
    err = p_hat - p_true
    out = {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE_%": float(np.mean(np.abs(err) / p_true) * 100),
        "sMAPE_%": float(np.mean(2 * np.abs(err) / (np.abs(p_true) + np.abs(p_hat))) * 100),
    }
    scale = naive_mae_scale if naive_mae_scale is not None else float(np.mean(np.abs(close_t[m] * np.exp(actual_ret[m]) - close_t[m])))
    out["MASE"] = out["MAE"] / scale if scale > 0 else float("nan")
    return out


def oos_r2(actual: np.ndarray, pred: np.ndarray) -> float:
    """Campbell-Thompson out-of-sample R^2 of *returns* vs the zero-return (random walk) benchmark.

    Positive means the model beats the random walk; typical values for stocks are 0-2%.
    """
    m = _mask(actual, pred)
    a, p = actual[m], pred[m]
    denom = np.sum(a**2)
    return float(1 - np.sum((a - p) ** 2) / denom) if denom > 0 else float("nan")


def diebold_mariano(actual: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Diebold-Mariano test (squared-error loss) with Newey-West variance for h-step overlap.

    ``H0``: equal accuracy. Negative statistic => model A more accurate than B. Returns (stat, p-value).
    """
    m = _mask(actual, pred_a, pred_b)
    d = (actual[m] - pred_a[m]) ** 2 - (actual[m] - pred_b[m]) ** 2
    n = len(d)
    if n < 30 or np.allclose(d, 0):
        return float("nan"), float("nan")
    lag = max(h - 1, int(np.floor(n ** (1 / 3))))
    dm = d - d.mean()
    var = np.sum(dm**2) / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * np.sum(dm[k:] * dm[:-k]) / n
    if var <= 0:
        return float("nan"), float("nan")
    stat = d.mean() / np.sqrt(var / n)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)  # Harvey-Leybourne-Newbold correction
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))


def binom_p_value(actual: np.ndarray, pred: np.ndarray) -> float:
    """P-value that directional hit-rate exceeds 50% (one-sided, ignores overlap -> optimistic)."""
    m = _mask(actual, pred)
    if not np.any(pred[m] != 0):
        return float("nan")
    hits = int(np.sum(np.sign(actual[m]) == np.sign(pred[m])))
    return float(stats.binomtest(hits, int(m.sum()), 0.5, alternative="greater").pvalue) if m.sum() else float("nan")


def metrics_by_horizon(close_t: np.ndarray, actual: np.ndarray, preds: dict[str, np.ndarray],
                       horizons: list[int], benchmark: str = "naive") -> pd.DataFrame:
    """Long table of metrics: one row per (model, horizon)."""
    rows = []
    for h in horizons:
        j = h - 1
        a = actual[:, j]
        # scale for MASE: in-sample naive MAE at this horizon
        bench_price_err = close_t * np.exp(a) - close_t
        scale = float(np.nanmean(np.abs(bench_price_err)))
        bench_rmse = price_metrics(close_t, a, preds[benchmark][:, j], scale)["RMSE"] if benchmark in preds else np.nan
        for name, p in preds.items():
            pm = price_metrics(close_t, a, p[:, j], scale)
            dm_stat, dm_p = diebold_mariano(a, p[:, j], preds[benchmark][:, j], h) if benchmark in preds and name != benchmark else (np.nan, np.nan)
            rows.append({
                "model": name, "horizon": h, **pm,
                "DirAcc_%": directional_accuracy(a, p[:, j]) * 100,
                "OOS_R2_ret_%": oos_r2(a, p[:, j]) * 100,
                "Skill_vs_naive_%": (1 - pm["RMSE"] / bench_rmse) * 100 if bench_rmse else np.nan,
                "DM_stat": dm_stat, "DM_p": dm_p,
                "IC": float(stats.spearmanr(a[_mask(a, p[:, j])], p[_mask(a, p[:, j]), j])[0]) if np.std(p[:, j]) > 0 else np.nan,
                "n": int(_mask(a, p[:, j]).sum()),
            })
    return pd.DataFrame(rows)


def coverage_table(actual: np.ndarray, bounds: dict[float, tuple[np.ndarray, np.ndarray]],
                   horizons: list[int]) -> pd.DataFrame:
    """Empirical coverage + mean width (log-return units) per nominal level and horizon."""
    rows = []
    for level, (lo, hi) in bounds.items():
        for h in horizons:
            j = h - 1
            m = _mask(actual[:, j], lo[:, j], hi[:, j])
            inside = (actual[m, j] >= lo[m, j]) & (actual[m, j] <= hi[m, j])
            rows.append({"level": level, "horizon": h, "empirical_coverage": float(inside.mean()) if m.any() else np.nan,
                         "mean_width": float(np.mean(hi[m, j] - lo[m, j])) if m.any() else np.nan, "n": int(m.sum())})
    return pd.DataFrame(rows)
