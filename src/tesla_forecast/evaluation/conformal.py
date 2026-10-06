"""Volatility-normalised, rolling split-conformal prediction intervals.

Score ``s = (y - y_hat) / sigma`` where ``sigma = ewma_daily_vol * sqrt(h)`` makes residuals roughly
homoskedastic, so the empirical quantiles of ``s`` transfer across calm and turbulent regimes.
Quantiles at origin ``i`` use only residuals already *realised* at ``i`` (no look-ahead), and the
finite-sample ``ceil((n+1)(1-a))/n`` correction guarantees marginal coverage under exchangeability.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def horizon_scale(vol_ewma: np.ndarray, horizon: int) -> np.ndarray:
    """``(n, H)`` scale = daily vol * sqrt(h)."""
    return np.asarray(vol_ewma, dtype=float)[:, None] * np.sqrt(np.arange(1, horizon + 1))[None, :]


def standardized_residuals(actual: np.ndarray, pred: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (actual - pred) / scale


def _conformal_quantile(s: np.ndarray, p: float) -> float:
    n = len(s)
    level = min(1.0, np.ceil((n + 1) * p) / n)
    return float(np.quantile(s, level, method="higher"))


def rolling_conformal_bounds(
    scores: np.ndarray,
    positions: np.ndarray,
    point: np.ndarray,
    scale: np.ndarray,
    levels: list[float],
    target_positions: np.ndarray | None = None,
    window: int = 400,
    min_obs: int = 60,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Lower/upper bounds in log-return space for each level -> arrays ``(n_target, H)``.

    ``scores`` are standardised OOF residuals ``(n, H)`` (NaN = unrealised) aligned to ``positions``;
    ``point`` and ``scale`` are for the target origins.
    """
    tp = positions if target_positions is None else np.asarray(target_positions)
    H = scores.shape[1]
    out = {lv: (np.zeros((len(tp), H)), np.zeros((len(tp), H))) for lv in levels}
    for h in range(1, H + 1):
        hi = np.searchsorted(positions + h, tp, side="right")
        lo = np.minimum(np.searchsorted(positions, tp - window, side="left"), hi)
        col = scores[:, h - 1]
        for i in range(len(tp)):
            s = col[lo[i]:hi[i]]
            s = s[np.isfinite(s)]
            for lv in levels:
                a = 1 - lv
                if len(s) >= min_obs:
                    q_lo, q_hi = -_conformal_quantile(-s, 1 - a / 2), _conformal_quantile(s, 1 - a / 2)
                else:  # cold start: Gaussian quantiles
                    z = stats.norm.ppf(1 - a / 2)
                    q_lo, q_hi = -z, z
                out[lv][0][i, h - 1] = point[i, h - 1] + q_lo * scale[i, h - 1]
                out[lv][1][i, h - 1] = point[i, h - 1] + q_hi * scale[i, h - 1]
    # Per-horizon quantiles are estimated independently, so enforce what must be true: uncertainty
    # never shrinks as the horizon grows (cumulative max of the offsets from the point forecast).
    for lv, (lo, hi) in out.items():
        out[lv] = (point - np.maximum.accumulate(point - lo, axis=1), point + np.maximum.accumulate(hi - point, axis=1))
    return out


def prob_up(scores: np.ndarray, point: np.ndarray, scale: np.ndarray, min_obs: int = 60) -> np.ndarray:
    """P(return > 0) per horizon from the empirical distribution of standardised residuals."""
    H = point.shape[-1]
    out = np.zeros(H)
    for h in range(H):
        s = scores[:, h]
        s = s[np.isfinite(s)]
        thr = -point[h] / scale[h]  # return > 0  <=>  s > -point/scale
        out[h] = np.mean(s > thr) if len(s) >= min_obs else stats.norm.sf(thr)
    return out
