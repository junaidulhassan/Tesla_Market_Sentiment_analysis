"""Leak-free online forecast combination.

For origin ``i`` and horizon ``h`` the weights only use errors of earlier origins ``j`` whose h-day
outcome had *already been observed* at time ``pos_i`` (``pos_j + h <= pos_i``). The same function
yields production weights (origin = "now"), so evaluation and deployment match.
"""

from __future__ import annotations

import numpy as np


def online_inverse_error_weights(
    preds: dict[str, np.ndarray],
    actual: np.ndarray,
    positions: np.ndarray,
    target_positions: np.ndarray | None = None,
    window: int = 252,
    min_obs: int = 40,
    power: float = 2.0,
    shrink_to_equal: float = 0.15,
    exclude: tuple[str, ...] = (),
) -> tuple[list[str], np.ndarray]:
    """Weights of shape ``(n_target, H, n_models)`` summing to 1 over models.

    ``positions`` are the integer row positions of the (sorted) backtest origins; ``target_positions``
    are the origins to compute weights for (default: the same origins).
    """
    names = [n for n in preds if n not in exclude]
    M, H = len(names), actual.shape[1]
    tp = positions if target_positions is None else np.asarray(target_positions)
    se = np.stack([(preds[k] - actual) ** 2 for k in names], axis=-1)  # (n, H, M); NaN where unrealised
    valid = np.isfinite(actual)
    se = np.where(valid[..., None], se, 0.0)
    cnt = valid.astype(float)
    cs = np.concatenate([np.zeros((1, H, M)), np.cumsum(se, axis=0)])
    cc = np.concatenate([np.zeros((1, H)), np.cumsum(cnt, axis=0)])

    W = np.full((len(tp), H, M), 1.0 / M)
    for h in range(1, H + 1):
        # number of origins whose h-day outcome is known at each target position
        hi = np.searchsorted(positions + h, tp, side="right")  # origins j with pos_j + h <= tp
        lo = np.searchsorted(positions, tp - window, side="left")
        lo = np.minimum(lo, hi)
        for i in range(len(tp)):
            a, b = lo[i], hi[i]
            k = cc[b, h - 1] - cc[a, h - 1]
            if k < min_obs:
                continue
            mse = (cs[b, h - 1] - cs[a, h - 1]) / k + 1e-18
            w = mse ** (-power / 2)
            w /= w.sum()
            W[i, h - 1] = (1 - shrink_to_equal) * w + shrink_to_equal / M
    return names, W


def weighted_combination(preds: dict[str, np.ndarray], names: list[str], weights: np.ndarray) -> np.ndarray:
    """Combine ``(n, H)`` forecasts with weights ``(n, H, M)``."""
    stacked = np.stack([preds[k] for k in names], axis=-1)
    return np.sum(stacked * weights, axis=-1)
