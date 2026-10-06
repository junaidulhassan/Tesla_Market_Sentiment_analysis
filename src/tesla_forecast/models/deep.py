"""PyTorch sequence models for direct multi-horizon forecasting.

Three architectures share one training harness:

* ``GRUAttention``  - 2-layer GRU with temporal attention pooling
* ``TCN``           - dilated causal temporal convolutional network (residual blocks)
* ``TransformerNet``- small pre-LN Transformer encoder with learned positions

Each emits ``horizon`` standardised cumulative log-returns at once. Training uses a Huber loss, AdamW,
gradient clipping, input-noise augmentation, a time-ordered validation tail with early stopping, and a
small seed ensemble. These nets are intentionally compact: with ~3k daily samples, capacity is the
enemy of generalisation.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..config import DeepConfig
from ..utils import get_logger
from .base import Forecaster

log = get_logger(__name__)


# --------------------------------------------------------------------------- architectures
class GRUAttention(nn.Module):
    def __init__(self, n_features: int, horizon: int, hidden: int = 48, dropout: float = 0.15, **_):
        super().__init__()
        self.inp = nn.Linear(n_features, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(nn.LayerNorm(2 * hidden), nn.Dropout(dropout), nn.Linear(2 * hidden, hidden),
                                  nn.GELU(), nn.Linear(hidden, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(torch.relu(self.inp(x)))
        w = torch.softmax(self.attn(h).squeeze(-1), dim=1).unsqueeze(-1)
        return self.head(torch.cat([(w * h).sum(1), h[:, -1]], dim=-1))


class _CausalBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, dropout: float, kernel: int = 3):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.c1 = nn.Conv1d(ch, ch, kernel, dilation=dilation)
        self.c2 = nn.Conv1d(ch, ch, kernel, dilation=dilation)
        self.norm = nn.GroupNorm(1, ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.c1(nn.functional.pad(x, (self.pad, 0)))
        y = self.drop(torch.relu(self.norm(y)))
        y = self.c2(nn.functional.pad(y, (self.pad, 0)))
        return torch.relu(x + self.drop(y))


class TCN(nn.Module):
    def __init__(self, n_features: int, horizon: int, hidden: int = 48, dropout: float = 0.15, **_):
        super().__init__()
        ch = max(16, hidden * 2 // 3)
        self.inp = nn.Conv1d(n_features, ch, 1)
        self.blocks = nn.Sequential(*[_CausalBlock(ch, d, dropout) for d in (1, 2, 4, 8, 16)])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * ch, hidden), nn.GELU(), nn.Linear(hidden, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.inp(x.transpose(1, 2)))
        return self.head(torch.cat([h[:, :, -1], h.mean(dim=2)], dim=-1))


class TransformerNet(nn.Module):
    def __init__(self, n_features: int, horizon: int, hidden: int = 48, dropout: float = 0.15,
                 seq_len: int = 60, **_):
        super().__init__()
        d = max(16, (hidden // 4) * 4)
        self.inp = nn.Linear(n_features, d)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=2 * d, dropout=dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Dropout(dropout), nn.Linear(2 * d, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(self.inp(x) + self.pos[:, : x.shape[1]])
        return self.head(torch.cat([h[:, -1], h.mean(dim=1)], dim=-1))


ARCHITECTURES: dict[str, type[nn.Module]] = {
    "gru_attention": GRUAttention,
    "tcn": TCN,
    "transformer": TransformerNet,
}


def resolve_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(pref)


# --------------------------------------------------------------------------- forecaster
class DeepForecaster(Forecaster):
    """Sequence-to-horizon forecaster wrapping one of :data:`ARCHITECTURES` (seed-ensembled)."""

    def __init__(self, arch: str, horizon: int, feature_cols: list[str] | None = None,
                 cfg: DeepConfig | None = None, seed: int = 42, vol_norm: bool = True):
        super().__init__(horizon, feature_cols)
        self.vol_norm = vol_norm
        if arch not in ARCHITECTURES:
            raise ValueError(f"unknown architecture {arch!r}; choose from {sorted(ARCHITECTURES)}")
        self.arch, self.name = arch, arch
        self.cfg = cfg or DeepConfig()
        self.seed = seed
        self.nets_: list[nn.Module] = []
        self.mu_: np.ndarray | None = None
        self.sd_: np.ndarray | None = None
        self.y_sd_: np.ndarray | None = None
        self.history_: list[dict] = []

    # ---- data prep
    def _scaled(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.feature_cols].to_numpy(dtype=np.float32)
        return np.clip((x - self.mu_) / self.sd_, -5, 5).astype(np.float32)

    def _windows(self, xs: np.ndarray, ends: np.ndarray) -> np.ndarray:
        L = self.cfg.seq_len
        padded = np.concatenate([np.zeros((L - 1, xs.shape[1]), dtype=np.float32), xs])
        idx = ends[:, None] + np.arange(L)[None, :]  # window for end e covers padded[e : e+L]
        return padded[idx]

    # ---- training
    def fit(self, frame: pd.DataFrame) -> DeepForecaster:
        cfg, H = self.cfg, self.horizon
        x_raw = frame[self.feature_cols].to_numpy(dtype=np.float32)
        self.mu_, self.sd_ = x_raw.mean(0), x_raw.std(0) + 1e-6
        xs = self._scaled(frame)
        y = self._targets(frame) / self._row_scale(frame, self.vol_norm)
        ok = np.where(~np.isnan(y).any(axis=1))[0]
        y_ok = y[ok]
        med = np.median(y_ok, 0)
        mad = np.median(np.abs(y_ok - med), 0) * 1.4826 + 1e-9
        y_ok = np.clip(y_ok, med - 4 * mad, med + 4 * mad)
        self.y_sd_ = y_ok.std(0) + 1e-9
        yn = (y_ok / self.y_sd_).astype(np.float32)

        n_val = max(int(0.15 * len(ok)), 40)
        tr_ix, va_ix = np.arange(0, len(ok) - n_val - H), np.arange(len(ok) - n_val, len(ok))  # H-day embargo
        if len(tr_ix) < 100:
            raise ValueError("not enough training rows for a deep model")
        Xtr, Ytr = self._windows(xs, ok[tr_ix]), yn[tr_ix]
        Xva, Yva = self._windows(xs, ok[va_ix]), yn[va_ix]

        device = resolve_device(cfg.device)
        self.nets_, self.history_ = [], []
        for s in range(cfg.n_seeds):
            net = self._train_one(Xtr, Ytr, Xva, Yva, device, seed=self.seed + 1000 * s)
            self.nets_.append(net.cpu().eval())
        return self

    def _train_one(self, Xtr, Ytr, Xva, Yva, device, seed: int) -> nn.Module:
        cfg = self.cfg
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = ARCHITECTURES[self.arch](
            n_features=Xtr.shape[2], horizon=self.horizon, hidden=cfg.hidden,
            dropout=cfg.dropout, seq_len=cfg.seq_len,
        ).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
        loss_fn = nn.HuberLoss(delta=1.0)
        xva, yva = torch.from_numpy(Xva).to(device), torch.from_numpy(Yva).to(device)
        xtr, ytr = torch.from_numpy(Xtr).to(device), torch.from_numpy(Ytr).to(device)

        best, best_state, bad, hist = np.inf, copy.deepcopy(net.state_dict()), 0, []
        for epoch in range(cfg.max_epochs):
            net.train()
            perm = torch.from_numpy(rng.permutation(len(xtr))).to(device)
            tot = 0.0
            for i in range(0, len(perm), cfg.batch_size):
                b = perm[i: i + cfg.batch_size]
                xb = xtr[b] + 0.03 * torch.randn_like(xtr[b])  # input-noise augmentation
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(net(xb), ytr[b])
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                tot += loss.item() * len(b)
            net.eval()
            with torch.no_grad():
                val = loss_fn(net(xva), yva).item()
            sched.step(val)
            hist.append({"epoch": epoch + 1, "train_loss": tot / len(xtr), "val_loss": val})
            if val < best - 1e-5:
                best, best_state, bad = val, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        self.history_.append(hist)
        net.load_state_dict(best_state)
        return net

    # ---- inference
    @torch.no_grad()
    def predict(self, frame: pd.DataFrame, origins: Sequence[int] | None = None) -> np.ndarray:
        idx = self._origins(frame, origins)
        # only rows up to the largest origin are needed (and used)
        xs = self._scaled(frame.iloc[: idx.max() + 1])
        windows = torch.from_numpy(self._windows(xs, idx))
        preds = []
        for net in self.nets_:
            net.eval()
            out = torch.cat([net(windows[i: i + 512]) for i in range(0, len(windows), 512)]).numpy()
            preds.append(out)
        return np.mean(preds, axis=0) * self.y_sd_ * self._row_scale(frame, self.vol_norm)[idx]
