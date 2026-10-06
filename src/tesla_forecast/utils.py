"""Small shared helpers: logging, seeding, timing."""

from __future__ import annotations

import logging
import os
import random
import time
from contextlib import contextmanager

import numpy as np

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "tesla_forecast") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger("tesla_forecast").handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
        root = logging.getLogger("tesla_forecast")
        root.addHandler(handler)
        root.setLevel(os.environ.get("TESLA_FORECAST_LOG", "INFO").upper())
        root.propagate = False
    return logger


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and (if importable) torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - global seed for third-party libraries
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


@contextmanager
def timer(label: str, logger: logging.Logger | None = None):
    t0 = time.perf_counter()
    yield
    msg = f"{label} took {time.perf_counter() - t0:.1f}s"
    (logger or get_logger()).info(msg)
