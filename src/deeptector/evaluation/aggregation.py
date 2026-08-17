"""Frame-to-video score aggregation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def aggregate_scores(scores: Iterable[float], strategy: str = "mean", *, top_k: int = 3) -> float:
    """Aggregate uncalibrated frame scores for one video."""
    values = np.asarray(list(scores), dtype=float)
    if values.size == 0:
        raise ValueError("Cannot aggregate an empty score collection")
    if strategy == "mean":
        return float(values.mean())
    if strategy == "max":
        return float(values.max())
    if strategy == "top_k_mean":
        if top_k < 1:
            raise ValueError("top_k must be positive")
        return float(np.sort(values)[-min(top_k, values.size) :].mean())
    raise ValueError(f"Unsupported aggregation strategy: {strategy}")
