"""Scoring helpers for steering runs."""

from __future__ import annotations

import numpy as np


def severity_from_score(score: float) -> int:
    """Map z-scored probe value to severity bucket."""
    if score >= 2.0:
        return 3
    if score >= 1.0:
        return 2
    if score >= 0.0:
        return 1
    return 0


def score_projection(activation: np.ndarray, vector: np.ndarray) -> float:
    return float(np.dot(activation, vector))


def zscore(raw: float, *, mean: float, std: float) -> float:
    denom = std if abs(std) > 1e-6 else 1.0
    return float((raw - mean) / denom)
