"""Chronology-safe class-prior baseline fitted on training labels only."""

from __future__ import annotations

import numpy as np


def class_prior_probabilities(train_labels: np.ndarray, target_count: int) -> np.ndarray:
    train_labels = np.asarray(train_labels, dtype=np.float64)
    if train_labels.size == 0:
        raise ValueError("Training labels cannot be empty")
    return np.full(target_count, float(train_labels.mean()), dtype=np.float64)