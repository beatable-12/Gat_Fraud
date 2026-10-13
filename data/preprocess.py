"""Feature contract and chronological dataset partitioning."""

from __future__ import annotations

import pandas as pd

from config import TRAIN_FRACTION, VALIDATION_FRACTION
from data.generate_data import FEATURE_NAMES


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ordered transactions without shuffling or crossing timestamps."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than 1")
    ordered = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    train_end = int(len(ordered) * train_fraction)
    validation_end = train_end + int(len(ordered) * validation_fraction)
    return ordered.iloc[:train_end].copy(), ordered.iloc[train_end:validation_end].copy(), ordered.iloc[validation_end:].copy()


def feature_matrix(frame: pd.DataFrame) -> object:
    """Return the canonical 16-column transaction feature matrix."""
    if len(FEATURE_NAMES) != 16:
        raise RuntimeError("The transaction feature contract must contain exactly 16 features")
    return frame.loc[:, FEATURE_NAMES].to_numpy(dtype="float32", copy=True)