"""Shared preprocessing utilities for all model types."""

import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Columns not used as model features
NON_FEATURE_COLS = ["booking_time", "reservation_time", "member_birthdate"]


def load_features(path: str = "data/processed/features_ready.csv") -> pd.DataFrame:
    """Loads the feature-engineered dataset."""
    logger.info("Loading features from %s", path)
    return pd.read_csv(path)


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits into train/test sets, dropping non-feature columns.

    Args:
        df: Input DataFrame.
        target_col: Name of the target column.
        test_size: Fraction of data used for testing.
        random_state: Random seed for reproducibility.
        shuffle: If True, randomly shuffle before splitting (default).
            Set to False for temporal splits (caller should sort beforehand).

    Returns:
        X_train, X_test, y_train, y_test
    """
    drop_cols = [c for c in NON_FEATURE_COLS + [target_col] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=shuffle,
    )
    logger.info("Split data: train=%d, test=%d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


def fill_missing(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Fills missing values for numeric columns.

    Args:
        df: Input DataFrame.
        strategy: 'median', 'mean', 'none' (keep missing as-is), or 'drop' (drop rows with missing values).
    """
    result = df.copy()
    if strategy == "none":
        return result
    numeric_cols = result.select_dtypes(include="number").columns
    if strategy == "median":
        result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].median())
    else:
        result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].mean())
    return result

