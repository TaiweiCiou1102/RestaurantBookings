"""Regression-specific preprocessing: one-hot encoding, scaling, baseline dropping."""

import logging
from typing import Dict, List, Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Categorical columns that need one-hot encoding for regression
CATEGORICAL_COLS = [
    "dining_purpose",
    "booked_by_gender",
    "account_gender",
    "city_code",
    "city_area_code",
    "member_city_code",
]

# Integer-coded categorical columns that need one-hot encoding for regression
ORDINAL_TO_ONEHOT_COLS = [
    "booking_hour",
    "weekday",
    "popular_hour",
]

# Baseline categories to drop (first dummy) to avoid multicollinearity
BASELINE_DROP: Dict[str, str] = {
    "weekday": "weekday_1",       # Monday as baseline
    "booking_hour": "booking_hour_0",  # Midnight as baseline
}


def one_hot_encode(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    drop_baselines: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Applies one-hot encoding to specified columns.

    Args:
        df: Input DataFrame.
        columns: Columns to encode. Defaults to CATEGORICAL_COLS + ORDINAL_TO_ONEHOT_COLS.
        drop_baselines: Dict of {original_col: dummy_col_to_drop} for avoiding
            multicollinearity. Defaults to BASELINE_DROP.

    Returns:
        DataFrame with one-hot encoded columns.
    """
    if columns is None:
        columns = CATEGORICAL_COLS + ORDINAL_TO_ONEHOT_COLS
    if drop_baselines is None:
        drop_baselines = BASELINE_DROP

    result = df.copy()
    cols_to_encode = [c for c in columns if c in result.columns]

    result = pd.get_dummies(result, columns=cols_to_encode, dtype=int)

    # Drop baseline dummies to avoid dummy variable trap
    for orig_col, baseline_col in drop_baselines.items():
        if baseline_col in result.columns:
            result = result.drop(columns=[baseline_col])
            logger.debug("Dropped baseline column: %s", baseline_col)

    logger.info("One-hot encoded %d columns.", len(cols_to_encode))
    return result


def scale_numeric(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    exclude: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Applies StandardScaler to numeric columns, fit on train only.

    Args:
        df_train: Training features.
        df_test: Testing features.
        exclude: Numeric columns to exclude from scaling (e.g., binary flags).

    Returns:
        Scaled train, scaled test, fitted scaler.
    """
    if exclude is None:
        exclude = []

    numeric_cols = [
        c for c in df_train.select_dtypes(include="number").columns
        if c not in exclude
    ]

    scaler = StandardScaler()
    df_train_out = df_train.copy()
    df_test_out = df_test.copy()

    df_train_out[numeric_cols] = scaler.fit_transform(df_train[numeric_cols])
    df_test_out[numeric_cols] = scaler.transform(df_test[numeric_cols])

    logger.info("Scaled %d numeric columns.", len(numeric_cols))
    return df_train_out, df_test_out, scaler


def preprocess_for_regression(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Full regression preprocessing pipeline: one-hot → scale.

    Returns:
        Processed train, processed test, fitted scaler.
    """
    train_encoded = one_hot_encode(df_train)
    test_encoded = one_hot_encode(df_test)

    # Align columns (test may lack some dummies)
    train_encoded, test_encoded = train_encoded.align(
        test_encoded, join="left", axis=1, fill_value=0
    )

    train_scaled, test_scaled, scaler = scale_numeric(train_encoded, test_encoded)
    return train_scaled, test_scaled, scaler
