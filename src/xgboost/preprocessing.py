"""Preprocessing for XGBoost: shared utilities + tree-based label encoding."""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

logger = logging.getLogger(__name__)

# Columns not used as model features.
# Geographic codes dropped 2026-06-22 (see docs/feature_decisions.md):
#  - city_area_code: high-cardinality (38 cats, 65% missing), gain importance
#    was an overfitting artifact.
#  - city_code / member_city_code: head-to-head CV showed the model prefers
#    continuous lat/lng over nominal codes; the codes actively hurt (codes-only
#    scored below no-geo in every subset). lat/lng kept as the geo signal.
NON_FEATURE_COLS = [
    "booking_time", "reservation_time", "member_birthdate",
    "city_area_code", "city_code", "member_city_code",
]

# String categorical columns converted to native category dtype for XGBoost
CATEGORICAL_COLS = [
    "dining_purpose",
    "booked_by_gender",
    "account_gender",
]


# ------------------------------------------------------------------
#  Data loading & splitting (from common.py)
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
#  Label encoding (from tree.py)
# ------------------------------------------------------------------

def _clean_str(series: pd.Series) -> pd.Series:
    """Normalise a column to clean strings, filling missing with 'Unknown'.

    Casts via ``object`` first so it works regardless of the input dtype —
    including columns that are already ``category`` dtype (NaN cannot be filled
    in-place on a Categorical) or numeric with NaN.
    """
    return series.astype("object").fillna("Unknown").astype(str)


def label_encode(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, OrdinalEncoder]]:
    """Applies OrdinalEncoder to categorical columns, fit on train only.

    Unseen categories in test are mapped to -1.

    Args:
        df_train: Training features.
        df_test: Testing features.
        columns: Columns to encode. Defaults to CATEGORICAL_COLS.

    Returns:
        Encoded train, encoded test, dict of fitted encoders.
    """
    if columns is None:
        columns = CATEGORICAL_COLS

    train_out = df_train.copy()
    test_out = df_test.copy()
    encoders: Dict[str, OrdinalEncoder] = {}

    for col in columns:
        if col not in train_out.columns:
            continue

        train_out[col] = train_out[col].fillna("Unknown").astype(str)
        test_out[col] = test_out[col].fillna("Unknown").astype(str)

        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

        train_out[col] = enc.fit_transform(train_out[[col]])
        test_out[col] = enc.transform(test_out[[col]])

        encoders[col] = enc
        logger.debug("Label encoded column: %s", col)

    logger.info("Label encoded %d columns.", len(encoders))
    return train_out, test_out, encoders


def encode_categoricals(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.CategoricalDtype]]:
    """Convert categorical columns to pandas ``category`` dtype (XGBoost native).

    Unlike :func:`label_encode`, this applies **no numeric encoding**. With
    ``enable_categorical=True`` XGBoost reads the category codes directly and
    performs proper partition-based splits, instead of treating nominal codes
    as an ordered axis. Categories are fitted on *train* only; values unseen in
    train become NaN in *test*, which XGBoost handles as missing.

    Args:
        df_train: Training features.
        df_test: Testing features.
        columns: Columns to convert. Defaults to CATEGORICAL_COLS.

    Returns:
        Converted train, converted test, dict of fitted CategoricalDtype.
    """
    if columns is None:
        columns = CATEGORICAL_COLS

    train_out = df_train.copy()
    test_out = df_test.copy()
    cat_dtypes: Dict[str, pd.CategoricalDtype] = {}

    for col in columns:
        if col not in train_out.columns:
            continue

        # Cast via object first so this is robust to inputs that are already
        # category dtype (e.g. re-processing inside CV) or numeric with NaN.
        train_col = _clean_str(train_out[col]).astype("category")
        dtype = train_col.dtype  # CategoricalDtype carrying the train category set

        train_out[col] = train_col
        # Align test to train categories; unseen values -> NaN (missing for XGBoost)
        test_out[col] = _clean_str(test_out[col]).astype(dtype)

        cat_dtypes[col] = dtype
        logger.debug("Converted column to category dtype: %s (%d categories)", col, len(dtype.categories))

    logger.info("Converted %d columns to category dtype.", len(cat_dtypes))
    return train_out, test_out, cat_dtypes


def apply_saved_encoders(
    df: pd.DataFrame,
    encoders: Dict[str, object],
) -> pd.DataFrame:
    """Transform categorical columns using pre-fitted artifacts (inference only).

    Supports CategoricalDtype (native categorical, current), OrdinalEncoder, and
    LabelEncoder (legacy) saved artifacts. For CategoricalDtype, values unseen in
    train become NaN (missing); for the legacy encoders, unknown categories map to -1.

    Args:
        df: Input features.
        encoders: Dict of column name → fitted artifact (from training).

    Returns:
        DataFrame with categorical columns converted/encoded.
    """
    out = df.copy()
    for col, enc in encoders.items():
        if col not in out.columns:
            continue
        out[col] = _clean_str(out[col])
        if isinstance(enc, pd.CategoricalDtype):
            out[col] = out[col].astype(enc)
        elif isinstance(enc, OrdinalEncoder):
            out[col] = enc.transform(out[[col]])
        else:
            # LabelEncoder: map unseen labels to -1
            known = set(enc.classes_)
            out[col] = out[col].apply(lambda v: enc.transform([v])[0] if v in known else -1)
    return out


def preprocess_for_tree(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.CategoricalDtype]]:
    """Full tree-based preprocessing pipeline: native categorical dtype only.

    Categorical columns become pandas ``category`` dtype (consumed directly by
    XGBoost with ``enable_categorical=True``). Integer columns (weekday,
    booking_hour, popular_hour) are kept as-is. No scaling is applied.

    Returns:
        Processed train, processed test, dict of fitted CategoricalDtype.
    """
    return encode_categoricals(df_train, df_test)
