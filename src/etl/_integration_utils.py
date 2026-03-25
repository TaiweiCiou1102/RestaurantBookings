"""Utility functions for data integration and merging."""

import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

def check_duplicate_keys(df: pd.DataFrame, key_col: str) -> bool:
    """Checks if the specified key column contains duplicate values.

    This prevents unintended Cartesian products during dataframe merging.

    Args:
        df: The pandas DataFrame to check.
        key_col: The column name serving as the key.

    Returns:
        True if duplicates exist, False otherwise.
    """
    return df.duplicated(subset=[key_col]).any()

def safe_merge(
    left: pd.DataFrame, 
    right: pd.DataFrame, 
    on_cols: List[str], 
    how: str = "left"
) -> pd.DataFrame:
    """Safely merges two DataFrames and logs potential issues.

    Args:
        left: The left DataFrame.
        right: The right DataFrame.
        on_cols: A list of column names to merge on.
        how: The type of merge to perform ('left', 'inner', 'right', 'outer'). Defaults to "left".

    Returns:
        The merged pandas DataFrame.
    """
    logger.debug("Merging dataframes on columns: %s with strategy: %s", on_cols, how)
    return pd.merge(left, right, on=on_cols, how=how)
