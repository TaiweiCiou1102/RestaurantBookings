"""Utility functions for data cleaning and preprocessing."""

import logging
from typing import Dict, List, Any

import pandas as pd

logger = logging.getLogger(__name__)

def show_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and displays the number and percentage of missing values per column.

    Args:
        df: The input pandas DataFrame.

    Returns:
        A pandas DataFrame displaying the count and percentage of missing values,
        sorted in descending order.
    """
    missing_count = df.isnull().sum()
    missing_percent = 100 * df.isnull().sum() / len(df)
    
    missing_table = pd.concat([missing_count, missing_percent], axis=1)
    missing_table.columns = ['Amount of Missing Values', 'Percentage (%)']
    
    missing_table = missing_table[missing_table['Amount of Missing Values'] > 0]
    return missing_table.sort_values('Amount of Missing Values', ascending=False)

def fill_missing_values(
    df: pd.DataFrame, 
    col_name: str, 
    fill_value: Any = "Unknown"
) -> pd.DataFrame:
    """Fills missing values in a specific column with a designated value.

    Args:
        df: The input pandas DataFrame.
        col_name: The name of the column containing missing values.
        fill_value: The value to replace missing data with. Defaults to "Unknown".

    Returns:
        A new DataFrame with missing values filled in the specified column.
    """
    df_clean = df.copy()
    df_clean[col_name] = df_clean[col_name].fillna(fill_value)
    return df_clean

def get_duplicates(df: pd.DataFrame, subset: List[str]) -> pd.DataFrame:
    """Identifies and returns all rows with duplicate values in specified columns.

    Args:
        df: The input DataFrame to check for duplicates.
        subset: A list of column names to identify duplicates.

    Returns:
        A DataFrame containing all rows that are duplicates based on the subset,
        sorted by the subset columns. Returns an empty DataFrame if no duplicates 
        are found.
    """
    mask = df.duplicated(subset=subset, keep=False)
    if not mask.any():
        return pd.DataFrame(columns=df.columns)
    return df[mask].sort_values(by=subset)

def get_unique(df: pd.DataFrame, subset: List[str]) -> pd.DataFrame:
    """Filters out the duplicated rows and retains unique data.

    Args:
        df: The input DataFrame.
        subset: A list of column names used to identify duplicates.
    
    Returns:
        A DataFrame containing only unique rows (keeping the last occurrence),
        sorted by the subset columns.
    """
    mask = df.duplicated(subset=subset, keep='last')
    return df[~mask].sort_values(by=subset)

def parse_datetime_column(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Converts a string column into standardized Pandas datetime objects.

    Args:
        df: The input DataFrame.
        col_name: The name of the column to convert.

    Returns:
        A new DataFrame with the specified column converted to datetime format.
        Unparseable data is coerced into NaT.
    """
    df_clean = df.copy()
    df_clean[col_name] = pd.to_datetime(df_clean[col_name], errors='coerce')
    return df_clean

def trim_whitespace(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Removes leading and trailing whitespace from a specified string column.

    Args:
        df: The input DataFrame.
        col_name: The name of the string column to trim.

    Returns:
        A new DataFrame with the specified column's whitespace trimmed.
    """
    df_clean = df.copy()
    if col_name in df_clean.columns and pd.api.types.is_string_dtype(df_clean[col_name]):
        df_clean[col_name] = df_clean[col_name].str.strip()
    return df_clean

def map_categorical_values(
    df: pd.DataFrame, 
    col_name: str, 
    mapping: Dict[Any, Any]
) -> pd.DataFrame:
    """Replaces categorical values in a specified column based on a mapping dictionary.

    Args:
        df: The input DataFrame.
        col_name: The column containing categories to map.
        mapping: A dictionary where keys are original values and values are the new values.

    Returns:
        A new DataFrame with mapped categorical values.
    """
    df_clean = df.copy()
    df_clean[col_name] = df_clean[col_name].replace(mapping)
    return df_clean

def convert_to_category(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Converts a specified column type to 'category'.

    Args:
        df: The input DataFrame.
        col_name: The column to convert.

    Returns:
        A new DataFrame with the column cast as categorical type.
    """
    df_clean = df.copy()
    df_clean[col_name] = df_clean[col_name].astype('category')
    return df_clean

def remove_specific_value_str(df: pd.DataFrame, col_name: str, value: str) -> pd.DataFrame:
    """Removes rows containing a specific exact string value in a specified column.
    
    Args: 
        df: The input DataFrame.
        col_name: The column to check for the undesired value.
        value: The specific string value to filter out.
    
    Returns:
        A new DataFrame where rows matching the value in the specified column are removed.
    """
    df_clean = df.copy()
    df_clean = df_clean[df_clean[col_name] != value]
    return df_clean

def filter_invalid_dates(
    df: pd.DataFrame, 
    created_at: str, 
    reservation_at: str
) -> pd.DataFrame:
    """Filters out rows where the creation timestamp is strictly after the reservation timestamp.

    Args:
        df: The input DataFrame containing booking records.
        created_at: Column name for the record creation timestamp.
        reservation_at: Column name for the actual dining reservation timestamp.

    Returns:
        A filtered DataFrame where 'created_at' <= 'reservation_at'.
    """
    valid_mask = df[created_at] <= df[reservation_at]
    return df[valid_mask].reset_index(drop=True)

def get_invalid_age_records(
    df: pd.DataFrame, 
    birthdate_col: str, 
    max_age: int = 120
) -> pd.DataFrame:
    """Identifies records with illogical ages (e.g., negative or exceeding maximum).
    
    Args:
        df: The input DataFrame.
        birthdate_col: The column containing birthdates (datetime type required).
        max_age: The maximum logical age allowed. Defaults to 120.
        
    Returns:
        A DataFrame containing only the invalid rows based on age constraints.
    """
    now = pd.Timestamp.now()
    age = now.year - df[birthdate_col].dt.year - (
        (now.month < df[birthdate_col].dt.month) | 
        ((now.month == df[birthdate_col].dt.month) & (now.day < df[birthdate_col].dt.day))
    ).astype(int)
    
    invalid_mask = (age < 0) | (age > max_age)
    invalid_mask = invalid_mask.fillna(False)
    
    if not invalid_mask.any():
        return pd.DataFrame(columns=df.columns).astype(df.dtypes)
    return df[invalid_mask].sort_values(by=birthdate_col)

def get_future_date_records(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Identifies records where the specified datetime column is set in the future.
    
    Args:
        df: The input DataFrame.
        date_col: The datetime column to check against the current time.
        
    Returns:
        A DataFrame containing records with future dates.
    """
    now = pd.Timestamp.now()
    invalid_mask = df[date_col] > now
    invalid_mask = invalid_mask.fillna(False)
    
    if not invalid_mask.any():
        return pd.DataFrame(columns=df.columns).astype(df.dtypes)
    return df[invalid_mask].sort_values(by=date_col)

def get_invalid_price(
    df: pd.DataFrame,
    col_name: str,
    min_price: int = 0
) -> pd.DataFrame:
    """Identifies records where the price is below the expected minimum.
    
    Args:
        df: The input DataFrame.
        col_name: The column containing pricing information.
        min_price: The minimum valid price limit. Defaults to 0.
        
    Returns:
        A DataFrame containing invalid pricing records.
    """
    invalid_mask = (df[col_name] < min_price)
    invalid_mask = invalid_mask.fillna(False)
    
    if not invalid_mask.any():
        return pd.DataFrame(columns=df.columns).astype(df.dtypes)
    return df[invalid_mask].sort_values(by=col_name)