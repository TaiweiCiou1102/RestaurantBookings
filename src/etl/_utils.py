"""Shared helper functions for the ETL pipeline.

Organized into three sections matching the pipeline steps:
    - Cleaning helpers      (used by step1_run_cleaning)
    - Integration helpers   (used by step2_run_integration)
    - Feature engineering   (used by step3_run_features)
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# Cleaning helpers (step1)
# ============================================================

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


# ============================================================
# Integration helpers (step2)
# ============================================================

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


# ============================================================
# Feature engineering helpers (step3)
# ============================================================

def lead_time(
    df: pd.DataFrame,
    booking_time: str,
    reservation_time: str
) -> pd.DataFrame:
    """Calculates the lead time (in hours) between booking and reservation.

    Args:
        df: The input pandas DataFrame.
        booking_time: Column name representing when the booking was made.
        reservation_time: Column name representing when the dining is scheduled.

    Returns:
        A new DataFrame with an added 'lead_time' column.
    """
    logger.debug("Calculating lead time feature.")
    result_df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(result_df[booking_time]):
        result_df[booking_time] = pd.to_datetime(result_df[booking_time], errors='coerce')
    if not pd.api.types.is_datetime64_any_dtype(result_df[reservation_time]):
        result_df[reservation_time] = pd.to_datetime(result_df[reservation_time], errors='coerce')

    result_df['lead_time'] = (result_df[reservation_time] - result_df[booking_time]).dt.total_seconds() / 3600
    return result_df

def calculate_age(
    df: pd.DataFrame,
    birth_date_col: str,
    ref_date_col: str
) -> pd.DataFrame:
    """Calculates the age of the customer based on a reference date.

    Args:
        df: The input pandas DataFrame.
        birth_date_col: Column name representing the customer's birthdate.
        ref_date_col: Column name representing the reference date (e.g., dining date).

    Returns:
        A new DataFrame with an added 'age' column.
    """
    logger.debug("Calculating age feature.")
    result_df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(result_df[birth_date_col]):
        result_df[birth_date_col] = pd.to_datetime(result_df[birth_date_col], errors='coerce')
    if not pd.api.types.is_datetime64_any_dtype(result_df[ref_date_col]):
        result_df[ref_date_col] = pd.to_datetime(result_df[ref_date_col], errors='coerce')

    result_df['age'] = result_df[ref_date_col].dt.year - result_df[birth_date_col].dt.year

    # Explicitly set age to NaN where either date is missing
    missing_mask = result_df[birth_date_col].isna() | result_df[ref_date_col].isna()
    result_df.loc[missing_mask, 'age'] = np.nan

    return result_df

def extract_hour(df: pd.DataFrame, datetime_col: str, output_col: str = "hour") -> pd.DataFrame:
    """Extracts the hour (0-23) from a datetime column.

    Args:
        df: The input pandas DataFrame.
        datetime_col: The datetime column to extract the hour from.
        output_col: Name of the output column.

    Returns:
        A new DataFrame with an added hour column.
    """
    logger.debug("Extracting hour from '%s' into '%s'.", datetime_col, output_col)
    result_df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(result_df[datetime_col]):
        result_df[datetime_col] = pd.to_datetime(result_df[datetime_col], errors='coerce')

    result_df[output_col] = result_df[datetime_col].dt.hour
    return result_df


def extract_half_hour_slot(df: pd.DataFrame, datetime_col: str, output_col: str = "half_hour_slot") -> pd.DataFrame:
    """Extracts the half-hour slot index (0-47) from a datetime column.

    Slot 0 = 00:00-00:29, slot 1 = 00:30-00:59, ..., slot 47 = 23:30-23:59.

    Args:
        df: The input pandas DataFrame.
        datetime_col: The datetime column to extract from.
        output_col: Name of the output column.

    Returns:
        A new DataFrame with an added half-hour slot column.
    """
    logger.debug("Extracting half-hour slot from '%s' into '%s'.", datetime_col, output_col)
    result_df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(result_df[datetime_col]):
        result_df[datetime_col] = pd.to_datetime(result_df[datetime_col], errors='coerce')

    dt = result_df[datetime_col]
    result_df[output_col] = dt.dt.hour * 2 + dt.dt.minute // 30
    return result_df


def extract_seconds_of_day(df: pd.DataFrame, datetime_col: str, output_col: str = "seconds_of_day") -> pd.DataFrame:
    """Extracts seconds since midnight (0-86399) from a datetime column.

    Args:
        df: The input pandas DataFrame.
        datetime_col: The datetime column to extract from.
        output_col: Name of the output column.

    Returns:
        A new DataFrame with an added seconds-of-day column.
    """
    logger.debug("Extracting seconds of day from '%s' into '%s'.", datetime_col, output_col)
    result_df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(result_df[datetime_col]):
        result_df[datetime_col] = pd.to_datetime(result_df[datetime_col], errors='coerce')

    dt = result_df[datetime_col]
    result_df[output_col] = dt.dt.hour * 3600 + dt.dt.minute * 60 + dt.dt.second
    return result_df

def weekday(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    """Extracts the weekday (1=Monday, 7=Sunday) from a datetime column.

    Args:
        df: The input pandas DataFrame.
        datetime_col: The datetime column to extract the weekday from.

    Returns:
        A new DataFrame with an added 'weekday' column.
    """
    logger.debug("Extracting weekday feature.")
    result_df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(result_df[datetime_col]):
        result_df[datetime_col] = pd.to_datetime(result_df[datetime_col], errors='coerce')

    result_df['weekday'] = result_df[datetime_col].dt.dayofweek + 1
    return result_df

def average_price(
    df: pd.DataFrame,
    price1_col: str,
    price2_col: str
) -> pd.DataFrame:
    """Calculates the average price given a lower and upper bound price limit.

    Args:
        df: The input pandas DataFrame.
        price1_col: Column name of the minimum price.
        price2_col: Column name of the maximum price.

    Returns:
        A new DataFrame with an added 'avg_price' column.
    """
    logger.debug("Calculating average price feature.")
    result_df = df.copy()
    p1 = pd.to_numeric(result_df[price1_col], errors='coerce')
    p2 = pd.to_numeric(result_df[price2_col], errors='coerce')
    result_df['avg_price'] = (p1 + p2) / 2
    return result_df

def is_holiday_vicinity(
    df: pd.DataFrame,
    datetime_col: str,
    holiday_list: List[str]
) -> pd.DataFrame:
    """Determines if a date falls within 7 days of any given holiday.

    Args:
        df: The input pandas DataFrame.
        datetime_col: The datetime column to evaluate.
        holiday_list: A list of date strings representing holidays.

    Returns:
        A new DataFrame with an added 'is_holiday_vicinity' binary column (1/0).
    """
    logger.debug("Calculating holiday vicinity features.")
    result_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(result_df[datetime_col]):
        result_df[datetime_col] = pd.to_datetime(result_df[datetime_col], errors='coerce')

    dates = result_df[datetime_col].dt.normalize()
    holiday_series = pd.to_datetime(pd.Series(holiday_list)).dt.normalize()

    def check_vicinity(d: pd.Timestamp) -> bool:
        if pd.isna(d):
            return False
        diffs = (holiday_series - d).dt.days.abs()
        return bool((diffs <= 7).any())

    result_df['is_holiday_vicinity'] = dates.apply(check_vicinity).astype(int)
    return result_df

def days_from_payday(
    df: pd.DataFrame,
    datetime_col: str,
    payday: int = 5
) -> pd.DataFrame:
    """Calculates the closest number of days to a recurring monthly payday.

    Args:
        df: The input pandas DataFrame.
        datetime_col: The datetime column to evaluate.
        payday: The day of the month when salaries are typically paid. Defaults to 5.

    Returns:
        A new DataFrame with an added 'days_from_payday' column.
    """
    logger.debug("Calculating days from payday feature.")
    result_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(result_df[datetime_col]):
        result_df[datetime_col] = pd.to_datetime(result_df[datetime_col], errors='coerce')

    dates = result_df[datetime_col].dt.normalize()

    def calc_payday_diff(d: pd.Timestamp) -> float:
        if pd.isna(d):
            return np.nan

        try:
            curr_payday = pd.Timestamp(year=d.year, month=d.month, day=payday)
        except ValueError:
            curr_payday = pd.Timestamp(year=d.year, month=d.month, day=28)

        next_m = d.month + 1 if d.month < 12 else 1
        next_y = d.year if d.month < 12 else d.year + 1
        try:
            next_payday = pd.Timestamp(year=next_y, month=next_m, day=payday)
        except ValueError:
            next_payday = pd.Timestamp(year=next_y, month=next_m, day=28)

        prev_m = d.month - 1 if d.month > 1 else 12
        prev_y = d.year if d.month > 1 else d.year - 1
        try:
            prev_payday = pd.Timestamp(year=prev_y, month=prev_m, day=payday)
        except ValueError:
            prev_payday = pd.Timestamp(year=prev_y, month=prev_m, day=28)

        c_diff = abs((d - curr_payday).days)
        n_diff = abs((d - next_payday).days)
        p_diff = abs((d - prev_payday).days)

        return float(min(c_diff, n_diff, p_diff))

    result_df['days_from_payday'] = dates.apply(calc_payday_diff)
    return result_df

def restaurant_popular_hour(
    df: pd.DataFrame,
    restaurant_col: str = 'restaurant_id',
    datetime_col: str = 'datetime'
) -> pd.DataFrame:
    """Calculates the most popular booking hour and dinner booking ratio per restaurant.

    Args:
        df: The input pandas DataFrame.
        restaurant_col: Column representing the unique restaurant identifier.
        datetime_col: Column representing the typical dining time.

    Returns:
        A new DataFrame with 'popular_hour' and 'dinner_ratio' columns mapped to each row.
    """
    logger.debug("Calculating restaurant popular hour features.")
    result_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(result_df[datetime_col]):
        result_df[datetime_col] = pd.to_datetime(result_df[datetime_col], errors='coerce')

    hours = result_df[datetime_col].dt.hour

    popular_hour_map = hours.groupby(result_df[restaurant_col]).agg(
        lambda x: x.mode()[0] if not x.mode().empty else None
    )

    is_dinner = hours.between(17, 21)
    dinner_ratio_map = is_dinner.groupby(result_df[restaurant_col]).mean()

    result_df['popular_hour'] = result_df[restaurant_col].map(popular_hour_map)
    result_df['dinner_ratio'] = result_df[restaurant_col].map(dinner_ratio_map)

    global_mode = hours.mode()[0] if not hours.mode().empty else 18
    result_df['popular_hour'] = result_df['popular_hour'].fillna(global_mode).astype(int)
    result_df['dinner_ratio'] = result_df['dinner_ratio'].fillna(0.0)

    return result_df

def area_restaurant_density(
    df: pd.DataFrame,
    area_col: str = 'cityarea',
    restaurant_col: str = 'restaurant_id'
) -> pd.DataFrame:
    """Calculates the number of unique restaurants in a specified geographical area.

    Args:
        df: The input pandas DataFrame.
        area_col: Column representing the region or city area.
        restaurant_col: Column representing unique restaurants.

    Returns:
        A new DataFrame with an added 'restaurant_density' column.
    """
    logger.debug("Calculating area restaurant density feature.")
    result_df = df.copy()

    density_map = result_df.groupby(area_col)[restaurant_col].nunique()
    result_df['restaurant_density'] = result_df[area_col].map(density_map)
    result_df['restaurant_density'] = result_df['restaurant_density'].fillna(0).astype(int)

    return result_df

def weather_condition(
    df: pd.DataFrame,
    weather_df: pd.DataFrame,
    datetime_col: str = 'datetime',
    city_col: str = 'city'
) -> pd.DataFrame:
    """Merges external weather data into the main dataset based on date and city.

    Args:
        df: The main pandas DataFrame containing booking records.
        weather_df: A pandas DataFrame containing external daily weather logs.
        datetime_col: The datetime column used to align the date in `df`.
        city_col: The column used to align the geographic location.

    Returns:
        A new DataFrame supplemented with weather features.

    Raises:
        ValueError: If `weather_df` lacks the required 'date' or geographic column.
    """
    logger.debug("Merging weather condition external data.")
    result_df = df.copy()

    temp_date_col = '_merge_date_'
    result_df[temp_date_col] = pd.to_datetime(result_df[datetime_col], errors='coerce').dt.normalize()

    if 'date' not in weather_df.columns:
        raise ValueError("weather_df must contain a 'date' column.")

    weather_copy = weather_df.copy()
    weather_copy['date'] = pd.to_datetime(weather_copy['date'], errors='coerce').dt.normalize()

    if city_col not in weather_copy.columns:
        raise ValueError(f"weather_df must contain '{city_col}' column to align geographic data.")

    result_df = pd.merge(
        left=result_df,
        right=weather_copy,
        left_on=[temp_date_col, city_col],
        right_on=['date', city_col],
        how='left'
    )

    result_df = result_df.drop(columns=[temp_date_col, 'date'], errors='ignore')
    return result_df

def drop_columns(df: pd.DataFrame, cols_to_drop: List[str]) -> pd.DataFrame:
    """Removes specified columns from the DataFrame.

    Args:
        df: The input pandas DataFrame.
        cols_to_drop: A list of column names to be removed.

    Returns:
        A new DataFrame with the specified columns dropped.
    """
    logger.debug("Dropping columns: %s", cols_to_drop)
    # 加上 errors='ignore' 確保即使輸入了不存在的欄位也不會報錯中斷
    return df.drop(columns=cols_to_drop, errors='ignore')

def rename_columns(df: pd.DataFrame, columns_mapping: dict) -> pd.DataFrame:
    """Renames specified columns in the DataFrame based on a mapping dictionary.

    Args:
        df: The input pandas DataFrame.
        columns_mapping: A dictionary where keys are original column names and values are new names.

    Returns:
        A new DataFrame with the renamed columns.
    """
    logger.debug("Renaming columns with mapping: %s", columns_mapping)
    return df.rename(columns=columns_mapping)
