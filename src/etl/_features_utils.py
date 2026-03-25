"""Utility functions for feature engineering."""

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def lead_time(
    df: pd.DataFrame, 
    booking_time: str, 
    reservation_time: str
) -> pd.DataFrame:
    """Calculates the lead time (in days) between booking and reservation.

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
        
    result_df['lead_time'] = (result_df[reservation_time] - result_df[booking_time]).dt.days
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
    return result_df

def booking_hour(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    """Extracts the hour from a datetime column and applies one-hot encoding.

    Args:
        df: The input pandas DataFrame.
        datetime_col: The datetime column to extract the hour from.

    Returns:
        A new DataFrame with one-hot encoded 'hour_X' columns.
    """
    logger.debug("Extracting booking hour features.")
    result_df = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(result_df[datetime_col]):
        result_df[datetime_col] = pd.to_datetime(result_df[datetime_col], errors='coerce')
        
    hour_series = result_df[datetime_col].dt.hour
    for h in range(24):
        result_df[f'hour_{h}'] = (hour_series == h).astype(int)
    return result_df

def weekday(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    """Extracts the weekday (0=Monday, 6=Sunday) from a datetime column.

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
        
    result_df['weekday'] = result_df[datetime_col].dt.dayofweek
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
