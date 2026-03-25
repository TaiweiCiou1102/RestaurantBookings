import pandas as pd
import numpy as np
from typing import List, Optional


##  資料品質層
## step1 Missing Values

def show_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """顯示各欄位缺失值數量及其百分比"""
    missing_count = df.isnull().sum()
    missing_percent = 100 * df.isnull().sum() / len(df)
    
    # 將結果合併成一個 DataFrame
    missing_table = pd.concat([missing_count, missing_percent], axis=1)
    
    # 重新命名欄位
    missing_table.columns = ['Amount of Missing Values', 'Percentage (%)']
    
    # 只顯示有缺失值的欄位，並由大到小排序
    missing_table = missing_table[missing_table['Amount of Missing Values'] > 0].sort_values('Amount of Missing Values', ascending=False)
    
    return missing_table

def fill_missing_values(df: pd.DataFrame, col_name: str, fill_value="Unknown") -> pd.DataFrame:
    """處理特定欄位的缺失值，將空值補上自訂的 fill_value"""
    df_clean = df.copy()
    df_clean[col_name] = df_clean[col_name].fillna(fill_value)
    return df_clean

## step2 duplicate
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
    # Identify all occurrences of duplicates (keep=False marks all as True).
    mask = df.duplicated(subset=subset, keep=False)
    
    if not mask.any():
        return pd.DataFrame(columns=df.columns)

    return df[mask].sort_values(by=subset)

## step3 Time Format

def parse_datetime_column(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """將字串格式的時間欄位轉換為標準的 Pandas datetime 物件"""
    df_clean = df.copy()
    # errors='coerce' 可以將無法解析的垃圾字串轉為 NaT (Not a Time) 避免報錯
    df_clean[col_name] = pd.to_datetime(df_clean[col_name], errors='coerce')
    return df_clean

## step4categorical data

def trim_whitespace(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """去除字串欄位前後的空白字元"""
    df_clean = df.copy()

    if col_name in df_clean.columns and pd.api.types.is_string_dtype(df_clean[col_name]):
        df_clean[col_name] = df_clean[col_name].str.strip()
        
    return df_clean

def map_categorical_values(df: pd.DataFrame, col_name: str, mapping: dict) -> pd.DataFrame:
    """將類別欄位的值重新對應"""
    df_clean = df.copy()
    df_clean[col_name] = df_clean[col_name].replace(mapping)
    return df_clean

def convert_to_category(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """將類別欄位轉換為 category 型態"""
    df_clean = df.copy()
    df_clean[col_name] = df_clean[col_name].astype('category')
    return df_clean

## remove some value

def remove_specific_value_str(df: pd.DataFrame, col_name: str, value: str):
    """remove the specific value from the specific column
    
    Args: 
        df: The input DataFrame containg booking records.
        col_name: The column name to remove the specific value.
        value: The specific value to remove.
    
    Returns:
        A filtered DataFrame where the specific value is removed from the specific column.
    """
    df_clean = df.copy()
    df_clean = df_clean[df_clean[col_name] != value]
    return df_clean

## 商業邏輯層

def get_invalid_date_records(
    df: pd.DataFrame, 
    created_at: str, 
    reservation_at: str
) -> pd.DataFrame:
    """Identifies records where creation time is logically inconsistent with reservation time.

    Inconsistency is defined as:
    1. created_at > reservation_at
    2. Either created_at or reservation_at is NaT (Not a Time).

    Args:
        df: The input DataFrame containing booking records.
        created_at: Column name for the record creation timestamp.
        reservation_at: Column name for the actual reservation timestamp.

    Returns:
        A DataFrame containing only the invalid rows, sorted by created_at.
        Returns an empty DataFrame with the same schema if no issues are found.
    """
    # Define the validity condition (logic we WANT to keep)
    valid_mask = (df[created_at] <= df[reservation_at])
    
    # The invalid records are those that do NOT meet the validity condition
    # This automatically captures (A > B) AND any rows containing NaT
    invalid_mask = ~valid_mask
    
    if not invalid_mask.any():
        return pd.DataFrame(columns=df.columns).astype(df.dtypes)

    return df[invalid_mask].sort_values(by=created_at)

def filter_invalid_dates(
    df: pd.DataFrame, 
    created_at: str, 
    reservation_at: str
) -> pd.DataFrame:
    """Filters rows where the creation timestamp is strictly after the reservation.

    Args:
        df: The input DataFrame containing booking records.
        created_at: Column name for the record creation timestamp.
        reservation_at: Column name for the actual reservation timestamp.

    Returns:
        A filtered DataFrame where created_at <= reservation_at. Rows with NaT 
        in either column are excluded by default.
    """
    # Create a boolean mask where logic holds: creation must precede reservation.
    # Note: Comparisons with NaT result in False, effectively filtering them out.
    valid_mask = df[created_at] <= df[reservation_at]
    
    return df[valid_mask].reset_index(drop=True)

def get_invalid_age_records(df: pd.DataFrame, birthdate_col: str, max_age: int = 120) -> pd.DataFrame:
    """找出年齡不符邏輯的資料（如：尚未出生或大於 max_age 歲）。
    
    Args:
        df: 輸入的 DataFrame。
        birthdate_col: 包含生日時間的欄位名稱 (必須轉換為 datetime 型態)。
        max_age: 允許的合理最大年齡，預設為 120。
        
    Returns:
        包含異常資料的 DataFrame。
    """
    now = pd.Timestamp.now()
    
    # 計算年齡：用當下年份減去出生年份，並視月份與日期進行修正
    age = now.year - df[birthdate_col].dt.year - (
        (now.month < df[birthdate_col].dt.month) | 
        ((now.month == df[birthdate_col].dt.month) & (now.day < df[birthdate_col].dt.day))
    ).astype(int)
    
    # 定義異常條件：年齡 < 0 (未來出生) 或 > 120 歲
    invalid_mask = (age < 0) | (age > max_age)
    
    # 忽略 NaT 的比較結果 (會產生 False)
    invalid_mask = invalid_mask.fillna(False)
    
    if not invalid_mask.any():
        return pd.DataFrame(columns=df.columns).astype(df.dtypes)
        
    return df[invalid_mask].sort_values(by=birthdate_col)

def get_future_date_records(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """找出特定欄位具有未來時間的異常資料。
    
    Args:
        df: 輸入的 DataFrame。
        date_col: 包含時間的欄位名稱 (必須轉換為 datetime 型態)。
        
    Returns:
        包含未來時間異常資料的 DataFrame。
    """
    now = pd.Timestamp.now()
    
    # 定義異常條件：時間大於現在 (也就是未來發生)
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
    """找出特定欄位價格不符合邏輯的異常資料。
    
    Args:
        df: 輸入的 DataFrame。
        col_name: 包含價格的欄位名稱。
        min_price: 允許的合理最小值，預設為 0。
        
    Returns:
        包含異常價格資料的 DataFrame。
    """
    invalid_mask = (df[col_name] < min_price)
    invalid_mask = invalid_mask.fillna(False)
    
    if not invalid_mask.any():
        return pd.DataFrame(columns=df.columns).astype(df.dtypes)
        
    return df[invalid_mask].sort_values(by=col_name)