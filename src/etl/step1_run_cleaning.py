"""Main ETL pipeline script for extracting and cleaning raw data."""

import pandas as pd

from src.etl._config import load_config
from src.etl._utils import (
    convert_to_category,
    fill_missing_values,
    map_categorical_values,
    parse_datetime_column,
    remove_specific_value_str,
    trim_whitespace,
)


def clean_order(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the order dataset.

    Args:
        df: Raw order DataFrame.

    Returns:
        Cleaned order DataFrame.
    """
    purpose_mapping = {
        "請選擇": "Unknown",
        "Please,Select": "Unknown",
        "Friends,Reunion": "朋友聚餐",
        "Family,Gathering": "家人用餐",
        "Birthday": "生日慶祝",
        "Friends": "朋友聚餐",
        "家人用餐,盡量靠W": "朋友聚餐",
        "Business,meeting": "商務聚餐",
        "家人&#65533;": "家人用餐",
        "家人用餐-生日": "生日慶祝",
        "慶生": "生日慶祝",
        "Birthday,Celebration": "生日慶祝",
        "甜蜜紀念日": "紀念日慶祝",
    }

    return (
        df.pipe(fill_missing_values, col_name="purpose", fill_value="Unknown")
        .pipe(parse_datetime_column, col_name="cdate")
        .pipe(parse_datetime_column, col_name="datetime")
        .pipe(trim_whitespace, col_name="purpose")
        .pipe(map_categorical_values, col_name="purpose", mapping=purpose_mapping)
        .pipe(
            remove_specific_value_str,
            col_name="purpose",
            value="&#65533;&#40115;&#65533;&#65533;&#26813;&#65533;",
        )
        .pipe(convert_to_category, col_name="purpose")
        .pipe(convert_to_category, col_name="gender")
        .pipe(convert_to_category, col_name="status")
    )


def clean_restaurant(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the restaurant dataset.

    Args:
        df: Raw restaurant DataFrame.

    Returns:
        Cleaned restaurant DataFrame.
    """
    city_mapping = {"桃園縣": "桃園市", "%": "Unknown"}
    cityarea_mapping = {"請選擇": "Unknown"}

    return (
        df.pipe(fill_missing_values, col_name="cityarea", fill_value="Unknown")
        .pipe(parse_datetime_column, col_name="cdate")
        .pipe(map_categorical_values, col_name="city", mapping=city_mapping)
        .pipe(map_categorical_values, col_name="cityarea", mapping=cityarea_mapping)
        .pipe(convert_to_category, col_name="country")
        .pipe(convert_to_category, col_name="currency")
        .pipe(convert_to_category, col_name="city")
        .pipe(convert_to_category, col_name="cityarea")
        .pipe(convert_to_category, col_name="timezone")
        .pipe(convert_to_category, col_name="locale")
    )


def clean_member(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the member dataset.

    Args:
        df: Raw member DataFrame.

    Returns:
        Cleaned member DataFrame.
    """
    member_city_mapping = {"桃園縣": "桃園市", "0": "Unknown", 0: "Unknown"}

    return (
        df.pipe(fill_missing_values, col_name="city", fill_value="Unknown")
        .pipe(fill_missing_values, col_name="gender", fill_value="Unknown")
        .pipe(parse_datetime_column, col_name="birthdate")
        .pipe(parse_datetime_column, col_name="cdate")
        .pipe(
            remove_specific_value_str,
            col_name="city",
            value="??蝮?>??蝮?/option><option value",
        )
        .pipe(map_categorical_values, col_name="city", mapping=member_city_mapping)
        .pipe(convert_to_category, col_name="gender")
        .pipe(convert_to_category, col_name="city")
    )


def main() -> None:
    """Main execution entrypoint for data cleaning pipeline."""
    dirs = load_config()["dirs"]
    raw_dir = dirs["raw"]
    interim_dir = dirs["interim"]

    interim_dir.mkdir(parents=True, exist_ok=True)

    order_df = pd.read_csv(raw_dir / "train.csv", sep="\t", encoding="utf-16", on_bad_lines="warn")
    clean_order(order_df).to_csv(interim_dir / "train_silver.csv", index=False, sep=",", encoding="utf-8")
    print(f"Cleaned order data → {interim_dir / 'train_silver.csv'}")

    restaurant_df = pd.read_csv(raw_dir / "restaurant_revised.csv", sep="\t", encoding="utf-16", on_bad_lines="warn")
    clean_restaurant(restaurant_df).to_csv(interim_dir / "restaurant_silver.csv", index=False, sep=",", encoding="utf-8")
    print(f"Cleaned restaurant data → {interim_dir / 'restaurant_silver.csv'}")

    member_df = pd.read_csv(raw_dir / "member.csv", sep="\t", encoding="utf-16", on_bad_lines="warn")
    clean_member(member_df).to_csv(interim_dir / "member_silver.csv", index=False, sep=",", encoding="utf-8")
    print(f"Cleaned member data → {interim_dir / 'member_silver.csv'}")

if __name__ == "__main__":
    main()
