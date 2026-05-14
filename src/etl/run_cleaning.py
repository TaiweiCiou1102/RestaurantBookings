"""Main ETL pipeline script for extracting and cleaning raw data."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.etl._cleaning_utils import (
    convert_to_category,
    fill_missing_values,
    map_categorical_values,
    parse_datetime_column,
    remove_specific_value_str,
    trim_whitespace,
)

# Configure logging standard
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

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

    logger.info("Applying cleaning pipeline on order data.")
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

    logger.info("Applying cleaning pipeline on restaurant data.")
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

    logger.info("Applying cleaning pipeline on member data.")
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
    parser = argparse.ArgumentParser(description="ETL Process: Data Cleaning")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Root directory for data storage.",
    )
    args = parser.parse_args()

    raw_dir = args.data_dir / "raw"
    interim_dir = args.data_dir / "interim"

    interim_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Loading raw order data from %s", raw_dir / "train.csv")
        order_df = pd.read_csv(raw_dir / "train.csv", sep="\t", encoding="utf-16", on_bad_lines="warn")
        order_final = clean_order(order_df)
        logger.info("Saving cleaned order data to %s", interim_dir / "train_silver.csv")
        order_final.to_csv(interim_dir / "train_silver.csv", index=False, sep=",", encoding="utf-8")

        logger.info("Loading raw restaurant data from %s", raw_dir / "restaurant_revised.csv")
        restaurant_df = pd.read_csv(raw_dir / "restaurant_revised.csv", sep="\t", encoding="utf-16", on_bad_lines="warn")
        restaurant_final = clean_restaurant(restaurant_df)
        logger.info("Saving cleaned restaurant data to %s", interim_dir / "restaurant_silver.csv")
        restaurant_final.to_csv(interim_dir / "restaurant_silver.csv", index=False, sep=",", encoding="utf-8")

        logger.info("Loading raw member data from %s", raw_dir / "member.csv")
        member_df = pd.read_csv(raw_dir / "member.csv", sep="\t", encoding="utf-16", on_bad_lines="warn")
        member_final = clean_member(member_df)
        logger.info("Saving cleaned member data to %s", interim_dir / "member_silver.csv")
        member_final.to_csv(interim_dir / "member_silver.csv", index=False, sep=",", encoding="utf-8")

        logger.info("Data cleaning completed successfully.")
    except Exception as e:
        logger.error("An error occurred during data cleaning: %s", e)
        raise

if __name__ == "__main__":
    main()
