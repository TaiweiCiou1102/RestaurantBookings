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

def clean_order(input_path: Path, output_path: Path) -> None:
    """Cleans the order dataset and saves it as an interim file.

    Args:
        input_path: Path to the raw train.csv file.
        output_path: Path to save the cleaned train_silver.csv file.
    """
    logger.info("Loading raw order data from %s", input_path)
    order_df = pd.read_csv(input_path, sep="\t", encoding="utf-16", on_bad_lines="warn")

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
    order_final = (
        order_df.pipe(fill_missing_values, col_name="purpose", fill_value="Unknown")
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

    logger.info("Saving cleaned order data to %s", output_path)
    order_final.to_csv(output_path, index=False, sep=",", encoding="utf-8")


def clean_restaurant(input_path: Path, output_path: Path) -> None:
    """Cleans the restaurant dataset and saves it as an interim file.

    Args:
        input_path: Path to the raw restaurant_revised.csv file.
        output_path: Path to save the cleaned restaurant_silver.csv file.
    """
    logger.info("Loading raw restaurant data from %s", input_path)
    restaurant_df = pd.read_csv(
        input_path, sep="\t", encoding="utf-16", on_bad_lines="warn"
    )

    city_mapping = {"桃園縣": "桃園市", "%": "Unknown"}
    cityarea_mapping = {"請選擇": "Unknown"}

    logger.info("Applying cleaning pipeline on restaurant data.")
    restaurant_final = (
        restaurant_df.pipe(fill_missing_values, col_name="cityarea", fill_value="Unknown")
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

    logger.info("Saving cleaned restaurant data to %s", output_path)
    restaurant_final.to_csv(output_path, index=False, sep=",", encoding="utf-8")


def clean_member(input_path: Path, output_path: Path) -> None:
    """Cleans the member dataset and saves it as an interim file.

    Args:
        input_path: Path to the raw member.csv file.
        output_path: Path to save the cleaned member_silver.csv file.
    """
    logger.info("Loading raw member data from %s", input_path)
    member_df = pd.read_csv(
        input_path, sep="\t", encoding="utf-16", on_bad_lines="warn"
    )

    member_city_mapping = {"桃園縣": "桃園市", "0": "Unknown", 0: "Unknown"}

    logger.info("Applying cleaning pipeline on member data.")
    member_final = (
        member_df.pipe(fill_missing_values, col_name="city", fill_value="Unknown")
        .pipe(fill_missing_values, col_name="gender", fill_value="Unknown")
        .pipe(parse_datetime_column, col_name="birthdate")
        .pipe(parse_datetime_column, col_name="cdate")
        .pipe(
            remove_specific_value_str,
            col_name="city",
            value="??蝮?>??蝮?/option><option value",
        )
        .pipe(map_categorical_values, col_name="city", mapping=member_city_mapping)
        .pipe(convert_to_category, col_name="gender")
        .pipe(convert_to_category, col_name="city")
    )

    logger.info("Saving cleaned member data to %s", output_path)
    member_final.to_csv(output_path, index=False, sep=",", encoding="utf-8")


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
        clean_order(
            input_path=raw_dir / "train.csv",
            output_path=interim_dir / "train_silver.csv",
        )
        clean_restaurant(
            input_path=raw_dir / "restaurant_revised.csv",
            output_path=interim_dir / "restaurant_silver.csv",
        )
        clean_member(
            input_path=raw_dir / "member.csv",
            output_path=interim_dir / "member_silver.csv",
        )
        logger.info("Data cleaning completed successfully.")
    except Exception as e:
        logger.error("An error occurred during data cleaning: %s", e)
        raise

if __name__ == "__main__":
    main()
