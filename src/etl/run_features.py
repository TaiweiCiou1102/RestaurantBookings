"""Main ETL pipeline script for executing feature engineering transformations."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.etl._features_utils import (
    area_restaurant_density,
    average_price,
    calculate_age,
    days_from_payday,
    drop_columns,
    extract_half_hour_slot,
    extract_hour,
    extract_seconds_of_day,
    is_holiday_vicinity,
    lead_time,
    rename_columns,
    restaurant_popular_hour,
    weekday,
)

# Configure logging standard
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies a continuous pipeline of statistical feature engineering functions.

    Args:
        df: The integrated interim pandas DataFrame.

    Returns:
        The fully processed DataFrame with new engineered features.
    """
    logger.info("Applying feature engineering pipeline.")
    
    # Example holiday dates for Taiwan 2014
    holidays_2014 = [
        "2014-01-01", "2014-01-30", "2014-01-31", "2014-02-03", "2014-02-28", 
        "2014-04-04", "2014-06-02", "2014-09-08", "2014-10-10",
    ]
    
    df_engineered = (
        df.pipe(lead_time, booking_time="cdate", reservation_time="datetime")
        .pipe(calculate_age, birth_date_col="birthdate", ref_date_col="datetime")
        .pipe(extract_hour, datetime_col="cdate", output_col="booking_hour")
        .pipe(extract_hour, datetime_col="datetime", output_col="reservation_hour")
        .pipe(extract_seconds_of_day, datetime_col="datetime", output_col="reservation_seconds")
        .pipe(extract_half_hour_slot, datetime_col="datetime", output_col="reservation_half_hour")
        .pipe(weekday, datetime_col="datetime")
        .pipe(average_price, price1_col="price1", price2_col="price2")
        .pipe(is_holiday_vicinity, datetime_col="datetime", holiday_list=holidays_2014)
        .pipe(days_from_payday, datetime_col="datetime", payday=5)
        .pipe(restaurant_popular_hour, restaurant_col="restaurant_id", datetime_col="datetime")
        .pipe(area_restaurant_density, area_col="cityarea", restaurant_col="restaurant_id")
        .pipe(rename_columns, columns_mapping={
            'cdate': 'booking_time',
            'datetime': 'reservation_time',
            'people': 'party_size',
            'purpose': 'dining_purpose',
            'gender': 'booked_by_gender',
            'status': 'booking_status',
            'is_required_prepay_satisfied': 'prepay_satisfied',
            'is_hotel': 'hotel_restaurant',
            'city': 'city_code',
            'cityarea': 'city_area_code',
            'good_for_family': 'family_friendly',
            'accept_credit_card': 'accepts_credit_card',
            'parking': 'has_parking',
            'outdoor_seating': 'outdoor_seating',
            'wifi': 'has_wifi',
            'wheelchair_accessible': 'wheelchair_accessible',
            'price1': 'min_price',
            'price2': 'max_price',
            'is_vip': 'is_vip_member',
            'member_gender': 'account_gender',
            'birthdate': 'member_birthdate',
            'member_city': 'member_city_code'
        })
        .pipe(drop_columns, cols_to_drop=['booking_id', 'member_id', 'restaurant_id','booking_status','return90','min_price','max_price'])
    )
    
    logger.info("Feature engineering pipeline applied successfully.")
    return df_engineered


def main() -> None:
    """Main execution entrypoint for feature engineering pipeline."""
    parser = argparse.ArgumentParser(description="ETL Process: Feature Engineering")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Root directory for data storage.",
    )
    args = parser.parse_args()

    interim_dir = args.data_dir / "interim"
    processed_dir = args.data_dir / "processed"

    processed_dir.mkdir(parents=True, exist_ok=True)

    input_path = interim_dir / "joined_data.csv"
    output_path = processed_dir / "features_ready.csv"

    try:
        logger.info("Loading joined dataset from %s", input_path)
        df_input = pd.read_csv(input_path, on_bad_lines="warn")
        
        df_feats = transform_features(df_input)
        
        logger.info("Saving engineered dataset to %s", output_path)
        df_feats.to_csv(output_path, index=False)
        logger.info("Feature engineering completed successfully.")
        
    except FileNotFoundError as e:
        logger.error("Required integration dataset not found: %s", e)
        raise
    except Exception as e:
        logger.error("An error occurred during feature engineering: %s", e)
        raise


if __name__ == "__main__":
    main()
