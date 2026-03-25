"""Main ETL pipeline script for integrating and merging datasets."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.etl._integration_utils import safe_merge

# Configure logging standard
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def integrate_data(
    train_path: Path, 
    restaurant_path: Path, 
    member_path: Path, 
    output_path: Path
) -> None:
    """Loads interim data, joins dataframes, and saves the result.

    Args:
        train_path: Path to the cleaned order dataset.
        restaurant_path: Path to the cleaned restaurant dataset.
        member_path: Path to the cleaned member dataset.
        output_path: Path to save the fully joined dataset.
    """
    logger.info("Loading cleaned datasets for integration.")
    
    try:
        df_orders = pd.read_csv(train_path)
        df_restaurants = pd.read_csv(restaurant_path)
        df_members = pd.read_csv(member_path)
    except FileNotFoundError as e:
        logger.error("Missing interim file: %s", e)
        raise
        
    logger.info("Merging orders with restaurants on 'restaurant_id'")
    # Example logic assuming 'restaurant_id' exists in both
    joined_df = safe_merge(
        left=df_orders,
        right=df_restaurants,
        on_cols=["restaurant_id"],
        how="left"
    )
    
    logger.info("Merging combined data with members on 'member_id'")
    # Example logic assuming 'member_id' exists in both
    joined_df = safe_merge(
        left=joined_df,
        right=df_members,
        on_cols=["member_id"],
        how="left"
    )
    
    logger.info("Saving integrated dataset to %s", output_path)
    joined_df.to_csv(output_path, index=False)


def main() -> None:
    """Main execution entrypoint for data integration pipeline."""
    parser = argparse.ArgumentParser(description="ETL Process: Data Integration")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Root directory for data storage.",
    )
    args = parser.parse_args()

    interim_dir = args.data_dir / "interim"
    
    try:
        integrate_data(
            train_path=interim_dir / "train_silver.csv",
            restaurant_path=interim_dir / "restaurant_silver.csv",
            member_path=interim_dir / "member_silver.csv",
            output_path=interim_dir / "joined_data.csv"
        )
        logger.info("Data integration completed successfully.")
    except Exception as e:
        logger.error("An error occurred during data integration: %s", e)
        raise

if __name__ == "__main__":
    main()
