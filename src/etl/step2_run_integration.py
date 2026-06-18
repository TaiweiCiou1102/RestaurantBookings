"""Main ETL pipeline script for integrating and merging datasets."""

from pathlib import Path

import pandas as pd

from src.etl._config import load_config
from src.etl._utils import safe_merge


def integrate_data(
    train_path: Path,
    restaurant_path: Path,
    member_path: Path,
    output_path: Path,
) -> None:
    """Loads interim data, joins dataframes, and saves the result.

    Args:
        train_path: Path to the cleaned order dataset.
        restaurant_path: Path to the cleaned restaurant dataset.
        member_path: Path to the cleaned member dataset.
        output_path: Path to save the fully joined dataset.
    """
    df_orders = pd.read_csv(train_path)
    df_restaurants = pd.read_csv(restaurant_path)
    df_members = pd.read_csv(member_path)

    joined_df = safe_merge(df_orders, df_restaurants, on_cols=["restaurant_id"], how="left")
    joined_df = safe_merge(joined_df, df_members, on_cols=["member_id"], how="left")

    joined_df.to_csv(output_path, index=False)


def main() -> None:
    """Main execution entrypoint for data integration pipeline."""
    interim_dir = load_config()["dirs"]["interim"]

    integrate_data(
        train_path=interim_dir / "train_silver.csv",
        restaurant_path=interim_dir / "restaurant_silver.csv",
        member_path=interim_dir / "member_silver.csv",
        output_path=interim_dir / "joined_data.csv",
    )
    print(f"Integrated dataset → {interim_dir / 'joined_data.csv'}")


if __name__ == "__main__":
    main()
