"""Inference script for XGBoost classifier (JSON I/O).

Two entry points:
  1. run_inference_from_raw() — full pipeline from raw test.csv to predictions
  2. run_inference()          — inference on an already feature-engineered DataFrame

CLI usage:
    # From raw CSV (full ETL + preprocessing + predict)
    uv run python -m src.exploration.inference \\
        --raw data/raw/test.csv \\
        --granularity hour

    # From pre-processed JSON (preprocessing + predict)
    uv run python -m src.exploration.inference \\
        --input request.json \\
        --granularity hour \\
        --output predictions.json

Input JSON format (for --input):
    {
      "input_data": {
        "columns": ["party_size", ...],
        "index":   [0, 1],
        "data":    [[...], [...]]
      }
    }

Output JSON format:
    {
      "predictions": {
        "columns": ["prediction"],
        "index":   [0, 1],
        "data":    [[12], [18]]
      }
    }
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from src.etl._utils import safe_merge
from src.etl.step1_run_cleaning import clean_order
from src.etl.step3_run_features import transform_features
from src.common.model_utils import find_nested_runs, load_config, resolve_granularity
from src.exploration.preprocessing.common import NON_FEATURE_COLS, fill_missing
from src.exploration.preprocessing.tree import CATEGORICAL_COLS, apply_saved_encoders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/xgboost.yaml"
DEFAULT_RESTAURANT_CSV = "data/interim/restaurant_silver.csv"
DEFAULT_MEMBER_CSV = "data/interim/member_silver.csv"

# Column order matching features_ready.csv (minus NON_FEATURE_COLS).
# Used to map numeric indices → column names when JSON columns are integers.
FEATURE_COLUMNS = [
    "party_size", "dining_purpose", "booked_by_gender", "prepay_satisfied",
    "hotel_restaurant", "city_code", "city_area_code", "family_friendly",
    "accepts_credit_card", "has_parking", "outdoor_seating", "has_wifi",
    "wheelchair_accessible", "lat", "lng", "is_vip_member", "account_gender",
    "member_city_code", "lead_time", "age", "booking_hour",
    "reservation_hour", "reservation_seconds", "reservation_half_hour",
    "weekday", "avg_price", "is_holiday_vicinity", "days_from_payday",
    "popular_hour", "dinner_ratio", "restaurant_density",
]


# ------------------------------------------------------------------
#  ETL pipeline (raw CSV → feature-engineered DataFrame)
# ------------------------------------------------------------------

def etl_from_raw(
    test_csv_path: str | Path,
    restaurant_csv_path: str | Path = DEFAULT_RESTAURANT_CSV,
    member_csv_path: str | Path = DEFAULT_MEMBER_CSV,
) -> pd.DataFrame:
    """Full ETL: load raw test.csv, clean, integrate, and engineer features.

    Replicates the same pipeline applied to training data:
      clean_order → merge restaurant + member → transform_features

    Args:
        test_csv_path: Path to raw test CSV (tab-separated, UTF-8).
        restaurant_csv_path: Path to cleaned restaurant_silver.csv.
        member_csv_path: Path to cleaned member_silver.csv.

    Returns:
        Feature-engineered DataFrame equivalent to features_ready.csv.
    """
    logger.info("Loading raw test data from %s", test_csv_path)
    df_raw = pd.read_csv(test_csv_path, sep="\t", encoding="utf-8", on_bad_lines="warn")
    df_clean = clean_order(df_raw)

    df_restaurant = pd.read_csv(restaurant_csv_path)
    df_member = pd.read_csv(member_csv_path)

    # Align join keys to match training integration
    df_restaurant = df_restaurant.rename(columns={"id": "restaurant_id"})
    df_member = df_member.rename(columns={
        "id": "member_id",
        "gender": "member_gender",
        "city": "member_city",
    })

    df_joined = safe_merge(df_clean, df_restaurant, on_cols=["restaurant_id"], how="left")
    df_joined = safe_merge(df_joined, df_member, on_cols=["member_id"], how="left")
    logger.info("Joined shape: %s", df_joined.shape)

    df_features = transform_features(df_joined)
    logger.info("Feature shape: %s", df_features.shape)
    return df_features


# ------------------------------------------------------------------
#  Preprocessing helpers
# ------------------------------------------------------------------

def _prepare_X(
    df: pd.DataFrame,
    target: str,
    leakage: List[str],
) -> pd.DataFrame:
    """Drop all columns that must not be seen by the model.

    Removes:
    - Datetime columns (XGBoost cannot consume them)
    - Object columns not in CATEGORICAL_COLS (e.g. name, tel, opening_hours)
    - NON_FEATURE_COLS (booking_time, reservation_time, member_birthdate)
    - Leakage and target columns
    - Merge-suffix leftovers (cdate_x, cdate_y) not caught by rename

    The remaining object columns in CATEGORICAL_COLS are handled by
    apply_saved_encoders (OrdinalEncoder).
    """
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    extra_obj = [c for c in df.columns if df[c].dtype == object and c not in CATEGORICAL_COLS]

    drop = list(set(
        dt_cols
        + extra_obj
        + NON_FEATURE_COLS
        + leakage
        + [target, "cdate_x", "cdate_y"]
    ))
    drop = [c for c in drop if c in df.columns]

    logger.debug("Dropping %d columns before model input: %s", len(drop), drop)
    return df.drop(columns=drop)


# ------------------------------------------------------------------
#  JSON I/O
# ------------------------------------------------------------------

def parse_json_input(path: str) -> pd.DataFrame:
    """Parse input JSON into a DataFrame.

    Accepts ``columns`` as either string names or integer indices.
    Integer indices are mapped to FEATURE_COLUMNS.
    """
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    body = payload["input_data"]
    columns = body["columns"]
    index = body.get("index")
    data = body["data"]

    if columns and isinstance(columns[0], int):
        columns = [FEATURE_COLUMNS[i] for i in columns]

    return pd.DataFrame(data=data, columns=columns, index=index)


def build_json_output(predictions: pd.Series) -> dict:
    """Build output JSON from a prediction Series."""
    return {
        "predictions": {
            "columns": ["prediction"],
            "index": predictions.index.tolist(),
            "data": [[v] for v in predictions.tolist()],
        }
    }


# ------------------------------------------------------------------
#  Routing: split rows into subsets by reservation_hour
# ------------------------------------------------------------------

def route_to_subsets(
    df: pd.DataFrame,
    subsets: Dict[str, dict],
) -> Dict[str, pd.DataFrame]:
    """Split *df* into subsets based on reservation_hour ranges.

    Args:
        df: Full input DataFrame (must contain ``reservation_hour``).
        subsets: Subset definitions from config, e.g.
            ``{"午餐 Lunch (11-14h)": {"hour_min": 11, "hour_max": 14}, ...}``

    Returns:
        Dict mapping subset name → filtered DataFrame (preserving original index).
        Rows outside all defined ranges are collected under ``"_unmatched"``.
    """
    matched_indices: list[int] = []
    result: Dict[str, pd.DataFrame] = {}

    for name, rng in subsets.items():
        mask = df["reservation_hour"].between(rng["hour_min"], rng["hour_max"])
        subset_df = df.loc[mask]
        result[name] = subset_df
        matched_indices.extend(subset_df.index.tolist())
        logger.info("Route '%s': %d rows", name, len(subset_df))

    unmatched = df.loc[~df.index.isin(matched_indices)]
    if len(unmatched) > 0:
        logger.warning("%d rows did not match any subset — skipped", len(unmatched))
        result["_unmatched"] = unmatched

    return result


# ------------------------------------------------------------------
#  Single-subset inference
# ------------------------------------------------------------------

def load_subset_artifacts(
    run_id: str,
) -> Tuple[object, Dict[str, OrdinalEncoder], LabelEncoder]:
    """Load model, feature encoders, and target encoder from an MLflow run.

    Returns:
        (xgb_model, feature_encoders, target_encoder)
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    artifact_dir = client.download_artifacts(run_id, "")
    artifact_path = Path(artifact_dir)

    model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")

    encoders: Dict[str, OrdinalEncoder] = joblib.load(artifact_path / "encoders.joblib")
    target_enc: LabelEncoder = joblib.load(artifact_path / "target_encoder.joblib")

    return model, encoders, target_enc


def predict_subset(
    df: pd.DataFrame,
    run_id: str,
    target: str,
    leakage: List[str],
) -> pd.Series:
    """Run inference on a single subset DataFrame.

    Args:
        df: Feature DataFrame for one subset (already routed).
        run_id: MLflow run ID of the nested (subset) run.
        target: Target column name (dropped if present).
        leakage: Leakage columns to drop.

    Returns:
        pd.Series of predicted labels (original scale), indexed like *df*.
    """
    model, encoders, target_enc = load_subset_artifacts(run_id)

    X = _prepare_X(df, target, leakage)
    X_proc = apply_saved_encoders(X, encoders)

    y_enc = model.predict(X_proc)
    y_labels = target_enc.inverse_transform(y_enc)

    return pd.Series(y_labels, index=df.index, name="prediction")


# ------------------------------------------------------------------
#  Orchestrators
# ------------------------------------------------------------------

def run_inference(
    df: pd.DataFrame,
    config_path: str = DEFAULT_CONFIG,
    granularity: str = "hour",
    parent_run_id: str | None = None,
) -> pd.Series:
    """Inference on a feature-engineered DataFrame.

    Expects *df* to be the output of ``etl_from_raw()`` or equivalent
    (i.e. features_ready.csv level, with reservation_hour present for routing).

    Returns:
        Series of predicted labels indexed like *df*.
        Rows that cannot be routed will be absent from the result.
    """
    config = load_config(config_path)
    target, leakage = resolve_granularity(config, granularity)

    df = fill_missing(df, strategy="median")

    nested_runs = find_nested_runs(
        "xgboost", parent_run_id,
        parent_run_name_contains=f"xgb_{granularity}",
    )

    routed = route_to_subsets(df, config["subsets"])

    predictions: list[pd.Series] = []
    for subset_name, subset_df in routed.items():
        if subset_name == "_unmatched" or len(subset_df) == 0:
            continue
        if subset_name not in nested_runs:
            logger.warning(
                "No model found for subset '%s' — %d rows skipped",
                subset_name, len(subset_df),
            )
            continue
        run_id = nested_runs[subset_name]
        logger.info("Predicting '%s' (%d rows) with run %s", subset_name, len(subset_df), run_id)
        preds = predict_subset(subset_df, run_id, target, leakage)
        predictions.append(preds)

    all_preds = pd.concat(predictions).sort_index() if predictions else pd.Series(dtype=float)

    n_missing = len(df) - len(all_preds)
    if n_missing > 0:
        logger.warning("%d rows have no prediction (unmatched or missing model)", n_missing)

    return all_preds


def run_inference_from_raw(
    test_csv_path: str | Path,
    restaurant_csv_path: str | Path = DEFAULT_RESTAURANT_CSV,
    member_csv_path: str | Path = DEFAULT_MEMBER_CSV,
    config_path: str = DEFAULT_CONFIG,
    granularity: str = "hour",
    parent_run_id: str | None = None,
) -> pd.Series:
    """Full pipeline from raw test.csv to predictions.

    Combines ETL + preprocessing + model inference in one call.

    Steps:
        1. clean_order(test.csv)
        2. merge with restaurant_silver + member_silver
        3. transform_features()
        4. fill_missing (median)
        5. route rows to subset models by reservation_hour
        6. drop non-model columns (datetime, free-text, leakage, target)
        7. apply_saved_encoders (OrdinalEncoder on CATEGORICAL_COLS)
        8. model.predict → inverse_transform → original labels

    Args:
        test_csv_path: Path to raw test CSV (tab-separated, UTF-8).
        restaurant_csv_path: Path to restaurant_silver.csv (default: data/interim/).
        member_csv_path: Path to member_silver.csv (default: data/interim/).
        config_path: Path to xgboost.yaml config.
        granularity: ``"hour"`` or ``"half_hour"``.
        parent_run_id: MLflow parent run ID. If None, uses the latest finished run.

    Returns:
        pd.Series of predicted labels, indexed by the original DataFrame index.
    """
    df_features = etl_from_raw(test_csv_path, restaurant_csv_path, member_csv_path)
    return run_inference(df_features, config_path, granularity, parent_run_id)


# ------------------------------------------------------------------
#  CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="XGBoost inference (三分法)")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--raw",
        help="Path to raw test CSV (tab-separated). Runs full ETL pipeline.",
    )
    input_group.add_argument(
        "--input",
        help="Path to pre-processed input JSON file.",
    )

    parser.add_argument(
        "--granularity", required=True, choices=["hour", "half_hour"],
        help="Prediction granularity: 'hour' or 'half_hour'.",
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--restaurant-csv", default=DEFAULT_RESTAURANT_CSV,
        help=f"Path to restaurant_silver.csv (default: {DEFAULT_RESTAURANT_CSV}). Used with --raw.",
    )
    parser.add_argument(
        "--member-csv", default=DEFAULT_MEMBER_CSV,
        help=f"Path to member_silver.csv (default: {DEFAULT_MEMBER_CSV}). Used with --raw.",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="MLflow parent run ID. If omitted, uses the latest finished run.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save output JSON. If omitted, prints to stdout.",
    )
    args = parser.parse_args()

    if args.raw:
        predictions = run_inference_from_raw(
            test_csv_path=args.raw,
            restaurant_csv_path=args.restaurant_csv,
            member_csv_path=args.member_csv,
            config_path=args.config,
            granularity=args.granularity,
            parent_run_id=args.run_id,
        )
    else:
        df = parse_json_input(args.input)
        logger.info("Loaded %d rows from %s", len(df), args.input)
        predictions = run_inference(
            df,
            config_path=args.config,
            granularity=args.granularity,
            parent_run_id=args.run_id,
        )

    output = build_json_output(predictions)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info("Saved predictions to %s", args.output)
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
